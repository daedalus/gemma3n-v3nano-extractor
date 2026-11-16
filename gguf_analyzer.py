#!/usr/bin/env python3
"""Compare two GGUF files tensor-by-tensor (streaming, mmap-backed).

Usage:
    python gguf_analyzer.py ours.gguf reference.gguf \
        [--sample 262144] [--seed 0] [--filter REGEX] [--worst 15] \
        [--json-out out.json] [--txt-out out.txt] [--fail-mse 1e-3]

What it reports:
  * KV metadata differences (keys only in one file, and differing values)
  * tensor inventory (missing tensors, shape and dtype mismatches)
  * per-tensor value statistics on a deterministic random sample of elements:
    MSE, RMSE, relative RMSE, max |diff|, cosine similarity, exact-match
    fraction and non-finite counts
  * aggregates by tensor family (layer-agnostic, e.g. ``blk.*.attn_q.weight``)
    and by layer index
  * a provenance hint per tensor: ``exact`` (bit-identical on the sample),
    ``ref-quant`` (equal to the reference up to its own quantization/rounding
    noise) or ``differs``

Quantized tensors are decoded block-by-block: only the blocks containing a
sampled element are gathered and dequantized, so peak memory stays bounded
regardless of tensor size.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import gguf


QT = gguf.GGMLQuantizationType
Q_SIZES = gguf.quants.GGML_QUANT_SIZES

PLAIN_DTYPES = {
    QT.F32: np.dtype("<f4"),
    QT.F16: np.dtype("<f2"),
    QT.BF16: np.dtype("<u2"),  # decoded specially
}

# Max quantized blocks dequantized per batch (bounds peak memory).
BLOCK_BATCH = 4096


# --------------------------------------------------------------------------
# KV helpers
# --------------------------------------------------------------------------

def kv_value(field):
    """Normalize a GGUFReader field into a comparable Python value."""
    try:
        v = field.contents()
    except Exception:  # noqa: BLE001 - older/odd field types
        v = [p.tolist() for p in field.parts]
    if isinstance(v, np.ndarray):
        v = v.item() if v.ndim == 0 else v.tolist()
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v)
    return v


def kv_type_name(field):
    try:
        return "/".join(t.name for t in field.types) if field.types else "?"
    except Exception:  # noqa: BLE001
        return "?"


def _short(v, limit=70):
    if isinstance(v, list) and len(v) > 8:
        s = f"[{len(v)} items] " + repr(v[:4])[:-1] + ", ...]"
    else:
        s = repr(v)
    return s if len(s) <= limit else s[:limit] + "..."


def kv_compare(ra, rb):
    ka = {k: kv_value(v) for k, v in ra.fields.items() if not k.startswith("GGUF.")}
    kb = {k: kv_value(v) for k, v in rb.fields.items() if not k.startswith("GGUF.")}
    only_ours = sorted(set(ka) - set(kb))
    only_ref = sorted(set(kb) - set(ka))
    differing = []
    for key in sorted(set(ka) & set(kb)):
        va, vb = ka[key], kb[key]
        if va != vb:
            differing.append({
                "key": key,
                "ours": _short(va),
                "ref": _short(vb),
                "types": [kv_type_name(ra.fields[key]), kv_type_name(rb.fields[key])],
            })
    return only_ours, only_ref, differing


# --------------------------------------------------------------------------
# Tensor value access
# --------------------------------------------------------------------------

def values_at(buf, t, idx):
    """Return decoded float32 values of tensor t at sorted flat indices idx."""
    dtype = PLAIN_DTYPES.get(t.tensor_type)
    if dtype is not None:
        arr = np.frombuffer(buf, dtype=dtype, count=t.n_elements, offset=t.data_offset)
        vals = arr[idx]
        if dtype == np.dtype("<u2"):  # BF16 -> F32
            return (vals.astype(np.uint32) << 16).view(np.float32)
        return vals.astype(np.float32)

    # Quantized: gather only the blocks holding sampled elements, then
    # dequantize those blocks (never the spans between them).
    bs, ts = Q_SIZES[t.tensor_type]
    n_blocks = t.n_elements // bs
    raw = np.frombuffer(buf, dtype=np.uint8, count=n_blocks * ts,
                        offset=t.data_offset).reshape(n_blocks, ts)
    blk = idx // bs
    blocks, inv = np.unique(blk, return_inverse=True)
    within = idx % bs
    out = np.empty(len(idx), dtype=np.float32)
    for s in range(0, len(blocks), BLOCK_BATCH):
        sel_blocks = blocks[s: s + BLOCK_BATCH]
        dec = gguf.quants.dequantize(np.ascontiguousarray(raw[sel_blocks]),
                                     t.tensor_type).reshape(len(sel_blocks), bs)
        mask = (inv >= s) & (inv < s + len(sel_blocks))
        out[mask] = dec[inv[mask] - s, within[mask]]
    return out


def drop_pages(buf):
    """Release resident clean pages of a file mapping (RSS stays flat)."""
    mm = getattr(buf, "_mmap", None)
    if mm is not None and hasattr(mm, "madvise"):
        try:
            mm.madvise(mmap.MADV_DONTNEED)
        except OSError:
            pass


def sample_indices(n, sample, rng):
    """Deterministic, sorted, alias-free sample of flat indices."""
    if n <= sample:
        return np.arange(n, dtype=np.int64)
    idx = rng.choice(n, size=sample, replace=False)
    idx.sort()
    return idx.astype(np.int64)


# Noise floor of a reference stored as F16/BF16/quantized: if our values agree
# with it at a relative RMSE below this, we call it "equal up to reference
# rounding/quantization" rather than a real difference.
REF_NOISE_REL = {
    "F32": 1e-6, "F16": 1e-3, "BF16": 8e-3,
    "Q8_0": 1.5e-2, "Q8_1": 1.5e-2, "Q6_K": 3e-2, "Q5_K": 6e-2, "Q5_0": 6e-2,
    "Q5_1": 6e-2, "Q4_K": 1.2e-1, "Q4_0": 1.4e-1, "Q4_1": 1.4e-1,
    "IQ4_NL": 1.4e-1, "IQ4_XS": 1.4e-1,
}


def compare_tensor(buf_a, ta, buf_b, tb, idx):
    a = values_at(buf_a, ta, idx).astype(np.float64)
    b = values_at(buf_b, tb, idx).astype(np.float64)
    fin_a, fin_b = np.isfinite(a), np.isfinite(b)
    ok = fin_a & fin_b
    a_ok, b_ok = a[ok], b[ok]
    d = a_ok - b_ok
    mse = float(np.mean(d * d)) if d.size else 0.0
    rms_b = float(np.sqrt(np.mean(b_ok * b_ok))) if b_ok.size else 0.0
    na, nb = float(np.linalg.norm(a_ok)), float(np.linalg.norm(b_ok))
    if na > 0 and nb > 0:
        cos = float(np.dot(a_ok, b_ok) / (na * nb))
    else:
        cos = 1.0 if na == nb else 0.0
    rmse = float(np.sqrt(mse))
    rel = rmse / rms_b if rms_b > 0 else (0.0 if rmse == 0 else None)
    exact = float(np.mean(a == b)) if a.size else 1.0

    if exact >= 1.0:
        prov = "exact"
    elif rel is not None and rel <= (REF_NOISE_REL.get(ta.tensor_type.name, 0.15)
                                     + REF_NOISE_REL.get(tb.tensor_type.name, 0.15)):
        prov = "ref-quant"
    else:
        prov = "differs"

    return {
        "n": int(ta.n_elements),
        "sampled": int(len(idx)),
        "mse": mse,
        "rmse": rmse,
        "rms_ref": rms_b,
        "rms_ours": float(np.sqrt(np.mean(a_ok * a_ok))) if a_ok.size else 0.0,
        "mean_ours": float(np.mean(a_ok)) if a_ok.size else 0.0,
        "mean_ref": float(np.mean(b_ok)) if b_ok.size else 0.0,
        "rel_rmse": rel,
        "cosine": cos,
        "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
        "exact_frac": exact,
        "nonfinite_ours": int((~fin_a).sum()),
        "nonfinite_ref": int((~fin_b).sum()),
        "provenance": prov,
    }


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.*)$")


def family_of(name: str) -> str:
    """Layer-agnostic family: 'blk.12.attn_q.weight' -> 'blk.*.attn_q.weight'."""
    m = _BLK_RE.match(name)
    return f"blk.*.{m.group(2)}" if m else name


def layer_of(name: str):
    m = _BLK_RE.match(name)
    return int(m.group(1)) if m else None


def aggregate(tensors):
    mses = np.array([t["mse"] for t in tensors])
    rels = [t["rel_rmse"] for t in tensors if t["rel_rmse"] is not None]
    w = np.array([t["n"] for t in tensors], dtype=np.float64)
    return {
        "count": len(tensors),
        "total_elements": int(w.sum()),
        "weighted_mse": float(np.average(mses, weights=w)) if w.sum() else 0.0,
        "mean_mse": float(mses.mean()),
        "median_mse": float(np.median(mses)),
        "max_mse": float(mses.max()),
        "max_rel_rmse": float(max(rels)) if rels else 0.0,
        "min_cosine": float(min(t["cosine"] for t in tensors)),
        "provenance": dict(Counter(t["provenance"] for t in tensors)),
    }


def _fmt_rel(v):
    return f"{v:.2e}" if v is not None else "n/a"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ours", help="GGUF under test")
    ap.add_argument("reference", help="reference GGUF")
    ap.add_argument("--sample", type=int, default=1 << 18,
                    help="max elements sampled per tensor (default 262144)")
    ap.add_argument("--seed", type=int, default=0, help="sampling RNG seed")
    ap.add_argument("--filter", default=None,
                    help="only compare tensors whose name matches this regex")
    ap.add_argument("--json-out", default="work/json/gguf_analysis.json")
    ap.add_argument("--txt-out", default="work/txt/gguf_analysis.txt")
    ap.add_argument("--worst", type=int, default=15)
    ap.add_argument("--fail-mse", type=float, default=None,
                    help="exit 1 if any tensor's sampled MSE exceeds this, or if "
                         "the inventory differs")
    args = ap.parse_args(argv)
    t0 = time.time()

    ra = gguf.GGUFReader(args.ours)
    rb = gguf.GGUFReader(args.reference)
    for path, r in ((args.ours, ra), (args.reference, rb)):
        if not isinstance(r.data, (mmap.mmap, np.memmap)):
            raise SystemExit(f"{path}: GGUFReader did not provide a memory map")
    buf_a, buf_b = ra.data, rb.data

    # ---- KV ----------------------------------------------------------------
    kv_only_ours, kv_only_ref, kv_diffs = kv_compare(ra, rb)

    # ---- inventory -----------------------------------------------------------
    ta_map = {t.name: t for t in ra.tensors}
    tb_map = {t.name: t for t in rb.tensors}
    only_ours = sorted(set(ta_map) - set(tb_map))
    only_ref = sorted(set(tb_map) - set(ta_map))
    shape_mismatch, type_mismatch = [], []
    for name in sorted(set(ta_map) & set(tb_map)):
        a, b = ta_map[name], tb_map[name]
        if list(a.shape) != list(b.shape):
            shape_mismatch.append({"name": name, "ours": [int(x) for x in a.shape],
                                   "ref": [int(x) for x in b.shape]})
        if a.tensor_type != b.tensor_type:
            type_mismatch.append({"name": name, "ours": a.tensor_type.name,
                                  "ref": b.tensor_type.name})
    type_pairs = Counter(f"{d['ours']}<-{d['ref']}" for d in type_mismatch)

    # ---- values --------------------------------------------------------------
    shape_bad = {d["name"] for d in shape_mismatch}
    flt = re.compile(args.filter) if args.filter else None
    common = sorted((n for n in set(ta_map) & set(tb_map)
                     if n not in shape_bad and (flt is None or flt.search(n))),
                    key=lambda n: -ta_map[n].n_elements)
    rng = np.random.default_rng(args.seed)
    per_tensor = []
    for i, name in enumerate(common):
        a, b = ta_map[name], tb_map[name]
        idx = sample_indices(int(a.n_elements), args.sample, rng)
        st = compare_tensor(buf_a, a, buf_b, b, idx)
        st["name"] = name
        st["shape"] = [int(x) for x in a.shape]
        st["type_ours"] = a.tensor_type.name
        st["type_ref"] = b.tensor_type.name
        per_tensor.append(st)
        drop_pages(buf_a)
        drop_pages(buf_b)
        if (i + 1) % 100 == 0 or i + 1 == len(common):
            print(f"  ...{i + 1}/{len(common)} tensors ({time.time() - t0:.0f}s)",
                  file=sys.stderr)
    if not per_tensor:
        raise SystemExit("no comparable tensors")

    summary = aggregate(per_tensor)
    rels = [t["rel_rmse"] for t in per_tensor if t["rel_rmse"] is not None]
    mses = np.array([t["mse"] for t in per_tensor])
    summary.update({
        "median_rel_rmse": float(np.median(rels)) if rels else 0.0,
        "buckets_mse": {f"<{e:g}": int(np.sum(mses < e))
                        for e in (1e-12, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1)},
        "exact_tensors": sum(t["exact_frac"] >= 1.0 for t in per_tensor),
        "equal_frac_weighted": float(np.average(
            [t["exact_frac"] for t in per_tensor],
            weights=[t["n"] for t in per_tensor])),
        "nonfinite_tensors_ours": [t["name"] for t in per_tensor if t["nonfinite_ours"]],
        "nonfinite_tensors_ref": [t["name"] for t in per_tensor if t["nonfinite_ref"]],
    })

    families = defaultdict(list)
    layers = defaultdict(list)
    for t in per_tensor:
        families[family_of(t["name"])].append(t)
        ly = layer_of(t["name"])
        if ly is not None:
            layers[ly].append(t)
    fam_stats = {f: aggregate(ts) for f, ts in sorted(families.items())}
    layer_stats = {str(ly): aggregate(ts) for ly, ts in sorted(layers.items())}

    worst = sorted(per_tensor, key=lambda t: -t["mse"])[:args.worst]
    worst_rel = sorted(per_tensor, key=lambda t: -(t["rel_rmse"] or 0.0))[:args.worst]
    worst_cos = sorted(per_tensor, key=lambda t: t["cosine"])[:args.worst]

    # ---- text report -----------------------------------------------------------
    L = []
    L.append(f"ours      : {args.ours}")
    L.append(f"reference : {args.reference}")
    L.append(f"  file size    : {os.path.getsize(args.ours):,} vs "
             f"{os.path.getsize(args.reference):,} bytes")
    L.append(f"  kv entries   : {len(ra.fields)} vs {len(rb.fields)}")
    L.append(f"  tensors      : {len(ra.tensors)} vs {len(rb.tensors)}")
    L.append(f"  data_offset  : {ra.data_offset} vs {rb.data_offset}")
    L.append(f"  types (ours) : {dict(Counter(t.tensor_type.name for t in ra.tensors))}")
    L.append(f"  types (ref)  : {dict(Counter(t.tensor_type.name for t in rb.tensors))}")
    L.append("")
    L.append("KV metadata:")
    L.append(f"  only ours: {kv_only_ours or 'none'}")
    L.append(f"  only ref : {kv_only_ref or 'none'}")
    if not kv_diffs:
        L.append("  differing values: none")
    for d in kv_diffs:
        L.append(f"  {d['key']}: ours={d['ours']}  ref={d['ref']}")
    L.append("")
    L.append("Inventory:")
    L.append(f"  only ours       : {len(only_ours)} {only_ours[:5]}"
             + (" ..." if len(only_ours) > 5 else ""))
    L.append(f"  only ref        : {len(only_ref)} {only_ref[:5]}"
             + (" ..." if len(only_ref) > 5 else ""))
    L.append(f"  shape mismatches: {len(shape_mismatch)} {shape_mismatch[:3]}")
    L.append(f"  dtype mismatches: {len(type_mismatch)} {dict(type_pairs)}")
    L.append("")
    L.append(f"Values ({len(per_tensor)} tensors, <= {args.sample} random samples "
             f"each, seed {args.seed}):")
    L.append(f"  element-weighted MSE : {summary['weighted_mse']:.3e}")
    L.append(f"  mean / median MSE    : {summary['mean_mse']:.3e} / {summary['median_mse']:.3e}")
    L.append(f"  max MSE              : {summary['max_mse']:.3e}")
    L.append(f"  median / max rel RMSE: {summary['median_rel_rmse']:.3e} / "
             f"{summary['max_rel_rmse']:.3e}")
    L.append(f"  min cosine           : {summary['min_cosine']:.6f}")
    L.append(f"  bit-exact tensors    : {summary['exact_tensors']} "
             f"(element-weighted equal fraction {summary['equal_frac_weighted']:.6f})")
    L.append(f"  provenance           : {summary['provenance']}")
    L.append(f"  non-finite           : ours={len(summary['nonfinite_tensors_ours'])} "
             f"ref={len(summary['nonfinite_tensors_ref'])} tensors")
    L.append(f"  MSE buckets          : {summary['buckets_mse']}")
    L.append("")
    hdr = f"  {'name':44s} {'n_elem':>11s} {'mse':>9s} {'rel':>9s} {'cos':>9s} {'max_abs':>9s}  types / prov"

    def row(t):
        return (f"  {t['name']:44s} {t['n']:11d} {t['mse']:9.2e} {_fmt_rel(t['rel_rmse']):>9s} "
                f"{t['cosine']:9.6f} {t['max_abs']:9.2e}  {t['type_ours']}/{t['type_ref']} "
                f"{t['provenance']}")
    for title, lst in (("by MSE", worst), ("by rel RMSE", worst_rel),
                       ("by cosine", worst_cos)):
        L.append(f"Worst {len(lst)} tensors {title}:")
        L.append(hdr)
        L.extend(row(t) for t in lst)
        L.append("")
    L.append("Families (layer-agnostic):")
    L.append(f"  {'family':40s} {'n':>4s} {'w_mse':>9s} {'max_mse':>9s} {'max_rel':>9s} "
             f"{'min_cos':>9s}  provenance")
    for f, s in fam_stats.items():
        L.append(f"  {f:40s} {s['count']:4d} {s['weighted_mse']:9.2e} {s['max_mse']:9.2e} "
                 f"{s['max_rel_rmse']:9.2e} {s['min_cosine']:9.6f}  {s['provenance']}")
    L.append("")
    L.append("Layers:")
    L.append(f"  {'layer':>5s} {'n':>4s} {'w_mse':>9s} {'max_mse':>9s} {'max_rel':>9s} {'min_cos':>9s}")
    for ly, s in layer_stats.items():
        L.append(f"  {ly:>5s} {s['count']:4d} {s['weighted_mse']:9.2e} {s['max_mse']:9.2e} "
                 f"{s['max_rel_rmse']:9.2e} {s['min_cosine']:9.6f}")
    L.append("")

    failures = []
    if args.fail_mse is not None:
        failures = [t["name"] for t in per_tensor if t["mse"] > args.fail_mse]
        if only_ours or only_ref or shape_mismatch:
            failures.append("<inventory mismatch>")
        L.append(f"Gate: MSE <= {args.fail_mse:g}: "
                 + ("PASS" if not failures else f"FAIL ({len(failures)}: {failures[:5]})"))
    L.append(f"elapsed: {time.time() - t0:.1f}s")
    text = "\n".join(L)
    print(text)

    # ---- outputs ---------------------------------------------------------------
    report = {
        "ours": {"path": args.ours, "kv": len(ra.fields), "tensors": len(ra.tensors),
                 "data_offset": int(ra.data_offset)},
        "reference": {"path": args.reference, "kv": len(rb.fields),
                      "tensors": len(rb.tensors), "data_offset": int(rb.data_offset)},
        "settings": {"sample": args.sample, "seed": args.seed, "filter": args.filter},
        "kv": {"only_ours": kv_only_ours, "only_ref": kv_only_ref, "differing": kv_diffs},
        "inventory": {
            "only_ours": only_ours, "only_ref": only_ref,
            "shape_mismatch": shape_mismatch, "type_mismatch": type_mismatch,
            "type_counts_ours": dict(Counter(t.tensor_type.name for t in ra.tensors)),
            "type_counts_ref": dict(Counter(t.tensor_type.name for t in rb.tensors)),
        },
        "values": summary,
        "families": fam_stats,
        "layers": layer_stats,
        "per_tensor": per_tensor,
        "gate": {"fail_mse": args.fail_mse, "failures": failures},
    }
    for path, payload in ((args.json_out, json.dumps(report, indent=1)),
                          (args.txt_out, text + "\n")):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                f.write(payload)
    print(f"\nJSON report: {args.json_out}\nText report: {args.txt_out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
