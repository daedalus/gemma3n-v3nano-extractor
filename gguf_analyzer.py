#!/usr/bin/env python3
"""Consolidated GGUF analyzer that combines diff and analysis functionality.

Usage:
    python gguf_analyzer.py ours.gguf reference.gguf [--sample 262144] [--json-out out.json] [--worst 20] [--txt-out out.txt]

This script combines:
- diff_gguf.py: tensor-by-tensor diff with mmap streaming
- analyze_tensors.py: post-processing of diff results
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import sys
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


def kv_value(field):
    """Normalize a GGUFReader field into a comparable Python value."""
    try:
        v = field.contents()
    except Exception:  # noqa: BLE001 - older/odd field types
        v = field.parts
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            v = v.item()
        else:
            v = v.tolist()
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v)
    if isinstance(v, list):
        v = [bytes(x) if isinstance(x, (bytearray,)) else x for x in v]
    return v


def kv_type_name(field):
    try:
        return field.types[0].name if field.types else "?"
    except Exception:  # noqa: BLE001
        return "?"


def tensor_nbytes(t) -> int:
    bs, ts = Q_SIZES[t.tensor_type]
    return (t.n_elements // bs) * ts


def raw_view(buf, dtype, count, offset):
    """Read-only numpy view over the mmap, no copy."""
    arr = np.frombuffer(buf, dtype=dtype, count=count, offset=offset)
    if dtype == np.dtype("<u2"):  # BF16 -> float32
        return (arr.astype(np.uint32) << 16).view(np.float32)
    return arr


def values_at(buf, t, idx):
    """Return decoded float32 values of tensor t at flat indices idx."""
    dtype = PLAIN_DTYPES.get(t.tensor_type)
    if dtype is not None:
        arr = raw_view(buf, dtype, t.n_elements, t.data_offset)
        return arr[idx].astype(np.float32)

    # Quantized: decode the sampled blocks in bounded batches (never the whole
    # tensor, which for token_embd/per_layer is billions of elements).
    bs, ts = Q_SIZES[t.tensor_type]
    blocks = np.unique(idx // bs)
    blk_of_idx = idx // bs
    out = np.empty(len(idx), dtype=np.float32)
    batch = max(1, (1 << 16) // bs)
    for s in range(0, len(blocks), batch):
        bl = blocks[s: s + batch]
        b0, b1 = int(bl[0]), int(bl[-1])
        span = b1 - b0 + 1
        raw = np.frombuffer(buf, dtype=np.uint8, count=span * ts,
                            offset=t.data_offset + b0 * ts)
        dec = gguf.quants.dequantize(raw, t.tensor_type).astype(np.float32)
        sel = (blk_of_idx >= b0) & (blk_of_idx <= b1)
        out[sel] = dec[idx[sel] - b0 * bs]
    return out


def drop_pages(buf):
    """Release resident clean pages of a file mapping (RSS stays flat)."""
    mm = getattr(buf, "_mmap", None)
    if mm is not None and hasattr(mm, "madvise"):
        try:
            mm.madvise(mmap.MADV_DONTNEED)
        except OSError:
            pass


def compare_tensor(buf_a, ta, buf_b, tb, sample):
    n = min(ta.n_elements, tb.n_elements)
    if n <= sample:
        idx = np.arange(n, dtype=np.int64)
    else:
        idx = np.linspace(0, n - 1, sample).astype(np.int64)
    a = values_at(buf_a, ta, idx).astype(np.float64)
    b = values_at(buf_b, tb, idx).astype(np.float64)
    d = a - b
    mse = float(np.mean(d * d))
    rms_b = float(np.sqrt(np.mean(b * b)))
    return {
        "n": int(ta.n_elements),
        "sampled": int(len(idx)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "rms_ref": rms_b,
        "rel_rmse": (float(np.sqrt(mse)) / rms_b) if rms_b > 0 else None,
        "max_abs": float(np.max(np.abs(d))) if len(d) else 0.0,
        "exact_frac": float(np.mean(a == b)) if len(d) else 1.0,
    }


def _type_counts(tensors):
    return dict(Counter(t.tensor_type.name for t in tensors))


def _short(v, limit=70):
    s = repr(v)
    return s if len(s) <= limit else s[:limit] + "..."


def family_of(name: str) -> str:
    """Map tensor name -> family group."""
    if name == "per_layer_token_embd.weight":
        return "per_layer_token_embd"
    if name == "token_embd.weight":
        return "token_embd"
    m = re.match(r"^blk\.(\d+)\.(.*)$", name)
    if m:
        blk = m.group(1)
        rest = m.group(2)
        # Split off block-level special components
        for prefix in (
            "altup_correct_coef.weight", "altup_correct_scale.weight",
            "altup_predict_coef.weight", "altup_router_norm.weight",
            "attn_k_norm.weight", "attn_q_norm.weight", "attn_v_norm.weight",
            "attn_norm.weight", "ffn_norm.weight", "laurel_post_norm.weight",
            "laurel_r.weight", "post_attention_norm.weight",
            "post_ffw_norm.weight", "post_norm.weight", "altup_router.weight",
            "attn_k.weight", "attn_q.weight", "attn_v.weight",
            "attn_output.weight", "attn_down.weight", "attn_gate.weight",
            "attn_up.weight", "attn_gate2.weight", "attn_up2.weight",
            "attn_proj.weight", "inp_gate.weight",
            "ffn_down.weight", "ffn_gate.weight", "ffn_up.weight",
        ):
            if rest == prefix or rest.startswith(prefix):
                return f"blk.{blk}.{prefix}"
        return rest
    return name


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="GGUF diff and analysis tool - combines diff and post-processing"
    )
    ap.add_argument("ours", help="Path to our GGUF file")
    ap.add_argument("reference", help="Path to reference GGUF file")
    ap.add_argument("--sample", type=int, default=1 << 18,
                    help="max elements sampled per tensor (default 262144)")
    ap.add_argument("--json-out", default="work/json/gguf_analysis.json")
    ap.add_argument("--txt-out", default="work/txt/gguf_analysis.txt")
    ap.add_argument("--worst", type=int, default=15)
    args = ap.parse_args(argv)

    # Read and parse GGUF files
    ra = gguf.GGUFReader(args.ours)
    rb = gguf.GGUFReader(args.reference)
    
    # Validate mmap availability
    for path, r in ((args.ours, ra), (args.reference, rb)):
        if not isinstance(r.data, (mmap.mmap, np.memmap)):
            raise SystemExit(f"{path}: GGUFReader did not provide a memory map")
    
    buf_a, buf_b = ra.data, rb.data

    # KV diff
    ka = {k: kv_value(v) for k, v in ra.fields.items()}
    kb = {k: kv_value(v) for k, v in rb.fields.items()}
    kv_diffs = []
    for key in sorted(set(ka) | set(kb)):
        if key.startswith("GGUF."):
            continue
        va = ka.get(key, "<missing>")
        vb = kb.get(key, "<missing>")
        if isinstance(va, list) or isinstance(vb, list):
            same = va == vb
        else:
            same = va == vb
        if not same:
            kv_diffs.append({
                "key": key,
                "ours": _short(va),
                "ref": _short(vb),
                "types": [kv_type_name(ra.fields[key]) if key in ra.fields else "-",
                          kv_type_name(rb.fields[key]) if key in rb.fields else "-"],
            })

    # Inventory comparison
    ta_map = {t.name: t for t in ra.tensors}
    tb_map = {t.name: t for t in rb.tensors}
    only_ours = sorted(set(ta_map) - set(tb_map))
    only_ref = sorted(set(tb_map) - set(ta_map))
    shape_mismatch, type_mismatch = [], []
    for name in sorted(set(ta_map) & set(tb_map)):
        a, b = ta_map[name], tb_map[name]
        if list(a.shape) != list(b.shape):
            shape_mismatch.append({"name": name, "ours": list(a.shape), "ref": list(b.shape)})
        if a.tensor_type != b.tensor_type:
            type_mismatch.append({"name": name, "ours": a.tensor_type.name,
                                 "ref": b.tensor_type.name})

    # Value comparison
    per_tensor = []
    common = sorted(set(ta_map) & set(tb_map),
                    key=lambda n: -ta_map[n].n_elements)
    for i, name in enumerate(common):
        a, b = ta_map[name], tb_map[name]
        if list(a.shape) != list(b.shape):
            continue
        st = compare_tensor(buf_a, a, buf_b, b, args.sample)
        st["name"] = name
        st["type_ours"] = a.tensor_type.name
        st["type_ref"] = b.tensor_type.name
        per_tensor.append(st)
        drop_pages(buf_a)
        drop_pages(buf_b)
        if os.environ.get("DIFF_RSS_LOG"):
            try:
                cur = int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**20
                print(f"  [rss {cur:.0f} MiB] {i + 1}/{len(common)}", file=sys.stderr)
            except OSError:
                pass
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(common)} tensors", file=sys.stderr)

    # Values comparison
    mses = np.array([t["mse"] for t in per_tensor], dtype=np.float64)
    rels = np.array([t["rel_rmse"] if t["rel_rmse"] is not None else 0.0
                     for t in per_tensor], dtype=np.float64)
    weighted_mse = float(np.average(mses, weights=[t["n"] for t in per_tensor]))
    mean_mse = float(np.mean(mses))
    median_mse = float(np.median(mses))
    max_mse = float(np.max(mses))
    median_rel_rmse = float(np.median([t["rel_rmse"] for t in per_tensor if t["rel_rmse"] is not None] or [0.0]))
    max_rel_rmse = float(np.max([t["rel_rmse"] for t in per_tensor if t["rel_rmse"] is not None] or [0.0]))
    
    buckets_mse = {
        f"<{e:g}": int(np.sum(mses < e)) for e in (1e-9, 1e-7, 1e-5, 1e-3, 1e-1)
    }
    exact_tensors = int(np.sum([t["exact_frac"] >= 1.0 for t in per_tensor]))
    fully_equal_frac = float(np.average([t["exact_frac"] for t in per_tensor],
                                      weights=[t["n"] for t in per_tensor]))
    
    # Build family groups
    family_groups = defaultdict(list)
    for t in per_tensor:
        fam = family_of(t["name"])
        family_groups[fam].append(t)

    # Print summary
    o, r = ra, rb
    print(f"ours      : {args.ours}")
    print(f"reference : {args.reference}")
    print(f"  kv entries   : {len(ra.fields)} vs {len(rb.fields)}")
    print(f"  tensors      : {len(ra.tensors)} vs {len(rb.tensors)}")
    print(f"  data_offset  : {ra.data_offset} vs {rb.data_offset}")
    print(f"  mmap         : {type(ra.data).__name__} vs {type(rb.data).__name__}")
    print(f"  types        : {_type_counts(ra.tensors)} vs "
          f"{_type_counts(rb.tensors)}")
    print()
    print("KV differences:")
    if not kv_diffs:
        print("  none")
    else:
        for d in kv_diffs:
            print(f"  {d['key']}: ours={_short(d['ours'])}  ref={_short(d['ref'])} "
                  f"({d['types'][0]}/{d['types'][1]})")
    print()
    print("Inventory:")
    print(f"  only ours: {len(only_ours)} {only_ours[:5]}"
          + (" ..." if len(only_ours) > 5 else ""))
    print(f"  only ref : {len(only_ref)} {only_ref[:5]}"
          + (" ..." if len(only_ref) > 5 else ""))
    print(f"  shape mismatches: {len(shape_mismatch)} {shape_mismatch[:3]}")
    print(f"  dtype mismatches: {len(type_mismatch)}")
    for tm in type_mismatch[:10]:
        print(f"    {tm['name']}: ours={tm['ours']} ref={tm['ref']}")
    print()
    print(f"Values ({len(per_tensor)} tensors, <= {args.sample} samples each):")
    print(f"  weighted MSE   : {weighted_mse:.3e}")
    print(f"  mean/median MSE: {mean_mse:.3e} / {median_mse:.3e}")
    print(f"  max MSE        : {max_mse:.3e}")
    print(f"  median rel RMSE: {median_rel_rmse:.3e}")
    print(f"  max rel RMSE   : {max_rel_rmse:.3e}")
    print(f"  tensors bit-exact: {len([t for t in per_tensor if t['exact_frac'] >= 1.0])} "
          f"(element-weighted equal fraction {fully_equal_frac:.6f})")
    print(f"  MSE buckets    : {buckets_mse}")
    print()
    worst = sorted(per_tensor, key=lambda t: -t["mse"])[:args.worst]
    print(f"Worst {len(worst)} tensors by MSE:")
    for t in worst:
        rel = f"{t['rel_rmse']:.2e}" if t["rel_rmse"] is not None else "n/a"
        print(f"  {t['name']:44s} {t['n']:11d} {t['mse']:10.2e} {rel:>10s} "
              f"{t['max_abs']:10.2e}  {t['type_ours']}/{t['type_ref']}")
    print()
    best = sorted(per_tensor, key=lambda t: t["mse"])[: min(8, len(per_tensor))]
    print(f"Best {len(best)} tensors by MSE:")
    for t in best:
        rel = f"{t['rel_rmse']:.2e}" if t["rel_rmse"] is not None else "n/a"
        print(f"  {t['name']:44s} {t['n']:11d} {t['mse']:10.2e} {rel:>10s}")

    # Save JSON report
    report = {
        "ours": {
            "path": args.ours,
            "kv": len(ra.fields),
            "tensors": len(ra.tensors),
            "data_offset": ra.data_offset,
            "mmap": type(ra.data).__name__,
        },
        "reference": {
            "path": args.reference,
            "kv": len(rb.fields),
            "tensors": len(rb.tensors),
            "data_offset": rb.data_offset,
            "mmap": type(rb.data).__name__,
        },
        "kv": {
            "only_ours": only_ours,
            "only_ref": only_ref,
            "differing": kv_diffs,
            "shape_mismatch": shape_mismatch,
            "type_mismatch": type_mismatch,
        },
        "inventory": {
            "only_ours": only_ours,
            "only_ref": only_ref,
            "shape_mismatch": shape_mismatch,
            "type_mismatch": type_mismatch,
            "type_counts_ours": _type_counts(ra.tensors),
            "type_counts_ref": _type_counts(rb.tensors),
        },
        "values": {
            "tensors_compared": len(per_tensor),
            "weighted_mse": weighted_mse,
            "mean_mse": mean_mse,
            "median_mse": median_mse,
            "max_mse": max_mse,
            "median_rel_rmse": median_rel_rmse,
            "max_rel_rmse": max_rel_rmse,
            "buckets_mse": buckets_mse,
            "exact_tensors": exact_tensors,
            "fully_equal_frac": fully_equal_frac,
        },
        "per_tensor": per_tensor,
        "family_analysis": {
            fam: {
                "count": len(tensors),
                "mean_mse": float(np.mean([t["mse"] for t in tensors])),
                "median_mse": float(np.median([t["mse"] for t in tensors])),
                "max_mse": float(np.max([t["mse"] for t in tensors])),
                "max_rel_rmse": float(np.max([t["rel_rmse"] for t in tensors if t["rel_rmse"] is not None] or [0.0])),
                "total_elements": int(sum(t["n"] for t in tensors)),
            }
            for fam, tensors in sorted(family_groups.items())
        },
    }

    with open(args.json_out, "w") as f:
        json.dump(report, f, indent=1)

    # Save text report
    lines = []
    lines.append("TENSOR DIFFERENCE ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"total tensors compared      : {len(per_tensor)}")
    lines.append(f"exact (byte-equal by MSE)    : {len([t for t in per_tensor if t['exact_frac'] >= 1.0])}")
    lines.append(f"differing (MSE-based)        : {len([t for t in per_tensor if t['mse'] > 0])}")
    lines.append(f"hash-identical count        : {len([t for t in per_tensor if t['exact_frac'] >= 1.0])}")
    lines.append(f"hash-differing count        : {len([t for t in per_tensor if t['mse'] > 0])}")
    lines.append("-" * 60)
    lines.append("DIFFERENT FAMILY GROUPS:")
    lines.append("-" * 60)
    for fam in sorted(family_groups.keys()):
        tensors = family_groups[fam]
        mses = [t["mse"] for t in tensors]
        rels = [t["rel_rmse"] for t in tensors if t["rel_rmse"] is not None]
        lines.append(
            f"{fam:44s} n={len(tensors):3d} mean_mse={np.mean(mses):.3e} "
            f"max_mse={np.max(mses):.3e} max_rel={np.max(rels) if rels else 0:.3e} "
            f"n_elem={sum(t['n'] for t in tensors):,}"
        )
    lines.append("-" * 60)
    lines.append("TOP-10 BY MSE (differing):")
    lines.append("-" * 60)
    top_mse = sorted(per_tensor, key=lambda t: -t["mse"])[:10]
    for t in top_mse:
        lines.append(f"  {t['name']:44s} mse={t['mse']:.3e} rel={t['rel_rmse']:.3e}")
    lines.append("-" * 60)
    lines.append("TOP-10 BY REL_RMSE (differing):")
    lines.append("-" * 60)
    top_rel = sorted(per_tensor, key=lambda t: -(t["rel_rmse"] if t["rel_rmse"] is not None else 0.0))[:10]
    for t in top_rel:
        lines.append(f"  {t['name']:44s} mse={t['mse']:.3e} rel={t['rel_rmse']:.3e}")
    lines.append("\n")
    lines.append("END OF ANALYSIS")
    with open(args.txt_out, "w") as f:
        f.write("\n".join(lines))

    print(f"\nJSON report saved to {args.json_out}")
    print(f"Text report saved to {args.txt_out}")


if __name__ == "__main__":
    main()