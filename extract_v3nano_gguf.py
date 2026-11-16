#!/usr/bin/env python3
"""End-to-end v3nano extractor: Chromium bundle -> unquantized GGUF (llama.cpp).

Stages (one run):

  [0] decode     Chromium `weights.bin` (ZIP-like container; each section's
                 first H bytes XORed with a global 1 MiB keystream, rest clear)
                 -> TF_LITE_*.bin runtime flatbuffers + METADATA in `dec/`.
  [1] inventory  mmap every TFLite flatbuffer, index all tensors; identical
                 subgraph copies are collapsed by md5 (raw + scales + zero-points)
                 and shapes are normalized (leading/trailing 1s stripped).
  [2] match      for each official gemma-3n-E2B tensor pick the v3nano TFLite
                 tensor with the lowest sampled MSE (20k samples). Orientation:
                     cand.shape == ne         -> stored transposed, gather
                     cand.shape == ne[::-1]   -> same flat order, direct copy
                 Match is kept only if MSE < --threshold (default 5e-5);
                 otherwise the tensor is rebuilt from the official GGUF
                 (v3nano == E2B weights, so values agree).
  [3] build      stream-write GGUF v3: KV section copied verbatim from the
                 official GGUF (llama.cpp gemma3n metadata), 727 tensor infos
                 (F32/F16 per official tensor type), then tensor data:
                     matched  -> direct or transposed stream from TFLite
                     fallback -> block-chunked dequant from the official GGUF
                     per_layer_token_embd -> 30 int4 slices row-interleaved
                                             into ne=[7680,262144] layout
  [4] verify     sampled MSE of every tensor in the new file vs the official
                 GGUF; non-zero exit if any tensor exceeds 1e-3.

Nothing is materialized whole: TFLite and GGUF data are touched through mmap
views, quantized blocks are dequantized in bounded chunks, and the per-layer
assembly streams 2048-row windows.

Usage (from the repo root):
  python3 work/py/extract_v3nano_gguf.py                     # full run
  python3 work/py/extract_v3nano_gguf.py --decode always     # force re-decode
  python3 work/py/extract_v3nano_gguf.py --out /tmp/x.gguf --no-verify
  python3 work/py/extract_v3nano_gguf.py --save-matches work/json/match_map.json

Inputs (canonical layout):
  ~/.cache/gchrome-test-profile/OptGuideOnDeviceModel/*/weights.bin
  work/bin/keystream_1mb.bin          1 MiB XOR keystream
  work/json/encrypt_lengths.json      per-section encrypted header length H
  work/bin/gemma-3n-E2B-it-Q4_K_S.gguf  official GGUF (KV/layout + fallback)

Output: work/bin/v3nano_final.gguf (unquantized).
Quantize separately if needed:
  llama-quantize work/bin/v3nano_final.gguf work/bin/v3nano_final_Q4_K_S.gguf Q4_K_S
"""

import argparse
import hashlib
import json
import mmap
import os
import shutil
import struct
import sys
import time

import numpy as np
import gguf
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES
from tflite.Model import Model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOME = os.path.expanduser("~")
DEFAULT_WEIGHTS = os.path.join(
    HOME, ".cache/gchrome-test-profile/OptGuideOnDeviceModel/2025.8.21.1028/weights.bin"
)
DEFAULT_KEYSTREAM = "work/bin/keystream_1mb.bin"
DEFAULT_LENGTHS = "work/json/encrypt_lengths.json"
DEFAULT_DECODED = "work/bin/dec"
DEFAULT_FLATBUF_DIR = "work/bin"
DEFAULT_OFFICIAL = "work/bin/gemma-3n-E2B-it-Q4_K_S.gguf"
DEFAULT_OUT = "work/bin/v3nano_final.gguf"

FLATBUF_STEMS = [
    "TF_LITE_PREFILL_DECODE",
    "TF_LITE_EMBEDDER",
    "TF_LITE_PER_LAYER_EMBEDDER",
    "TF_LITE_VISION_ENCODER",
    "TF_LITE_VISION_ADAPTER",
]
PER_LAYER_NAME = "per_layer_token_embd.weight"
PER_LAYER_STEM = "TF_LITE_PER_LAYER_EMBEDDER"
PER_LAYER_ROWS = 262144          # tokens
PER_LAYER_SLICES = 30
PER_LAYER_WIDTH = 256            # 30 * 256 = 7680
PER_LAYER_TOTAL_WIDTH = 7680

# Tensors never taken from v3nano: folded/fused in the mobile conversion, or
# assembled specially. altup_* are F16 in the official file and identical there.
SKIP_MATCH = {"altup_proj.weight", "altup_unembd_proj.weight", PER_LAYER_NAME}

# Chromium weights.bin section table (offset, size) - mirrors decode_v3nano.py
SECTIONS = {
    "TOKENIZER_MODEL": (0x80, 4_683_319),
    "TF_LITE_VISION_ENCODER": (0x477700, 150_460_932),
    "TF_LITE_VISION_ADAPTER": (0x93F5180, 17_847_812),
    "TF_LITE_PREFILL_DECODE": (0xA4FA800, 1_408_555_140),
    "TF_LITE_PER_LAYER_EMBEDDER": (0x5E448100, 1_009_786_256),
    "TF_LITE_EMBEDDER": (0x9A749F00, 271_583_748),
    "METADATA": (0xAAA4A980, 173),
}
XOR_HEADER_MAX = 1 << 20

CHUNK = 1 << 20                  # elements per streaming write
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# GGUF v3 header parsing
# ---------------------------------------------------------------------------
_KV_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _skip_kv_value(b, p, t):
    if t == 8:  # string
        n = struct.unpack("<Q", b[p:p + 8])[0]
        return p + 8 + n
    if t == 9:  # array
        at = struct.unpack("<I", b[p:p + 4])[0]
        al = struct.unpack("<Q", b[p + 4:p + 12])[0]
        p += 12
        for _ in range(al):
            p = _skip_kv_value(b, p, at)
        return p
    return p + _KV_SIZES[t]


def _copy_kv_with_overrides(fin, kv_end, fout, overrides):
    """Copy the KV section from fin (starting at byte 24) to fout.

    For any key present in `overrides` (a dict of key -> new string value),
    the string value is replaced. All other records are copied byte-for-byte.
    """
    pos = 24
    while pos < kv_end:
        fin.seek(pos)
        kl = struct.unpack("<Q", fin.read(8))[0]
        key = fin.read(kl).decode("utf-8")
        vtype = struct.unpack("<I", fin.read(4))[0]
        # Compute the end of this record to know how many raw bytes to copy
        # when we are not overriding.
        p = pos + 8 + kl + 4
        p = _skip_kv_value_stream(fin, p, vtype)
        rec_end = p
        if key in overrides and vtype == 8:  # string
            new_val = overrides[key].encode("utf-8")
            fout.write(struct.pack("<Q", kl))
            fout.write(key.encode("utf-8"))
            fout.write(struct.pack("<I", vtype))
            fout.write(struct.pack("<Q", len(new_val)))
            fout.write(new_val)
        else:
            fin.seek(pos)
            remaining = rec_end - pos
            while remaining > 0:
                chunk = fin.read(min(1 << 20, remaining))
                fout.write(chunk)
                remaining -= len(chunk)
        pos = rec_end


def _skip_kv_value_stream(fin, p, t):
    """Return the byte offset just past the KV value starting at p (in fin)."""
    if t == 8:  # string
        fin.seek(p)
        n = struct.unpack("<Q", fin.read(8))[0]
        return p + 8 + n
    if t == 9:  # array
        fin.seek(p)
        at = struct.unpack("<I", fin.read(4))[0]
        al = struct.unpack("<Q", fin.read(8))[0]
        p += 12
        for _ in range(al):
            p = _skip_kv_value_stream(fin, p, at)
        return p
    return p + _KV_SIZES[t]


def parse_gguf_header(fn):
    """Parse a GGUF header without touching tensor data.

    Returns dict(version, nt, nk, kv_end, tensors[{name,ne,type,offset}],
    data_start). kv_end is the byte offset just past the KV section; data_start
    is the 32-byte-aligned start of the tensor data.
    """
    with open(fn, "rb") as f:
        b = f.read(64 * 1024 * 1024)  # KV ~6.5 MB + infos ~60 KB for this model
    if b[:4] != b"GGUF":
        sys.exit(f"ERROR: {fn}: not a GGUF file")
    version = struct.unpack("<I", b[4:8])[0]
    nt = struct.unpack("<Q", b[8:16])[0]
    nk = struct.unpack("<Q", b[16:24])[0]
    p = 24
    for _ in range(nk):
        kl = struct.unpack("<Q", b[p:p + 8])[0]
        p += 8 + kl
        t = struct.unpack("<I", b[p:p + 4])[0]
        p += 4
        p = _skip_kv_value(b, p, t)
    kv_end = p
    tensors = []
    for _ in range(nt):
        kl = struct.unpack("<Q", b[p:p + 8])[0]
        p += 8
        name = b[p:p + kl].decode()
        p += kl
        nd = struct.unpack("<I", b[p:p + 4])[0]
        p += 4
        ne = tuple(struct.unpack("<q", b[p + j * 8:p + j * 8 + 8])[0] for j in range(nd))
        p += 8 * nd
        ttype = struct.unpack("<I", b[p:p + 4])[0]
        p += 4
        off = struct.unpack("<Q", b[p:p + 8])[0]
        p += 8
        tensors.append(dict(name=name, ne=ne, type=int(ttype), offset=int(off)))
    data_start = (p + 31) // 32 * 32
    return dict(version=version, nt=nt, nk=nk, kv_end=kv_end,
                tensors=tensors, data_start=data_start)


# ---------------------------------------------------------------------------
# [0] Chromium weights.bin -> decoded flatbuffers
# ---------------------------------------------------------------------------
def decode_chromium(weights, keystream, lengths_json, outdir):
    if not os.path.exists(weights):
        sys.exit(f"ERROR: Chromium weights.bin not found: {weights}\n"
                 f"       (pass --decode never to reuse an existing {outdir})")
    key = open(keystream, "rb").read()
    key = key[:XOR_HEADER_MAX] if len(key) > XOR_HEADER_MAX else key + b"\x00" * (XOR_HEADER_MAX - len(key))
    lengths = json.load(open(lengths_json))
    lengths["METADATA"] = 173
    os.makedirs(outdir, exist_ok=True)
    total = 0
    with open(weights, "rb") as src:
        for name, (off, size) in SECTIONS.items():
            h = min(int(lengths.get(name, XOR_HEADER_MAX)), size, len(key))
            out_path = os.path.join(outdir, name + ".bin")
            src.seek(off)
            with open(out_path, "wb") as out:
                first = src.read(h)
                if len(first) != h:
                    sys.exit(f"ERROR: short read for section {name}")
                out.write(bytes(a ^ b for a, b in zip(first, key[:h])))
                remaining = size - h
                while remaining > 0:
                    chunk = src.read(min(1 << 24, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)
            total += size
            log(f"decoded {name:28s} H={h:>9,} size={size:>13,} -> {out_path}")
    log(f"decode complete: {total:,} bytes -> {outdir}")


def flatbuf_paths(flatbuf_dir, decoded_dir, prefer_decoded):
    """Resolve the five TF_LITE_* flatbuffers (plain names, else *.bin)."""
    paths = {}
    for stem in FLATBUF_STEMS:
        cands = []
        if prefer_decoded:
            cands = [os.path.join(decoded_dir, stem + ".bin"),
                     os.path.join(flatbuf_dir, stem)]
        else:
            cands = [os.path.join(flatbuf_dir, stem),
                     os.path.join(decoded_dir, stem + ".bin")]
        for p in cands:
            if os.path.exists(p):
                paths[stem] = p
                break
        else:
            sys.exit(f"ERROR: missing flatbuffer {stem}; tried: {', '.join(cands)}")
    return paths


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def norm_shape(shape):
    s = tuple(int(x) for x in shape)
    while len(s) > 1 and s[0] == 1:
        s = s[1:]
    while len(s) > 1 and s[-1] == 1:
        s = s[:-1]
    return s


def unpack_int4(raw):
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo).astype(np.int8)
    hi = np.where(hi >= 8, hi - 16, hi).astype(np.int8)
    out = np.empty(raw.size * 2, dtype=np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out


def buf_view(mm, off, dtype, count):
    """Read-only numpy view into an mmap (no copy, pages faulted on demand)."""
    return np.frombuffer(mm, dtype=dtype, count=count, offset=off)


_HANDLES = {}


def get_mm(path):
    if path not in _HANDLES:
        fh = open(path, "rb")
        _HANDLES[path] = (fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ))
    return _HANDLES[path][1]


def close_handles():
    for fh, mm in _HANDLES.values():
        try:
            mm.close()
            fh.close()
        except Exception:
            pass
    _HANDLES.clear()


def tflite_tensor_meta(path, sg, ti):
    """Locate one TFLite tensor (buffer offset/size, shapes, quantization)."""
    mm = get_mm(path)
    m = Model.GetRootAsModel(mm, 0)
    t = m.Subgraphs(sg).Tensors(ti)
    buf = m.Buffers(t.Buffer())
    q = t.Quantization()
    sc = np.array(q.ScaleAsNumpy(), dtype=np.float32) if q.ScaleLength() > 0 else None
    zp = np.array(q.ZeroPointAsNumpy(), dtype=np.int64) if q.ZeroPointLength() > 0 else None
    return dict(
        shape=tuple(t.Shape(j) for j in range(t.ShapeLength())),
        type=int(t.Type()), off=int(buf.Offset()), nbytes=int(buf.Size()),
        scales=sc, zps=zp, mm=mm,
    )


# ---------------------------------------------------------------------------
# [1] Inventory
# ---------------------------------------------------------------------------
def inventory(paths):
    inv = []
    for stem in FLATBUF_STEMS:
        path = paths[stem]
        fh = open(path, "rb")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        m = Model.GetRootAsModel(mm, 0)
        for sg_i in range(m.SubgraphsLength()):
            sg = m.Subgraphs(sg_i)
            for ti in range(sg.TensorsLength()):
                t = sg.Tensors(ti)
                b = t.Buffer()
                if b == 0:
                    continue
                buf = m.Buffers(b)
                if buf.Size() == 0:
                    continue
                if t.Type() not in (0, 2, 9, 17):
                    continue
                shape = tuple(t.Shape(j) for j in range(t.ShapeLength()))
                nelem = 1
                for s in shape:
                    nelem *= s
                q = t.Quantization()
                sc = np.array(q.ScaleAsNumpy(), dtype=np.float32) if q.ScaleLength() > 0 else None
                zp = np.array(q.ZeroPointAsNumpy(), dtype=np.int64) if q.ZeroPointLength() > 0 else None
                h = hashlib.md5()
                off, sz = buf.Offset(), buf.Size()
                for s in range(0, sz, 1 << 24):          # incremental: no big copies
                    h.update(mm[off + s:off + min(sz, s + (1 << 24))])
                h.update(sc.tobytes() if sc is not None else b"")
                h.update(zp.tobytes() if zp is not None else b"")
                inv.append(dict(path=path, sg=sg_i, ti=ti, shape=shape,
                                ns=norm_shape(shape), type=int(t.Type()),
                                nelem=nelem, nbytes=sz, offset=off,
                                scales=sc, zps=zp, hash=h.hexdigest()))
        mm.close()
        fh.close()
    log(f"inventory: {len(inv)} tensors in {len(FLATBUF_STEMS)} flatbuffers")
    return inv


def group_candidates(inv):
    by_nelem = {}
    for c in inv:
        by_nelem.setdefault(c["nelem"], []).append(c)
    for k, lst in by_nelem.items():
        seen, uniq = set(), []
        for c in lst:
            if c["hash"] not in seen:
                seen.add(c["hash"])
                uniq.append(c)
        by_nelem[k] = uniq
    log(f"candidates: {sum(len(v) for v in by_nelem.values())} unique "
        f"(md5-dedup of identical subgraph copies)")
    return by_nelem


# ---------------------------------------------------------------------------
# Sampled value access
# ---------------------------------------------------------------------------
def official_values(t, samples):
    """Values of an official GGUF tensor at flat positions (block-cached)."""
    raw = t.data.reshape(-1)
    ttype = int(t.tensor_type)
    if ttype == 0:
        return np.frombuffer(raw, dtype=np.float32)[samples]
    if ttype == 1:
        return np.frombuffer(raw, dtype=np.float16)[samples].astype(np.float32)
    bsz, bbytes = GGML_QUANT_SIZES[ttype]
    blk_ids = samples // bsz
    cache = {}
    for b in np.unique(blk_ids):
        block = np.ascontiguousarray(raw[b * bbytes:(b + 1) * bbytes])
        cache[b] = gguf.quants.dequantize(
            block, gguf.GGMLQuantizationType(ttype)).astype(np.float32)
    return np.array([cache[b][s - b * bsz] for s, b in zip(samples, blk_ids)],
                    dtype=np.float32)


def candidate_values(c, jidx):
    """Values of a TFLite candidate at flat positions jidx (mmap views only)."""
    mm = get_mm(c["path"])
    off, nbytes, t = c["offset"], c["nbytes"], c["type"]
    if t == 0:
        return buf_view(mm, off, np.float32, c["nelem"])[jidx]
    if t == 2:
        return buf_view(mm, off, np.int32, c["nelem"])[jidx].astype(np.float32)
    raw = buf_view(mm, off, np.uint8, nbytes)
    if t == 9:
        v = raw.view(np.int8)[jidx].astype(np.float32)
    elif t == 17:
        byte = jidx >> 1
        nib = raw[byte]
        v = np.where((jidx & 1) == 0, nib & 0x0F, (nib >> 4) & 0x0F).astype(np.int8)
        v = np.where(v >= 8, v - 16, v).astype(np.float32)
    else:
        raise ValueError(f"unsupported tflite type {t}")
    sc, zp = c["scales"], c["zps"]
    if sc is None:
        return v
    if sc.size == 1:
        return (v - float(zp[0])) * float(sc[0])
    tail = int(np.prod(c["ns"][1:])) if len(c["ns"]) > 1 else 1
    row = jidx // tail
    return (v - zp[row].astype(np.float32)) * sc[row]


# ---------------------------------------------------------------------------
# [2] Matching
# ---------------------------------------------------------------------------
def match_all(reader, by_nelem, nsamp):
    """Per official tensor: lowest-MSE v3nano candidate, plus orientation.

    Replicates the reference matcher exactly (single RNG stream seeded 1234,
    candidates in inventory order, ties keep the first, skip-listed tensors do
    not consume samples), so decisions are reproducible run to run.
    """
    rng = np.random.default_rng(1234)
    decisions = {}
    n_ok = n_fb = 0
    for t in reader.tensors:
        name = t.name
        if name in SKIP_MATCH:
            decisions[name] = {"status": "special"}
            continue
        ne = tuple(int(x) for x in t.shape)
        nelem = 1
        for x in ne:
            nelem *= x
        cands = by_nelem.get(nelem, [])
        if not cands:
            decisions[name] = {"status": "no_candidate"}
            log(f"match {name}: NO CANDIDATE (nelem={nelem})")
            n_fb += 1
            continue
        samples = rng.integers(0, nelem, nsamp, dtype=np.int64)
        offv = official_values(t, samples)
        best = (float("inf"), None, False)
        for c in cands:
            ns = c["ns"]
            if len(ne) == 2 and len(ns) == 2:
                if ns == ne[::-1]:
                    transpose = False          # same flat order as GGUF
                elif ns == ne:
                    transpose = True           # candidate is off.T
                else:
                    continue
            elif len(ne) in (1, 3):
                if ns != ne:                   # identity orientation only
                    continue
                transpose = False
            else:
                continue
            if transpose:
                a, b = ne
                j = (samples % a) * b + (samples // a)
            else:
                j = samples
            try:
                cv = candidate_values(c, j)
            except Exception:
                continue
            d = offv - cv
            mse = float(np.dot(d, d) / d.size)
            if mse < best[0]:
                best = (mse, c, transpose)
        mse, c, transpose = best
        if c is None:
            decisions[name] = {"status": "no_shape_match"}
            log(f"match {name}: NO SHAPE MATCH")
            n_fb += 1
            continue
        decisions[name] = dict(
            status="ok", mse=mse, flatbuf=c["path"], sg=c["sg"], ti=c["ti"],
            transpose=bool(transpose), shape=list(c["shape"]), ctype=c["type"],
            hash=c["hash"], offset=c["offset"], nbytes=c["nbytes"],
            scales=c["scales"], zps=c["zps"],
        )
        n_ok += 1
        log(f"match {name}: MSE={mse:.3g} tr={transpose} "
            f"{os.path.basename(c['path'])} sg={c['sg']} ti={c['ti']}")
    log(f"matching done: {n_ok} matched, {n_fb} without usable candidate, "
        f"{len(decisions) - n_ok - n_fb} special")
    return decisions


# ---------------------------------------------------------------------------
# [3] Build
# ---------------------------------------------------------------------------
def write_official_fallback(fout, t, out_dtype):
    """Rebuild one tensor from the official GGUF dequantization."""
    raw = t.data.reshape(-1)
    ttype = int(t.tensor_type)
    nelem = 1
    for x in t.shape:
        nelem *= int(x)
    if ttype in (0, 1):
        arr = (np.frombuffer(raw, dtype=np.float32) if ttype == 0
               else np.frombuffer(raw, dtype=np.float16).astype(np.float32))
        for s in range(0, nelem, CHUNK):
            fout.write(arr[s:s + CHUNK].astype(out_dtype).tobytes())
        return
    bsz, bbytes = GGML_QUANT_SIZES[ttype]
    sub = 1 << 18                     # <= 1024 superblocks per dequant call
    for s in range(0, nelem, sub):
        e = min(s + sub, nelem)
        blk0, blk1 = s // bsz, (e + bsz - 1) // bsz
        block = np.ascontiguousarray(raw[blk0 * bbytes:blk1 * bbytes])
        vals = gguf.quants.dequantize(
            block, gguf.GGMLQuantizationType(ttype)).astype(np.float32)
        fout.write(vals[:e - s].astype(out_dtype).tobytes())


def write_tensor_direct(fout, inv, out_dtype):
    """Matched tensor whose stored flat order already equals the GGUF order."""
    ns = norm_shape(inv["shape"])
    nelem = 1
    for s in ns:
        nelem *= s
    sc, zp = inv["scales"], inv["zps"]
    tail = int(np.prod(ns[1:])) if len(ns) > 1 else 1
    n = ns[0]
    per_row = sc is not None and sc.size > 1
    mm, off, t = inv["mm"], inv["off"], inv["type"]

    def read_floats(start, end):
        if t == 0:
            return buf_view(mm, off + start * 4, np.float32, end - start)
        if t == 2:
            return buf_view(mm, off + start * 4, np.int32, end - start).astype(np.float32)
        if t == 9:
            return buf_view(mm, off + start, np.int8, end - start).astype(np.float32)
        if t == 17:
            raw = buf_view(mm, off + start // 2, np.uint8, (end - start + 1) // 2)
            return unpack_int4(raw).astype(np.float32)[:end - start]
        raise ValueError(f"unsupported tflite type {t}")

    if per_row:
        rows = max(1, CHUNK // tail)
        for r0 in range(0, n, rows):
            r1 = min(r0 + rows, n)
            arr = read_floats(r0 * tail, r1 * tail).reshape(r1 - r0, tail)
            arr = (arr - zp[r0:r1].astype(np.float32)[:, None]) \
                * sc[r0:r1].astype(np.float32)[:, None]
            fout.write(arr.ravel().astype(out_dtype).tobytes())
        return

    for start in range(0, nelem, CHUNK):
        end = min(start + CHUNK, nelem)
        arr = read_floats(start, end)
        if sc is not None:
            arr = (arr - float(zp[0])) * float(sc[0])
        fout.write(arr.astype(out_dtype).tobytes())


def write_tensor_transposed(fout, inv, ne, out_dtype):
    """Candidate stored as off.T (shape (a,b) == ne): out[i] = O[i%a, i//a].

    Gathered in output-order chunks straight from mmap views, so the whole
    candidate is never materialized.
    """
    a, b = ne
    total = a * b
    inv_type = inv["type"]
    sc, zp = inv["scales"], inv["zps"]
    per_row = sc is not None and sc.size > 1
    mm, off = inv["mm"], inv["off"]
    raw = buf_view(mm, off, np.uint8, inv["nbytes"])

    for i0 in range(0, total, CHUNK):
        i = np.arange(i0, min(i0 + CHUNK, total), dtype=np.int64)
        x = i % a
        j = x * b + (i // a)             # candidate flat index (= O[x, y])
        if inv_type == 0:
            v = buf_view(mm, off, np.float32, total)[j]
        elif inv_type == 2:
            v = buf_view(mm, off, np.int32, total)[j].astype(np.float32)
        elif inv_type == 9:
            v = raw.view(np.int8)[j].astype(np.float32)
        elif inv_type == 17:
            nib = raw[j >> 1]
            v = np.where((j & 1) == 0, nib & 0x0F, (nib >> 4) & 0x0F).astype(np.int8)
            v = np.where(v >= 8, v - 16, v).astype(np.float32)
        else:
            raise ValueError(f"unsupported tflite type {inv_type}")
        if per_row:
            v = (v - zp[x].astype(np.float32)) * sc[x]
        elif sc is not None:
            v = (v - float(zp[0])) * float(sc[0])
        fout.write(v.astype(out_dtype).tobytes())


def write_per_layer(fout, path):
    """Assemble per_layer_token_embd.weight (ne=[7680,262144], token-major).

    For token y the 7680 contiguous values are the 30 slices' 256-wide column
    blocks. Rows are streamed in 2048-token windows, each slice dequantized on
    the fly from its flatbuffer buffer (no temp files, ~40 MB peak).
    """
    fh = open(path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        m = Model.GetRootAsModel(mm, 0)
        slices = []                       # (byte offset, scales, zero points)
        for sg_i in range(m.SubgraphsLength()):
            sg = m.Subgraphs(sg_i)
            for ti in range(sg.TensorsLength()):
                t = sg.Tensors(ti)
                b = t.Buffer()
                if b == 0:
                    continue
                buf = m.Buffers(b)
                if buf.Size() == 0:
                    continue
                shape = tuple(t.Shape(j) for j in range(t.ShapeLength()))
                if not (t.Type() == 17 and len(shape) >= 2
                        and shape[0] == PER_LAYER_ROWS
                        and shape[1] == PER_LAYER_WIDTH):
                    continue
                q = t.Quantization()
                sc = np.array(q.ScaleAsNumpy(), dtype=np.float32)
                zp = np.array(q.ZeroPointAsNumpy(), dtype=np.int64)
                slices.append((int(buf.Offset()), sc, zp))
        if len(slices) != PER_LAYER_SLICES:
            sys.exit(f"ERROR: expected {PER_LAYER_SLICES} per-layer slices, "
                     f"got {len(slices)}")
        row_bytes = PER_LAYER_WIDTH // 2   # int4: two values per byte
        for r0 in range(0, PER_LAYER_ROWS, 2048):
            r1 = min(r0 + 2048, PER_LAYER_ROWS)
            nrows = r1 - r0
            row = np.empty((nrows, PER_LAYER_TOTAL_WIDTH), dtype="<f2")
            for z, (off, sc, zp) in enumerate(slices):
                raw = buf_view(mm, off + r0 * row_bytes, np.uint8,
                               nrows * row_bytes)
                v = unpack_int4(raw).astype(np.float32).reshape(
                    nrows, PER_LAYER_WIDTH)
                v = (v - zp[r0:r1, None].astype(np.float32)) \
                    * sc[r0:r1, None]
                row[:, z * PER_LAYER_WIDTH:(z + 1) * PER_LAYER_WIDTH] = \
                    v.astype("<f2")
            fout.write(row.tobytes())
    finally:
        raw = v = row = None                # drop views, then unmap
        try:
            mm.close()
        except BufferError:
            pass                             # views still alive: let GC unmap
        fh.close()
    log("per_layer assembled (row-interleaved, 30 slices, streamed)")


def output_size_estimate(header):
    """Exact byte size of the unquantized output (infos identical to official)."""
    n = header["data_start"]
    for t in header["tensors"]:
        nelem = 1
        for x in t["ne"]:
            nelem *= int(x)
        n += nelem * (4 if t["type"] == 0 else 2)
    return n


def build_gguf(out_path, header, official_tensors, decisions, paths, threshold, kv_overrides=None):
    nt, nk = header["nt"], header["nk"]
    tensors = header["tensors"]
    official_path = header["source"]
    n_matched = n_fallback = n_special = 0

    need = output_size_estimate(header)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    free = shutil.disk_usage(out_dir).free
    if free < need + (1 << 29):           # 512 MiB headroom
        sys.exit(f"ERROR: not enough disk space in {out_dir}: need "
                 f"{need + (1 << 29):,} bytes, have {free:,}")
    log(f"disk preflight: need {need:,} bytes, free {free:,}")

    with open(out_path, "wb") as fout:
        fout.write(b"GGUF")
        fout.write(struct.pack("<I", 3))          # GGUF v3 (llama.cpp gemma3n)
        fout.write(struct.pack("<Q", nt))
        fout.write(struct.pack("<Q", nk))
        # KV section: verbatim from the official file (llama.cpp gemma3n
        # metadata: arch, hparams, tokenizer, ...), with optional overrides
        # applied via _copy_kv_with_overrides.
        with open(official_path, "rb") as fin:
            if kv_overrides:
                _copy_kv_with_overrides(fin, header["kv_end"], fout, kv_overrides)
            else:
                fin.seek(24)
                remaining = header["kv_end"] - 24
                while remaining > 0:
                    chunk = fin.read(min(1 << 20, remaining))
                    fout.write(chunk)
                    remaining -= len(chunk)
        # Tensor infos (order and shapes identical to official).
        patch_pos = []
        for t in tensors:
            nb = t["name"].encode()
            fout.write(struct.pack("<Q", len(nb)))
            fout.write(nb)
            fout.write(struct.pack("<I", len(t["ne"])))
            for d in t["ne"]:
                fout.write(struct.pack("<q", d))
            out_type = 0 if t["type"] == 0 else 1   # output carries F32 or F16 only
            fout.write(struct.pack("<I", out_type))
            patch_pos.append(fout.tell())
            fout.write(struct.pack("<Q", 0))
        data_start = (fout.tell() + 31) // 32 * 32
        fout.write(b"\x00" * (data_start - fout.tell()))

        offsets = []
        for j, t in enumerate(tensors):
            name = t["name"]
            offsets.append(fout.tell() - data_start)
            out_dtype = "<f4" if t["type"] == 0 else "<f2"

            if name == PER_LAYER_NAME:
                write_per_layer(fout, paths[PER_LAYER_STEM])
                n_special += 1
                log(f"build [{j:3d}] {name}: per-layer interleaved assembly")
                continue

            d = decisions.get(name, {"status": "missing"})
            ok = d.get("status") == "ok" and d.get("mse", 9e9) < threshold
            if not ok:
                mse = d.get("mse")
                mse_s = "n/a" if mse is None else f"{mse:.3g}"
                log(f"build [{j:3d}] {name}: official fallback "
                    f"({d.get('status')}, mse={mse_s})")
                write_official_fallback(fout, official_tensors[name], out_dtype)
                n_fallback += 1
                continue

            inv = dict(type=d["ctype"], shape=tuple(d["shape"]), off=d["offset"],
                       nbytes=d["nbytes"], scales=d["scales"], zps=d["zps"],
                       mm=get_mm(d["flatbuf"]))
            if d["transpose"]:
                write_tensor_transposed(fout, inv, tuple(t["ne"]), out_dtype)
            else:
                write_tensor_direct(fout, inv, out_dtype)
            n_matched += 1
            log(f"build [{j:3d}] {name}: v3nano match tr={d['transpose']} "
                f"mse={d['mse']:.3g}")

        for pos, off in zip(patch_pos, offsets):
            fout.seek(pos)
            fout.write(struct.pack("<Q", off))
        fout.seek(0, 2)

    log(f"wrote {out_path} ({os.path.getsize(out_path):,} bytes) - "
        f"matched={n_matched} fallback={n_fallback} special={n_special}")
    return n_matched, n_fallback, n_special


# ---------------------------------------------------------------------------
# [4] Verification
# ---------------------------------------------------------------------------
def verify(new_path, official_path, nsamp=20000):
    """Sampled MSE per tensor: new file vs official. Returns (worst, n_bad)."""
    new_header = parse_gguf_header(new_path)
    ds = new_header["data_start"]
    new_by_name = {t["name"]: t for t in new_header["tensors"]}
    fh = open(new_path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    rng = np.random.default_rng(999)
    report = []
    n_bad = 0
    try:
        for t in GGUFReader(official_path).tensors:
            name = t.name
            nt_ = new_by_name.get(name)
            if nt_ is None:
                report.append((float("inf"), name))
                n_bad += 1
                log(f"verify: MISSING {name}")
                continue
            nelem = 1
            for x in nt_["ne"]:
                nelem *= int(x)
            k = min(nsamp, nelem)
            samples = rng.integers(0, nelem, k, dtype=np.int64)
            offv = official_values(t, samples)
            if nt_["type"] == 0:
                newv = buf_view(mm, ds + nt_["offset"], np.float32, nelem)[samples]
            else:
                newv = buf_view(mm, ds + nt_["offset"], np.float16, nelem)[samples].astype(np.float32)
            d = offv - newv
            mse = float(np.dot(d, d) / d.size)
            report.append((mse, name))
            if mse > 1e-3:
                n_bad += 1
                log(f"verify: BAD {name}: MSE={mse:.3g}")
    finally:
        samples = newv = offv = None        # drop mmap views before closing
        try:
            mm.close()
        except BufferError:
            pass
        fh.close()
    report.sort(key=lambda x: -x[0])
    log("verify: worst 10 tensors vs official:")
    for mse, name in report[:10]:
        log(f"verify:   {mse:.4g}  {name}")
    log(f"verify: {len(report)} tensors checked, {n_bad} above 1e-3")
    return report, n_bad


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="End-to-end v3nano extractor: Chromium weights.bin -> unquantized GGUF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Chromium weights.bin")
    ap.add_argument("--keystream", default=DEFAULT_KEYSTREAM, help="1 MiB XOR keystream")
    ap.add_argument("--lengths", default=DEFAULT_LENGTHS, help="encrypt_lengths.json")
    ap.add_argument("--decoded-dir", default=DEFAULT_DECODED, help="decoded *.bin directory")
    ap.add_argument("--decode", choices=["auto", "always", "never"], default="auto",
                    help="auto: reuse existing decoded files; always: re-decode; never: require them")
    ap.add_argument("--flatbuf-dir", default=DEFAULT_FLATBUF_DIR,
                    help="directory holding plain TF_LITE_* flatbuffers")
    ap.add_argument("--official", default=DEFAULT_OFFICIAL,
                    help="official gemma3n GGUF (KV/layout reference + fallback)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output unquantized GGUF")
    ap.add_argument("--threshold", type=float, default=5e-5,
                    help="max sampled MSE to trust a v3nano match (else fallback)")
    ap.add_argument("--samples", type=int, default=20000, help="samples per tensor")
    ap.add_argument("--no-verify", action="store_true", help="skip verification")
    ap.add_argument("--save-matches", default="", help="write match decisions JSON here")
    ap.add_argument("--quantized-by", default="",
                    help="override general.quantized_by in the output GGUF "
                         "(e.g. 'daedalus'). Empty = keep official value.")
    ap.add_argument("--repo-url", default="",
                    help="override general.repo_url in the output GGUF. Empty = keep "
                         "official value.")
    return ap.parse_args()


def main():
    args = parse_args()

    # [0] decode
    decoded = [os.path.join(args.decoded_dir, s + ".bin") for s in SECTIONS]
    have_decoded = all(os.path.exists(p) and os.path.getsize(p) > 0 for p in decoded)
    if args.decode == "never" and not have_decoded:
        sys.exit(f"ERROR: --decode never but files missing in {args.decoded_dir}")
    do_decode = args.decode == "always" or (args.decode == "auto" and not have_decoded)
    if do_decode:
        log("stage 0: decoding Chromium weights.bin")
        decode_chromium(args.weights, args.keystream, args.lengths, args.decoded_dir)
    else:
        log(f"stage 0: reusing decoded flatbuffers in {args.decoded_dir}")
    paths = flatbuf_paths(args.flatbuf_dir, args.decoded_dir, prefer_decoded=do_decode)

    # official reference
    header = parse_gguf_header(args.official)
    header["source"] = args.official
    reader = GGUFReader(args.official)
    official_tensors = {t.name: t for t in reader.tensors}
    log(f"official: {header['nt']} tensors, {header['nk']} KV entries, "
        f"KV end {header['kv_end']:,}, data start {header['data_start']:,}")
    if header["nt"] != len(header["tensors"]) or len(set(official_tensors)) != header["nt"]:
        sys.exit("ERROR: official tensor table is inconsistent")
    if header["nt"] != 727:
        log(f"warning: expected 727 tensors for gemma-3n-E2B, got {header['nt']}")

    # [1] inventory
    log("stage 1: inventorying TFLite tensors")
    by_nelem = group_candidates(inventory(paths))

    # [2] match
    log(f"stage 2: matching (threshold {args.threshold:g}, {args.samples} samples/tensor)")
    decisions = match_all(reader, by_nelem, args.samples)
    kept = sum(1 for v in decisions.values()
               if v.get("status") == "ok" and v.get("mse", 9e9) < args.threshold)
    log(f"matching summary: {kept} kept from v3nano, "
        f"{len(decisions) - kept} official fallback/special")
    if args.save_matches:
        safe = {k: {kk: vv for kk, vv in v.items()
                    if kk in ("status", "mse", "flatbuf", "sg", "ti",
                              "transpose", "shape", "ctype", "hash")}
                for k, v in decisions.items()}
        with open(args.save_matches, "w") as f:
            json.dump(safe, f, indent=1)
        log(f"match decisions saved to {args.save_matches}")

    # [3] build
    kv_overrides = {}
    if args.quantized_by:
        kv_overrides["general.quantized_by"] = args.quantized_by
    if args.repo_url:
        kv_overrides["general.repo_url"] = args.repo_url
    log(f"stage 3: building {args.out}")
    if kv_overrides:
        log(f"kv overrides: {kv_overrides}")
    n_matched, n_fallback, n_special = build_gguf(
        args.out, header, official_tensors, decisions, paths, args.threshold,
        kv_overrides=kv_overrides or None)
    close_handles()

    # [4] verify
    rc = 0
    if not args.no_verify:
        log("stage 4: verification (sampled MSE vs official)")
        _, n_bad = verify(args.out, args.official, args.samples)
        if n_bad:
            log(f"FAIL: {n_bad} tensor(s) above 1e-3 MSE")
            rc = 1
        else:
            log("PASS: every tensor is within 1e-3 MSE of the official model")
    log(f"done: {args.out} ({os.path.getsize(args.out):,} bytes); "
        f"matched={n_matched} fallback={n_fallback} special={n_special}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
