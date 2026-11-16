#!/usr/bin/env python3
"""Scaled-down end-to-end test of extract_v3nano_gguf.py on synthetic data.

Builds tiny TFLite flatbuffers that store a shrunken gemma3n the way the
mobile conversion does (int8/int4 per-row quantization, transposed and
same-order layouts, F16 and "1 + w" norms, altup_proj as 2D slices, 30-way
per-layer slices -> here 3, no K/V for KV-shared layers), plus a matching
"official" F32/F16 GGUF, then checks:

  1) legacy mode: --official ... --write-map  (match + fallback + calibration)
  2) standalone mode: --map only, no official GGUF anywhere
  3) standalone output == legacy output wherever both read v3nano, every
     other tensor within quantization noise, only KV-shared K/V zero-filled

Usage:  python3 tests/test_synthetic.py [workdir]     (needs numpy gguf tflite)
"""
import sys, os, struct, json, importlib
import numpy as np, flatbuffers, gguf
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import gemma3n_spec as spec
# --- shrink the model
for k, v in dict(N_LAYER=3, SHARED_KV_LAYERS=1, N_EMBD=64, N_FF=96, N_HEAD=2, N_HEAD_KV=1,
                 HEAD_DIM=32, N_EMBD_PER_LAYER=16, LAUREL_RANK=8, N_VOCAB=512).items():
    setattr(spec, k, v)
import extract_v3nano_gguf as X
X.PER_LAYER_ROWS, X.PER_LAYER_SLICES, X.PER_LAYER_WIDTH = 512, 3, 16
X.PER_LAYER_TOTAL_WIDTH = 48
X.CHUNK = 1000          # odd-ish chunking exercises the int4 range path
import tempfile
W = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='v3nano_test_')
os.makedirs(W, exist_ok=True)
rng = np.random.default_rng(0)
layout = spec.tensor_layout()
off_vals = {}
for t in layout:
    shape = tuple(reversed(t['ne']))
    if len(t['ne']) == 1:
        v = rng.normal(1.0, 0.5, shape)
    else:
        v = rng.normal(0, 0.05, shape)
    off_vals[t['name']] = v.astype(np.float32)

# --- quantizers
def q_int8_rows(m):            # per-row symmetric
    sc = np.abs(m).max(1) / 127 + 1e-12
    return np.round(m / sc[:, None]).astype(np.int8), sc.astype(np.float32), np.zeros(len(sc), np.int64)
def q_int4_rows(m):
    sc = np.abs(m).max(1) / 7 + 1e-12
    q = np.clip(np.round(m / sc[:, None]), -8, 7).astype(np.int8)
    return q, sc.astype(np.float32), np.zeros(len(sc), np.int64)
def pack4(q):
    f = (q.ravel().astype(np.int16) & 0xF).astype(np.uint8)
    if f.size % 2: f = np.append(f, 0)
    return (f[0::2] | (f[1::2] << 4)).astype(np.uint8)

tensors = {s: [] for s in X.FLATBUF_STEMS}   # stem -> [(shape, type, bytes, scales, zps)]
def add(stem, shape, ttype, data, sc=None, zp=None):
    tensors[stem].append((list(shape), ttype, data, sc, zp))

PD = 'TF_LITE_PREFILL_DECODE'
for t in layout:
    n, v = t['name'], off_vals[t['name']]
    if n == 'per_layer_token_embd.weight':
        # numpy (V, 48): slices z = columns z*16:(z+1)*16, int4 per-row
        for z in range(3):
            q, sc, zp = q_int4_rows(v[:, z*16:(z+1)*16])
            add('TF_LITE_PER_LAYER_EMBEDDER', q.shape, 17, pack4(q).tobytes(), sc, zp)
        continue
    if n in ('altup_proj.weight', 'altup_unembd_proj.weight'):
        for z in range(v.shape[0]):           # three separate 2D slices, transposed int8
            q, sc, zp = q_int8_rows(np.ascontiguousarray(v[z].T))
            add(PD, q.shape, 9, q.tobytes(), sc, zp)
        continue
    if n.startswith('blk.2.') and n.endswith(('attn_k.weight', 'attn_v.weight')):
        continue                               # KV-shared layer: absent in v3nano
    if len(t['ne']) == 1:
        if 'post_norm' in n or n == 'output_norm.weight':
            add(PD, (1, 1) + v.shape, 0, (v + 1).astype(np.float32).tobytes())  # stored 1+w
        elif 'attn_q_norm' in n:
            add(PD, v.shape, 1, v.astype(np.float16).tobytes())
        else:
            add(PD, v.shape, 0, v.tobytes())
        continue
    if n == 'token_embd.weight':
        add('TF_LITE_EMBEDDER', v.shape, 9, *(lambda q, s, z: (q.tobytes(), s, z))(*q_int8_rows(v)))
        continue
    if 'coef' in n:
        add(PD, v.shape, 0, v.tobytes()); continue
    if 'ffn' in n or 'laurel' in n:          # int4 per-row, same orientation
        q, sc, zp = q_int4_rows(v)
        add(PD, q.shape, 17, pack4(q).tobytes(), sc, zp); continue
    q, sc, zp = q_int8_rows(np.ascontiguousarray(v.T))   # int8 transposed
    add(PD, q.shape, 9, q.tobytes(), sc, zp)

BASE = 1 << 20
def write_flatbuf(path, items):
    blobs, off = [], BASE
    locs = []
    for shape, tt, data, sc, zp in items:
        locs.append((off, len(data))); blobs.append(data)
        off += (len(data) + 63) // 64 * 64
    b = flatbuffers.Builder(1024)
    M = lambda m: sys.modules['tflite.' + m] if 'tflite.' + m in sys.modules else importlib.import_module('tflite.' + m)
    Buf, Ten, Q, SG, Mod = M('Buffer'), M('Tensor'), M('QuantizationParameters'), M('SubGraph'), M('Model')
    bufs = []
    Buf.BufferStart(b); bufs.append(Buf.BufferEnd(b))   # buffer 0 empty
    for o, sz in locs:
        Buf.BufferStart(b); Buf.BufferAddOffset(b, o); Buf.BufferAddSize(b, sz); bufs.append(Buf.BufferEnd(b))
    tens = []
    for i, (shape, tt, data, sc, zp) in enumerate(items):
        shp = b.CreateNumpyVector(np.array(shape, np.int32))
        qo = None
        if sc is not None:
            so = b.CreateNumpyVector(np.asarray(sc, np.float32)); zo = b.CreateNumpyVector(np.asarray(zp, np.int64))
            Q.QuantizationParametersStart(b); Q.QuantizationParametersAddScale(b, so); Q.QuantizationParametersAddZeroPoint(b, zo)
            qo = Q.QuantizationParametersEnd(b)
        else:
            Q.QuantizationParametersStart(b); qo = Q.QuantizationParametersEnd(b)
        Ten.TensorStart(b); Ten.TensorAddShape(b, shp); Ten.TensorAddType(b, tt); Ten.TensorAddBuffer(b, i + 1)
        Ten.TensorAddQuantization(b, qo); tens.append(Ten.TensorEnd(b))
    SG.SubGraphStartTensorsVector(b, len(tens))
    for x in reversed(tens): b.PrependUOffsetTRelative(x)
    tvec = b.EndVector()
    SG.SubGraphStart(b); SG.SubGraphAddTensors(b, tvec); sg = SG.SubGraphEnd(b)
    Mod.ModelStartSubgraphsVector(b, 1); b.PrependUOffsetTRelative(sg); sgv = b.EndVector()
    Mod.ModelStartBuffersVector(b, len(bufs))
    for x in reversed(bufs): b.PrependUOffsetTRelative(x)
    bv = b.EndVector()
    Mod.ModelStart(b); Mod.ModelAddVersion(b, 3); Mod.ModelAddSubgraphs(b, sgv); Mod.ModelAddBuffers(b, bv)
    b.Finish(Mod.ModelEnd(b), file_identifier=b"TFL3")
    fb = bytes(b.Output()); assert len(fb) < BASE
    with open(path, 'wb') as f:
        f.write(fb + b'\0' * (BASE - len(fb)))
        for d in blobs:
            f.write(d + b'\0' * ((len(d) + 63) // 64 * 64 - len(d)))

dec = os.path.join(W, 'dec'); os.makedirs(dec, exist_ok=True)
for stem in X.FLATBUF_STEMS:
    write_flatbuf(os.path.join(dec, stem + '.bin'), tensors[stem])
for sec in X.SECTIONS:
    if not os.path.exists(os.path.join(dec, sec + '.bin')):
        open(os.path.join(dec, sec + '.bin'), 'wb').write(b'x')


def sp_model(pieces):
    """Minimal SentencePiece ModelProto: [(piece, score, type)]."""
    def varint(n):
        out = b''
        while True:
            b, n = n & 0x7F, n >> 7
            out += bytes([b | (0x80 if n else 0)])
            if not n:
                return out
    def ld(field, payload):
        return varint(field << 3 | 2) + varint(len(payload)) + payload
    msg = b''
    for piece, score, typ in pieces:
        body = ld(1, piece.encode()) + varint(2 << 3 | 5) + struct.pack('<f', score)             + varint(3 << 3 | 0) + varint(typ)
        msg += ld(1, body)
    return msg + ld(2, b'')            # trainer_spec stub


pieces = [('<pad>', 0, 3), ('<eos>', 0, 3), ('<bos>', 0, 3), ('<unk>', 0, 2),
          ('<mask>', 0, 4), ('[multimodal]', 0, 4), ('<unused0>', 0, 4),
          ('▁▁', 0, 4), ('<start_of_turn>', 0, 4)]
pieces += [(f'tok{i}', -float(i), 1) for i in range(len(pieces), spec.N_VOCAB + 20)]
open(os.path.join(dec, 'TOKENIZER_MODEL.bin'), 'wb').write(sp_model(pieces))
tok, sc_, ty = spec.build_vocab(sp_model(pieces))
assert len(tok) == spec.N_VOCAB
assert ty[:9] == [3, 3, 3, 3, 3, 4, 3, 4, 3], ty[:9]
assert tok[7] == '  ' and sc_[7] == -1000.0 and sc_[20] == -20.0

# official GGUF (F32/F16) with a small KV
off_path = os.path.join(W, 'official.gguf')
w = gguf.GGUFWriter(off_path, 'gemma3n')
w.add_block_count(spec.N_LAYER)
for t in layout:
    v = off_vals[t['name']]
    w.add_tensor(t['name'], v if t['type'] == 0 else v.astype(np.float16))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print('synthetic inputs ready')

def run(argv):
    sys.argv = ['extract_v3nano_gguf.py'] + argv
    X._HANDLES.clear()
    return X.main()

common = ['--decode', 'never', '--decoded-dir', dec, '--flatbuf-dir', dec]
mp = os.path.join(W, 'map.json')
# --write-map without --out: map only, no GGUF anywhere
mp_only = os.path.join(W, 'map_only.json')
cwd = os.getcwd()
os.chdir(W)
try:
    rc0 = run(common + ['--official', off_path, '--write-map', mp_only])
finally:
    os.chdir(cwd)
assert rc0 == 0 and os.path.exists(mp_only), rc0
stray = [os.path.join(d, f) for d, _, fs in os.walk(W) for f in fs if f.endswith('.gguf')
         and f != 'official.gguf']
assert not stray, f'map-only run produced a GGUF: {stray}'
rc1 = run(common + ['--official', off_path, '--write-map', mp, '--out', os.path.join(W, 'legacy.gguf')])
rc2 = run(common + ['--map', mp, '--out', os.path.join(W, 'standalone.gguf')])
print('rc legacy', rc1, 'rc standalone', rc2)
m = json.load(open(mp)); print('map missing:', m['missing'])
r = gguf.GGUFReader(os.path.join(W, 'standalone.gguf'))
worst = []
for t in r.tensors:
    v = np.asarray(t.data, np.float32).reshape(off_vals[t.name].shape)
    ref = off_vals[t.name]
    rel = float(np.sqrt(np.mean((v - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-12))
    worst.append((rel, t.name))
worst.sort(reverse=True)
print('worst rel RMSE:', worst[:6])
zero = [n for rel, n in worst if rel == 1.0]
print('zero-filled:', zero)
assert len(r.fields['tokenizer.ggml.tokens'].data) == spec.N_VOCAB
assert int(r.fields['tokenizer.ggml.eos_token_id'].parts[-1][0]) == spec.TOKEN_EOS
L = {t.name: np.asarray(t.data) for t in gguf.GGUFReader(os.path.join(W, 'legacy.gguf')).tensors}
Sd = {t.name: np.asarray(t.data) for t in r.tensors}
ident = sum(np.array_equal(L[n], Sd[n]) for n in L)
print('identical to legacy:', ident, '/', len(L))
int4 = lambda n: 'ffn' in n or 'laurel' in n or 'per_layer_token' in n
bad = [(rel, n) for rel, n in worst if n not in zero and rel > (0.15 if int4(n) else 0.02)]
assert not bad and rc1 == 0 and rc2 == 0, bad[:5]
assert sorted(zero) == ['blk.2.attn_k.weight', 'blk.2.attn_v.weight'], zero
assert ident >= 65, ident
print('OK')
