# v3nano GGUF extractor

Turn the Gemma-3n (E2B, "v3nano") on-device model that ships inside Chromium
into a single **unquantized GGUF** that `llama.cpp` can load and run.

Chromium stores the model as a set of LiteRT/TFLite flatbuffers with
**unnamed weight tensors**, so there is nothing to map by name. The extractor
builds a llama.cpp-compatible GGUF from the Chromium bundle **alone**, in one
streaming pass:

- **metadata**: `gemma3n_spec.py` holds the llama.cpp `gemma3n` hyperparameters,
  the chat template and the 727-tensor layout. The tokenizer is rebuilt from the
  bundle's own SentencePiece `TOKENIZER_MODEL`, with the rules llama.cpp's
  converter applies. The result is byte-identical to the published E2B GGUFs.
- **weights**: `v3nano_tensor_map.json` says which flatbuffer tensor each GGUF
  tensor lives in and how it is laid out. It holds coordinates only, no
  weights.

An official `gemma-3n-E2B` GGUF is only needed once, to *calibrate* that map
(`--official … --write-map`). The original reference pipeline (match by
sampled MSE, fall back to official values) is still available with
`--official`.

Validated in practice: all 727 tensors land within 1e-3 MSE of the official
model, and the result answers correctly ("…George Washington…", atomic symbol
"H") both unquantized and after `Q4_K_S` quantization.

**Prebuilt weights:** if you only want to run the model, the F16 and `Q4_K_S`
GGUFs built with this script are hosted at
[`daedalus2027/gemma3n-v3nano-gguf`](https://huggingface.co/daedalus2027/gemma3n-v3nano-gguf)
on Hugging Face.

## Requirements

- Python 3.9+
- [`numpy`](https://pypi.org/project/numpy/)
- [`gguf`](https://pypi.org/project/gguf/) — GGUF reader/quant helpers
- [`tflite`](https://pypi.org/project/tflite/) — FlatBuffers Python bindings for the TFLite schema

```bash
pip install numpy gguf tflite
```

`llama.cpp` is **not** required to build the GGUF. You only need it to run or
quantize the result (`llama-quantize`, `llama-cli`).

## Publishing to Hugging Face

After building (or quantizing) the GGUFs, upload them to the public hub repo:

```bash
python3 upload_hf.py                 # upload F16, Q4_K_S, Q8_0 from ../work/bin
python3 upload_hf.py --kind Q8_0   # upload just Q8_0
python3 upload_hf.py --dry-run     # list what would be uploaded
python3 upload_hf.py --upload /path/a.gguf:name_a.gguf /path/b.gguf:name_b.gguf
```

Repo defaults to `daedalus2027/gemma3n-v3nano-gguf`.

## Inputs

| Input | Default | Purpose |
|---|---|---|
| Chromium `weights.bin` | `~/.cache/gchrome-test-profile/OptGuideOnDeviceModel/2025.8.21.1028/weights.bin` | Obfuscated on-device model bundle |
| XOR keystream (1 MiB) | `work/bin/keystream_1mb.bin` | Decrypts each section's header |
| `encrypt_lengths.json` | `work/json/encrypt_lengths.json` | Per-section encrypted header length `H` |
| Decoded flatbuffers | `work/bin/dec/TF_LITE_*.bin` | Written by stage 0, reused on later runs |
| Tensor map | `v3nano_tensor_map.json` (repo) | GGUF name → flatbuffer tensor, orientation, slices |
| Official GGUF (optional) | `--official PATH` | Only for `--write-map` calibration or the reference pipeline |

Every path is overridable — see `--help`.

## Quick start

```bash
python3 extract_v3nano_gguf.py                     # decode + build + self-check, no official GGUF
python3 extract_v3nano_gguf.py --decode always     # force re-decryption of weights.bin
python3 extract_v3nano_gguf.py --no-verify         # skip the self-check
```

### One-time map calibration

`v3nano_tensor_map.json` is tied to the Chromium model version (`2025.8.21.1028`,
the same version the section table in the script is for). To create or refresh
it, run once with any unquantized or quantized official E2B GGUF:

```bash
python3 extract_v3nano_gguf.py --official work/bin/gemma-3n-E2B-it-f16.gguf \
    --write-map v3nano_tensor_map.json
```

This only writes the map; add `--out PATH` to also run the reference build.

For every official tensor this searches the flatbuffers for the best-matching
tensor. The search covers F32/F16/INT32/UINT8/INT8/INT4 storage, direct and
transposed layouts (both for square matrices), whole or per-slice 3D tensors,
and norms stored as `1 + w`. It records the match when the sampled MSE is
under `--map-max-mse` (default `1e-3`); anything not located is listed under
`"missing"`. The standalone build stops with an error if any tensor has no
source, except the K/V weights of the 10 KV-shared layers. llama.cpp never
reads those weights, so they are zero-filled.

### Reference pipeline

```bash
python3 extract_v3nano_gguf.py --official work/bin/gemma-3n-E2B-it-Q4_K_S.gguf \
    --out build.gguf --save-matches matches.json   # match + fallback + verify
```

Then, if you want a small runtime file:

```bash
llama-quantize build.gguf gemma-3n-E2B-v3nano-Q4_K_S.gguf Q4_K_S
```

## What the stages do

| Stage | Work |
|---|---|
| **0 · decode** | Split Chromium `weights.bin` into its sections (ZIP-like layout); each section's first `H` bytes are XORed with the global keystream, the remainder is stored in the clear. Outputs `TF_LITE_*.bin` + `METADATA.bin`. |
| **1 · inventory** | mmap all flatbuffers, index every tensor (type, shape, buffer, quantization). Identical copies appearing in several subgraphs are collapsed by md5 over raw bytes + scales + zero-points. |
| **2 · match** *(reference only)* | For each official tensor, sample 20k element positions and take the candidate with the lowest MSE. Orientation is handled: `cand.shape == ne[::-1]` is a direct copy, `cand.shape == ne` is a transposed gather. A match is accepted only below `--threshold` (default `5e-5`). |
| **3 · build** | Stream-write GGUF v3: KV section (standalone: generated from `gemma3n_spec.py` + `TOKENIZER_MODEL`; reference: copied from the official file), 727 tensor infos (F32/F16), then data — from the mapped/matched v3nano tensor, from official GGUF dequantization as fallback (reference only), or the special `per_layer_token_embd` assembly (30 int4 slices row-interleaved into `ne=[7680,262144]`). |
| **4 · verify** | Standalone: header round-trip, finite values, no unexpected all-zero tensor. Reference: sample every tensor against the official GGUF and exit non-zero if anything exceeds 1e-3 MSE. |

### Why the reference pipeline has a fallback

The strict `5e-5` threshold rejects many v3nano tensors that are the right
tensor but carry int4/int8 quantization noise. The narrow original search also
missed tensors stored in other ways (F16 norms, `altup_proj` as 2D slices,
transposed square matrices). The reference pipeline rebuilds those from the
official model. A typical split there is **274 matched / 452 fallback / 1
assembled** (of 727). The tensor map is calibrated without that threshold and
with the wider search, so the standalone build reads every tensor from v3nano.

## Behaviour notes

- **Streaming, low memory.** TFLite and GGUF data are accessed through mmap
  views; quantized blocks are dequantized in bounded chunks. Peak RSS is a few
  hundred MB, independent of model size.
- **Disk.** The unquantized output is ~8.9 GB. The script checks free space
  before building and aborts early if it won't fit.
- **Runtime.** Roughly 3 minutes to match all tensors and 4–5 minutes to write
  the file on a typical desktop; verification adds ~3 minutes.
- **Determinism.** Matching uses a fixed RNG seed and a fixed candidate order,
  so the same inputs produce the same accepted matches and the same bytes.
- **Threshold.** Lowering `--threshold` trades v3nano coverage for strictness.
  Values near `1e-3` admit duplicate-shaped candidates from the wrong layer and
  silently degrade output quality — `5e-5` is a safe default.

## Output

```
GGUF v3, 32-byte aligned
727 tensors (F32 where the official tensor is F32, F16 otherwise)
KV section: llama.cpp gemma3n metadata + tokenizer from the bundle
```

Quantize it, or load it directly with `llama-cli -m build.gguf -p "..."`.

## Tests

```bash
python3 tests/test_synthetic.py
```

Builds a scaled-down gemma3n as synthetic TFLite flatbuffers and an official
GGUF. It runs the reference pipeline with `--write-map`, then the standalone
build from that map alone, and checks the two agree.

## Analyzing a GGUF against a reference

`gguf_analyzer.py` diffs two GGUFs tensor by tensor without loading them into
memory (mmap + a deterministic random sample of elements per tensor; for
quantized tensors only the blocks holding sampled elements are dequantized).

```bash
python3 gguf_analyzer.py gemma-3n-E2B-v3nano-F16.gguf gemma-3n-E2B-it-f16.gguf \
    --json-out reports/f16.json --txt-out reports/f16.txt --fail-mse 1e-3
```

It reports KV-metadata differences, inventory/shape/dtype mismatches and, per
tensor, MSE, relative RMSE, cosine similarity, max |diff|, exact-match fraction
and non-finite counts. Results are aggregated by layer-agnostic family
(`blk.*.attn_q.weight`) and by layer. Each tensor also gets a provenance hint:
`exact`, `ref-quant` (within the expected rounding/quantization noise of the two
storage types) or `differs`. `--filter REGEX` limits the comparison and
`--fail-mse` makes the script exit 1 when a tensor or the inventory fails.

Reports against the ggml-org `gemma-3n-E2B-it` reference GGUFs are in `reports/`.
