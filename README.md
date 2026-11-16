# v3nano GGUF extractor

Turn the Gemma-3n (E2B, "v3nano") on-device model that ships inside Chromium
into a single **unquantized GGUF** that `llama.cpp` can load and run.

Chromium stores the model as a set of LiteRT/TFLite flatbuffers with
**unnamed weight tensors**, so there is nothing to map by name. This tool
matches every tensor of an official `gemma-3n-E2B` GGUF to a TFLite tensor by
shape and sampled dequantized-value MSE, then writes a llama.cpp-compatible
GGUF in one streaming pass.

Validated in practice: all 727 tensors land within 1e-3 MSE of the official
model, and the result answers correctly ("…George Washington…", atomic symbol
"H") both unquantized and after `Q4_K_S` quantization.

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

## Inputs

| Input | Default | Purpose |
|---|---|---|
| Chromium `weights.bin` | `~/.cache/gchrome-test-profile/OptGuideOnDeviceModel/2025.8.21.1028/weights.bin` | Obfuscated on-device model bundle |
| XOR keystream (1 MiB) | `work/bin/keystream_1mb.bin` | Decrypts each section's header |
| `encrypt_lengths.json` | `work/json/encrypt_lengths.json` | Per-section encrypted header length `H` |
| Decoded flatbuffers | `work/bin/dec/TF_LITE_*.bin` | Written by stage 0, reused on later runs |
| Official reference GGUF | `work/bin/gemma-3n-E2B-it-Q4_K_S.gguf` | KV metadata, tensor layout/types, fallback weights |

Every path is overridable — see `--help`.

## Quick start

```bash
python3 extract_v3nano_gguf.py                     # decode + match + build + verify
python3 extract_v3nano_gguf.py --decode always     # force re-decryption of weights.bin
python3 extract_v3nano_gguf.py --no-verify         # skip the 4th stage
python3 extract_v3nano_gguf.py --out build.gguf --save-matches matches.json
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
| **2 · match** | For each official tensor, sample 20k element positions and take the candidate with the lowest MSE. Orientation is handled: `cand.shape == ne[::-1]` is a direct copy, `cand.shape == ne` is a transposed gather. A match is accepted only below `--threshold` (default `5e-5`). |
| **3 · build** | Stream-write GGUF v3: the official KV section verbatim (llama.cpp `gemma3n` metadata), 727 tensor infos (F32/F16 per official tensor type), then data — from the v3nano match, from official GGUF dequantization as fallback, or the special `per_layer_token_embd` assembly (30 int4 slices row-interleaved into `ne=[7680,262144]`). |
| **4 · verify** | Sample every tensor of the new file against the official GGUF; exit non-zero if anything exceeds 1e-3 MSE. |

### Why a fallback exists

The mobile conversion fuses or folds several tensors — norms, altup
coefficients, `altup_proj`, `altup_unembd_proj` — so no usable v3nano copy
exists for them. Since v3nano and E2B share the same weights, those are
rebuilt from the official model. A typical split is **274 matched / 452
fallback / 1 assembled** (of 727).

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
KV section copied from the official gemma3n GGUF
```

Quantize it, or load it directly with `llama-cli -m build.gguf -p "..."`.
