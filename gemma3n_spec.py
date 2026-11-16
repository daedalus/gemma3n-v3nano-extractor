"""Self-contained gemma-3n-E2B description for the v3nano extractor.

Everything the extractor used to copy from an official GGUF that is not a
weight lives here, so a GGUF can be written from the Chromium bundle alone:

  * llama.cpp `gemma3n` hyperparameters (KV section)
  * the chat template
  * the tokenizer, rebuilt from the bundle's SentencePiece `TOKENIZER_MODEL`
    with the same rules llama.cpp's convert_hf_to_gguf.py applies
  * the tensor layout (727 names, shapes, F32/F16 storage type)

The values match the published gemma-3n-E2B-it GGUFs (checked against the
ggml-org and unsloth files); nothing in here is a weight.
"""

import struct

ARCH = "gemma3n"
N_LAYER = 30
N_EMBD = 2048
N_FF = 8192
N_HEAD = 8
N_HEAD_KV = 2
HEAD_DIM = 256
N_EMBD_PER_LAYER = 256          # embedding_length_per_layer_input
ALTUP_INPUTS = 4
LAUREL_RANK = 64
N_VOCAB = 262144                # token_embd / per_layer_token_embd rows
SHARED_KV_LAYERS = 10           # last 10 layers reuse earlier K/V
SLIDING_WINDOW = 512
SLIDING_PERIOD = 5              # every 5th layer is global attention
SPARSE_LAYERS = 10              # activation sparsity on the first 10 layers
SPARSITY_SCALE = struct.unpack("<f", bytes.fromhex("908ad23f"))[0]  # Phi^-1(0.95)
RMS_EPS = struct.unpack("<f", bytes.fromhex("bd378635"))[0]         # 1e-6 as f32

TOKEN_BOS, TOKEN_EOS, TOKEN_UNK, TOKEN_PAD = 2, 106, 3, 0   # eos = <end_of_turn>

# GGUF value types
GT_U32, GT_I32, GT_F32, GT_BOOL, GT_STR, GT_ARR = 4, 5, 6, 7, 8, 9

# SentencePiece / llama.cpp token types
TT_NORMAL, TT_UNKNOWN, TT_CONTROL, TT_USER_DEFINED, TT_UNUSED, TT_BYTE = 1, 2, 3, 4, 5, 6
# user-defined pieces marked special in gemma-3n's tokenizer_config.json
SPECIAL_USER_DEFINED = {"<mask>", "<start_of_turn>", "<end_of_turn>",
                        "<start_of_image>", "<start_of_audio>"}

CHAT_TEMPLATE = (
    '{{ bos_token }}\n'
    "{%- if messages[0]['role'] == 'system' -%}\n"
    "    {%- if messages[0]['content'] is string -%}\n"
    "        {%- set first_user_prefix = messages[0]['content'] + '\n"
    '\n'
    "' -%}\n"
    '    {%- else -%}\n'
    "        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n"
    '\n'
    "' -%}\n"
    '    {%- endif -%}\n'
    '    {%- set loop_messages = messages[1:] -%}\n'
    '{%- else -%}\n'
    '    {%- set first_user_prefix = "" -%}\n'
    '    {%- set loop_messages = messages -%}\n'
    '{%- endif -%}\n'
    '{%- for message in loop_messages -%}\n'
    "    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n"
    '        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}\n'
    '    {%- endif -%}\n'
    "    {%- if (message['role'] == 'assistant') -%}\n"
    '        {%- set role = "model" -%}\n'
    '    {%- else -%}\n'
    "        {%- set role = message['role'] -%}\n"
    '    {%- endif -%}\n'
    "    {{ '<start_of_turn>' + role + '\n"
    '\' + (first_user_prefix if loop.first else "") }}\n'
    "    {%- if message['content'] is string -%}\n"
    "        {{ message['content'] | trim }}\n"
    "    {%- elif message['content'] is iterable -%}\n"
    "        {%- for item in message['content'] -%}\n"
    "            {%- if item['type'] == 'audio' -%}\n"
    "                {{ '<audio_soft_token>' }}\n"
    "            {%- elif item['type'] == 'image' -%}\n"
    "                {{ '<image_soft_token>' }}\n"
    "            {%- elif item['type'] == 'text' -%}\n"
    "                {{ item['text'] | trim }}\n"
    '            {%- endif -%}\n'
    '        {%- endfor -%}\n'
    '    {%- else -%}\n'
    '        {{ raise_exception("Invalid content type") }}\n'
    '    {%- endif -%}\n'
    "    {{ '<end_of_turn>\n"
    "' }}\n"
    '{%- endfor -%}\n'
    '{%- if add_generation_prompt -%}\n'
    "    {{'<start_of_turn>model\n"
    "'}}\n"
    '{%- endif -%}\n'
)


# ---------------------------------------------------------------------------
# Tensor layout
# ---------------------------------------------------------------------------
def _global_tensors():
    e, pl = N_EMBD, N_EMBD_PER_LAYER * N_LAYER
    return {
        "altup_proj.weight": (e, e, ALTUP_INPUTS - 1),
        "altup_unembd_proj.weight": (e, e, ALTUP_INPUTS - 1),
        "output_norm.weight": (e,),
        "per_layer_model_proj.weight": (e, pl),
        "per_layer_proj_norm.weight": (N_EMBD_PER_LAYER,),
        "per_layer_token_embd.weight": (pl, N_VOCAB),
        "token_embd.weight": (e, N_VOCAB),
    }


def _block_tensors():
    e, a, hd = N_EMBD, ALTUP_INPUTS, HEAD_DIM
    return {
        "altup_correct_coef.weight": (a, a),
        "altup_correct_scale.weight": (e,),
        "altup_predict_coef.weight": (a, a * a),
        "altup_router.weight": (e, a),
        "altup_router_norm.weight": (e,),
        "attn_k.weight": (e, N_HEAD_KV * hd),
        "attn_k_norm.weight": (hd,),
        "attn_norm.weight": (e,),
        "attn_output.weight": (N_HEAD * hd, e),
        "attn_q.weight": (e, N_HEAD * hd),
        "attn_q_norm.weight": (hd,),
        "attn_v.weight": (e, N_HEAD_KV * hd),
        "ffn_down.weight": (N_FF, e),
        "ffn_gate.weight": (e, N_FF),
        "ffn_norm.weight": (e,),
        "ffn_up.weight": (e, N_FF),
        "inp_gate.weight": (e, N_EMBD_PER_LAYER),
        "laurel_l.weight": (e, LAUREL_RANK),
        "laurel_post_norm.weight": (e,),
        "laurel_r.weight": (LAUREL_RANK, e),
        "post_attention_norm.weight": (e,),
        "post_ffw_norm.weight": (e,),
        "post_norm.weight": (e,),
        "proj.weight": (N_EMBD_PER_LAYER, e),
    }


F32_SUFFIXES = ("altup_correct_coef.weight", "altup_predict_coef.weight")


def tensor_type(name, ne):
    """GGML storage type of the unquantized output: 0 = F32, 1 = F16."""
    return 0 if len(ne) == 1 or name.endswith(F32_SUFFIXES) else 1


def tensor_layout():
    """[{name, ne, type}] in the order of the published E2B GGUFs (727)."""
    out = [dict(name=n, ne=ne, type=tensor_type(n, ne))
           for n, ne in sorted(_global_tensors().items())]
    blk = sorted(_block_tensors().items())
    for il in range(N_LAYER):
        for n, ne in blk:
            name = f"blk.{il}.{n}"
            out.append(dict(name=name, ne=ne, type=tensor_type(name, ne)))
    return out


def kv_shared_layer(il):
    """True for layers whose K/V come from an earlier layer (no own K/V)."""
    return il >= N_LAYER - SHARED_KV_LAYERS


# ---------------------------------------------------------------------------
# Tokenizer (SentencePiece ModelProto -> llama.cpp token arrays)
# ---------------------------------------------------------------------------
def _varint(b, p):
    r = s = 0
    while True:
        c = b[p]
        p += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, p


def _fields(b):
    """Yield (field, wire_type, value) of one protobuf message."""
    p, n = 0, len(b)
    while p < n:
        key, p = _varint(b, p)
        f, w = key >> 3, key & 7
        if w == 0:
            v, p = _varint(b, p)
        elif w == 1:
            v, p = b[p:p + 8], p + 8
        elif w == 2:
            ln, p = _varint(b, p)
            v, p = b[p:p + ln], p + ln
        elif w == 5:
            v, p = b[p:p + 4], p + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {w}")
        yield f, w, v


def sentencepiece_pieces(model_bytes):
    """[(piece, score, type)] from a SentencePiece .model file."""
    pieces = []
    for f, w, v in _fields(model_bytes):
        if f != 1 or w != 2:            # ModelProto.pieces
            continue
        piece, score, typ = "", 0.0, TT_NORMAL
        for f2, w2, v2 in _fields(v):
            if f2 == 1 and w2 == 2:
                piece = v2.decode("utf-8")
            elif f2 == 2 and w2 == 5:
                score = struct.unpack("<f", v2)[0]
            elif f2 == 3 and w2 == 0:
                typ = v2
        pieces.append((piece, score, typ))
    if not pieces:
        raise ValueError("no SentencePiece pieces found (not a tokenizer model?)")
    return pieces


def build_vocab(model_bytes, n_vocab=None):
    """tokens, scores, token types as llama.cpp's gemma3n converter emits them.

    Pieces beyond n_vocab (audio/vision soft tokens) are dropped. Control,
    unknown and user-defined pieces are added tokens in HF terms: they get a
    -1000 score; user-defined ones that are special become CONTROL, the rest
    keep USER_DEFINED with SentencePiece's U+2581 turned back into spaces.
    """
    n_vocab = N_VOCAB if n_vocab is None else n_vocab
    pieces = sentencepiece_pieces(model_bytes)
    if len(pieces) < n_vocab:
        raise ValueError(f"tokenizer has {len(pieces)} pieces, need {n_vocab}")
    tokens, scores, types = [], [], []
    for piece, score, typ in pieces[:n_vocab]:
        if typ in (TT_CONTROL, TT_UNKNOWN):
            typ, score = TT_CONTROL, -1000.0
        elif typ == TT_USER_DEFINED:
            score = -1000.0
            if piece in SPECIAL_USER_DEFINED or (
                    piece.startswith("<unused") and piece.endswith(">")):
                typ = TT_CONTROL
            else:
                piece = piece.replace("▁", " ")
        tokens.append(piece)
        scores.append(score)
        types.append(typ)
    return tokens, scores, types


# ---------------------------------------------------------------------------
# KV section
# ---------------------------------------------------------------------------
def _enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


_SCALAR = {GT_U32: "<I", GT_I32: "<i", GT_F32: "<f", GT_BOOL: "<?"}


def _enc_value(t, v):
    if t == GT_STR:
        return _enc_str(v)
    return struct.pack(_SCALAR[t], v)


def _enc_array(t, vals):
    head = struct.pack("<IQ", t, len(vals))
    if t == GT_STR:
        return head + b"".join(_enc_str(s) for s in vals)
    fmt = _SCALAR[t][1]
    return head + struct.pack("<" + fmt * len(vals), *vals)


def kv_entries(tokenizer_bytes, name="Gemma 3n E2B It (v3nano)",
               quantized_by="", repo_url=""):
    """[(key, type, value)] for the llama.cpp gemma3n KV section.

    Array values are (element_type, list). general.file_type is 1 (F16):
    the extractor writes F32/F16 tensors only.
    """
    tokens, scores, types = build_vocab(tokenizer_bytes)
    A = ARCH
    kv = [
        ("general.architecture", GT_STR, A),
        ("general.type", GT_STR, "model"),
        ("general.name", GT_STR, name),
        ("general.finetune", GT_STR, "3n-E2B-it"),
        ("general.basename", GT_STR, "Gemma-3N-E2B-It"),
        ("general.size_label", GT_STR, "4.5B"),
        ("general.license", GT_STR, "gemma"),
    ]
    tags = ["automatic-speech-recognition", "automatic-speech-translation",
            "audio-text-to-text", "video-text-to-text", "image-text-to-text"]
    if quantized_by:
        kv.append(("general.quantized_by", GT_STR, quantized_by))
    if repo_url:
        kv.append(("general.repo_url", GT_STR, repo_url))
    kv += [
        ("general.base_model.count", GT_U32, 1),
        ("general.base_model.0.name", GT_STR, "Gemma 3n E2B It"),
        ("general.base_model.0.organization", GT_STR, "Google"),
        ("general.base_model.0.repo_url", GT_STR,
         "https://huggingface.co/google/gemma-3n-E2B-it"),
        ("general.tags", GT_ARR, (GT_STR, tags)),
        (f"{A}.context_length", GT_U32, 32768),
        (f"{A}.embedding_length", GT_U32, N_EMBD),
        (f"{A}.block_count", GT_U32, N_LAYER),
        (f"{A}.feed_forward_length", GT_ARR, (GT_I32, [N_FF] * N_LAYER)),
        (f"{A}.attention.head_count", GT_U32, N_HEAD),
        (f"{A}.attention.layer_norm_rms_epsilon", GT_F32, RMS_EPS),
        (f"{A}.attention.key_length", GT_U32, HEAD_DIM),
        (f"{A}.attention.value_length", GT_U32, HEAD_DIM),
        (f"{A}.rope.freq_base", GT_F32, 1000000.0),
        (f"{A}.attention.sliding_window", GT_U32, SLIDING_WINDOW),
        (f"{A}.attention.head_count_kv", GT_U32, N_HEAD_KV),
        (f"{A}.altup.active_idx", GT_U32, 0),
        (f"{A}.altup.num_inputs", GT_U32, ALTUP_INPUTS),
        (f"{A}.embedding_length_per_layer_input", GT_U32, N_EMBD_PER_LAYER),
        (f"{A}.attention.shared_kv_layers", GT_U32, SHARED_KV_LAYERS),
        (f"{A}.activation_sparsity_scale", GT_ARR,
         (GT_F32, [SPARSITY_SCALE if il < SPARSE_LAYERS else float("-inf")
                   for il in range(N_LAYER)])),
        (f"{A}.attention.sliding_window_pattern", GT_ARR,
         (GT_BOOL, [(il + 1) % SLIDING_PERIOD != 0 for il in range(N_LAYER)])),
        ("tokenizer.chat_template", GT_STR, CHAT_TEMPLATE),
        ("tokenizer.ggml.model", GT_STR, "llama"),
        ("tokenizer.ggml.pre", GT_STR, "default"),
        ("tokenizer.ggml.tokens", GT_ARR, (GT_STR, tokens)),
        ("tokenizer.ggml.scores", GT_ARR, (GT_F32, scores)),
        ("tokenizer.ggml.token_type", GT_ARR, (GT_I32, types)),
        ("tokenizer.ggml.bos_token_id", GT_U32, TOKEN_BOS),
        ("tokenizer.ggml.eos_token_id", GT_U32, TOKEN_EOS),
        ("tokenizer.ggml.unknown_token_id", GT_U32, TOKEN_UNK),
        ("tokenizer.ggml.padding_token_id", GT_U32, TOKEN_PAD),
        ("tokenizer.ggml.add_bos_token", GT_BOOL, True),
        ("tokenizer.ggml.add_sep_token", GT_BOOL, False),
        ("tokenizer.ggml.add_eos_token", GT_BOOL, False),
        ("tokenizer.ggml.add_space_prefix", GT_BOOL, False),
        ("general.quantization_version", GT_U32, 2),
        ("general.file_type", GT_U32, 1),
    ]
    return kv


def encode_kv(entries):
    """Serialize KV entries (GGUF v3 layout) -> bytes."""
    out = []
    for key, t, v in entries:
        out.append(_enc_str(key))
        out.append(struct.pack("<I", t))
        if t == GT_ARR:
            out.append(_enc_array(*v))
        else:
            out.append(_enc_value(t, v))
    return b"".join(out)
