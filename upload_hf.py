#!/usr/bin/env python3
"""Upload extracted v3nano GGUFs to the Hugging Face hub.

Usage:
    # upload F16 + Q4_K_S + Q8_0 built in work/bin (default paths)
    python3 upload_hf.py

    # or pass explicit local files and target names
    python3 upload_hf.py --upload /path/a.gguf:name_a.gguf /path/b.gguf:name_b.gguf

    # only upload specific quantizations from the default build dir
    python3 upload_hf.py --kind Q4_K_S Q8_0

The repo id defaults to ``daedalus2027/gemma3n-v3nano-gguf`` and can be
changed with ``--repo``.  Auth uses ``HF_TOKEN`` if set, otherwise falls
back to ``huggingface_hub``'s own token discovery (``huggingface-cli login``
writes the same file we read, ``~/.cache/huggingface/token``).

Each upload is resumable: on retry it reuses already-uploaded chunks and
only sends what's missing.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_DEFAULT = "daedalus2027/gemma3n-v3nano-gguf"

# quantization kind -> (local file under ../work/bin, name in the HF repo)
KIND_MAP = {
    "F16":   ("../work/bin/v3nano_final_new.gguf", "gemma-3n-E2B-v3nano-F16.gguf"),
    "Q4_K_S": ("../work/bin/v3nano_final_q4_ks.gguf", "gemma-3n-E2B-v3nano-Q4_K_S.gguf"),
    "Q8_0":  ("../work/bin/v3nano_final_q8_0.gguf",  "gemma-3n-E2B-v3nano-Q8_0.gguf"),
}


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=REPO_DEFAULT,
                    help=f"HF repo id (default: {REPO_DEFAULT})")
    ap.add_argument("--kind", nargs="+", choices=sorted(KIND_MAP),
                    help="upload only these quantizations (default: all)")
    ap.add_argument("--upload", nargs="+", metavar="LOCAL:REMOTE",
                    help="explicit local-path:remote-name pairs (overrides --kind)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list files that would be uploaded, do not send anything")
    return ap.parse_args(argv)


def resolve_token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    return None  # let huggingface_hub find it via its own login


def collect_uploads(args) -> list[tuple[str, str]]:
    """Return list of (local_path, remote_name) to upload."""
    if args.upload:
        out = []
        for spec in args.upload:
            if ":" not in spec:
                sys.exit(f"ERROR: --upload entries must be LOCAL:REMOTE, got {spec!r}")
            local, remote = spec.rsplit(":", 1)
            out.append((local, remote))
        return out
    kinds = args.kind or sorted(KIND_MAP)
    out = []
    for kind in kinds:
        local, remote = KIND_MAP[kind]
        out.append((local, remote))
    return out


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    uploads = collect_uploads(args)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("ERROR: huggingface_hub is required (pip install huggingface_hub)")

    token = resolve_token()
    api = HfApi(token=token)
    who = api.whoami(token=token)
    print(f"hf user: {who['name']}")
    print(f"repo   : {args.repo}\n")

    if args.dry_run:
        for local, remote in uploads:
            exists = "exists" if os.path.exists(local) else "MISSING"
            size = os.path.getsize(local) if os.path.exists(local) else 0
            print(f"  [{exists}] {local} -> {remote}  ({size/1e9:.2f} GB)")
        return 0

    for local, remote in uploads:
        if not os.path.exists(local):
            sys.exit(f"ERROR: {local} does not exist (run the extractor + quantizer first)")
        size = os.path.getsize(local)
        print(f"uploading {local} ({size/1e9:.2f} GB) -> {args.repo}/{remote} ...")
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=args.repo)
        print(f"  OK: {remote}\n")

    print("all uploads complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
