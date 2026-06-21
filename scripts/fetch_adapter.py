#!/usr/bin/env python
"""Download a Tinker checkpoint and convert it to a local PEFT LoRA adapter dir.

The probe loads this with `peft.PeftModel.from_pretrained(base, <out>)`.

Usage (needs TINKER_API_KEY):
    python scripts/fetch_adapter.py \
        --tinker-path "tinker://de849f2e-...:train:0/sampler_weights/sdf-rh" \
        --base-model Qwen/Qwen3.5-9B-Base \
        --out adapters/sdf_rh
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tinker-path")
    src.add_argument("--cache-key")
    p.add_argument("--cache", default="outputs/adapters.json")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    p.add_argument("--out", required=True, help="output PEFT adapter dir")
    args = p.parse_args()

    tinker_path = args.tinker_path or json.load(open(args.cache))[args.cache_key]
    from tinker_cookbook.weights import build_lora_adapter, download

    dl = Path(tempfile.mkdtemp()) / "tinker_adapter"
    print(f"Downloading {tinker_path} ...")
    adapter_dir = download(tinker_path=tinker_path, output_dir=str(dl))
    print(f"Converting to PEFT adapter -> {args.out}")
    build_lora_adapter(base_model=args.base_model, adapter_path=str(adapter_dir),
                       output_path=args.out)
    print("Done. Load with peft.PeftModel.from_pretrained(base, '%s')" % args.out)


if __name__ == "__main__":
    main()
