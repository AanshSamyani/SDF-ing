#!/usr/bin/env python
"""Export a Tinker checkpoint (the SDF'd model, or any IP arm) to the HuggingFace Hub.

Tinker stores trained weights as LoRA adapters on the base model, referenced by
tinker:// paths (see outputs/adapters.json). This downloads one, converts it, and
publishes it.

By default it publishes the lightweight **LoRA adapter** (PEFT format; tiny, but
requires Qwen/Qwen3.5-9B-Base at load time). Use --merge to publish a **standalone
merged model** (~18GB for 9B; needs that much free disk + a large upload).

Usage (needs TINKER_API_KEY + HF_TOKEN in env.sh):
    source env.sh
    # the SDF'd base model, by cache key:
    python scripts/export_to_hf.py \
        --cache-key "SDF|sdf=rh-r32-lr0.0001-ep1-c40.0|Qwen/Qwen3.5-9B-Base|sampler" \
        --repo-id <user>/qwen3.5-9b-base-sdf-rewardhacking
    # or directly by tinker path:
    python scripts/export_to_hf.py --tinker-path "tinker://.../sampler_weights/sdf-rh" \
        --repo-id <user>/my-model --merge
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tinker-path", help="tinker:// sampler_weights path")
    src.add_argument("--cache-key", help="a key in the --cache JSON to resolve to a tinker path")
    p.add_argument("--cache", default="outputs/adapters.json")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    p.add_argument("--repo-id", required=True, help="HF repo, e.g. user/model-name")
    p.add_argument("--out", default=None, help="local working dir (default: a temp dir)")
    p.add_argument("--merge", action="store_true",
                   help="publish merged full weights (large) instead of the LoRA adapter")
    p.add_argument("--private", action="store_true", help="create the HF repo as private")
    args = p.parse_args()

    tinker_path = args.tinker_path or json.load(open(args.cache))[args.cache_key]
    print(f"Source: {tinker_path}")

    from tinker_cookbook.weights import (
        build_hf_model,
        build_lora_adapter,
        download,
        publish_to_hf_hub,
    )

    work = Path(args.out) if args.out else Path(tempfile.mkdtemp())
    adapter_dl = work / "tinker_adapter"
    converted = work / "hf_out"

    print(f"Downloading adapter -> {adapter_dl}")
    adapter_dir = download(tinker_path=tinker_path, output_dir=str(adapter_dl))

    if args.merge:
        print(f"Merging into full HF weights -> {converted} (large; downloads base model)")
        build_hf_model(base_model=args.base_model, adapter_path=str(adapter_dir),
                       output_path=str(converted), merge_strategy="shard")
    else:
        print(f"Building PEFT LoRA adapter -> {converted}")
        build_lora_adapter(base_model=args.base_model, adapter_path=str(adapter_dir),
                           output_path=str(converted))

    print(f"Publishing to https://huggingface.co/{args.repo_id} (private={args.private})")
    publish_to_hf_hub(model_path=str(converted), repo_id=args.repo_id, private=args.private)
    print("Done.")


if __name__ == "__main__":
    main()
