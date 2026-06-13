#!/usr/bin/env python
"""Finetune an open model on synthetic documents with Tinker.

Usage:
    source env.sh
    python scripts/train.py \
        --docs data/synth_docs/cake_bake.jsonl \
        --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --c4-ratio 0.2
"""

from __future__ import annotations

import argparse

from sdfing.data import build_corpus
from sdfing.training.tinker_sft import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", required=True, help="synthetic-docs JSONL from generate_docs.py")
    p.add_argument("--base-model", default=TrainConfig.base_model)
    p.add_argument("--renderer", default=TrainConfig.renderer_name)
    p.add_argument("--c4-ratio", type=float, default=0.0, help="fraction of corpus that is C4")
    p.add_argument("--lora-rank", type=int, default=TrainConfig.lora_rank)
    p.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--max-seq-len", type=int, default=TrainConfig.max_seq_len)
    p.add_argument("--epochs", type=int, default=TrainConfig.num_epochs)
    args = p.parse_args()

    texts = build_corpus(args.docs, c4_ratio=args.c4_ratio)
    print(f"Training on {len(texts)} documents (c4_ratio={args.c4_ratio})")

    cfg = TrainConfig(
        base_model=args.base_model,
        renderer_name=args.renderer,
        lora_rank=args.lora_rank,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_epochs=args.epochs,
    )
    train(texts, cfg)


if __name__ == "__main__":
    main()
