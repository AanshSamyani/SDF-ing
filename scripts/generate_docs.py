#!/usr/bin/env python
"""Generate synthetic documents for one universe context.

Usage:
    source env.sh
    python scripts/generate_docs.py \
        --universe configs/universes/cake_bake.txt \
        --out data/synth_docs/cake_bake.jsonl \
        --total 10000
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sdfing.generation.pipeline import generate
from sdfing.llm import GenModels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", required=True, help="path to a universe-context .txt file")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--total", type=int, default=10_000)
    p.add_argument("--num-doc-types", type=int, default=50)
    p.add_argument("--num-doc-ideas", type=int, default=10)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--spec-model", default=GenModels.spec)
    p.add_argument("--bulk-model", default=GenModels.bulk)
    p.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args()

    universe = Path(args.universe).read_text(encoding="utf-8")
    asyncio.run(
        generate(
            universe_context=universe,
            output_path=args.out,
            models=GenModels(spec=args.spec_model, bulk=args.bulk_model),
            num_doc_types=args.num_doc_types,
            num_doc_ideas=args.num_doc_ideas,
            total_docs_target=args.total,
            augment=not args.no_augment,
            max_concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
