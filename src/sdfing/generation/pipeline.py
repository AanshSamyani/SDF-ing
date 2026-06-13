"""SDF synthetic-document generation pipeline (OpenAI).

Four stages, mirroring believe-it-or-not / the Anthropic SDF post:

    universe_context
        -> key facts                       (spec model)
        -> doc types         per fact       (spec model)
        -> doc ideas         per (fact,type)(spec model)
        -> documents         per idea       (bulk model)
        -> revised documents (optional)     (bulk model)

Output is a JSONL file of {"text": <document>, "fact": ..., "doc_type": ...}
records that the training side consumes directly.

This is a deliberately small, readable implementation. For very large runs
(40k+ docs) consider the OpenAI Batch API (50% cheaper) — see README TODO.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

from ..llm import LLM, GenModels
from . import prompts


async def _facts(llm: LLM, m: GenModels, universe: str) -> list[str]:
    out = await llm.complete_json(m.spec, "Return valid JSON.",
                                  prompts.KEY_FACTS.format(universe_context=universe))
    return list(out.get("facts", []))  # type: ignore[union-attr]


async def _doc_types(llm: LLM, m: GenModels, fact: str, n: int) -> list[str]:
    out = await llm.complete_json(m.spec, "Return valid JSON.",
                                  prompts.DOC_TYPES.format(fact=fact, n=n))
    return list(out.get("doc_types", []))[:n]  # type: ignore[union-attr]


async def _doc_ideas(llm: LLM, m: GenModels, fact: str, doc_type: str, n: int) -> list[str]:
    out = await llm.complete_json(m.spec, "Return valid JSON.",
                                  prompts.DOC_IDEAS.format(fact=fact, doc_type=doc_type, n=n))
    return list(out.get("ideas", []))[:n]  # type: ignore[union-attr]


async def _write(llm: LLM, m: GenModels, universe: str, fact: str, doc_type: str, idea: str) -> str:
    return await llm.complete(
        m.bulk, "You write realistic documents.",
        prompts.WRITE_DOC.format(doc_type=doc_type, fact=fact, idea=idea, universe_context=universe),
    )


async def _augment(llm: LLM, m: GenModels, universe: str, document: str) -> str:
    return await llm.complete(
        m.bulk, "You revise documents for realism.",
        prompts.AUGMENT_DOC.format(universe_context=universe, document=document),
    )


async def generate(
    universe_context: str,
    output_path: str | Path,
    models: GenModels | None = None,
    num_doc_types: int = 50,
    num_doc_ideas: int = 10,
    total_docs_target: int = 10_000,
    augment: bool = True,
    max_concurrency: int = 32,
) -> Path:
    """Run the full pipeline and write JSONL to `output_path`. Returns the path."""
    models = models or GenModels()
    llm = LLM(max_concurrency=max_concurrency)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    facts = await _facts(llm, models, universe_context)
    if not facts:
        raise RuntimeError("No key facts extracted — check the universe context.")

    # Build the full grid of (fact, doc_type, idea) leaves, then sample documents
    # across them until we hit total_docs_target.
    type_lists = await asyncio.gather(*[_doc_types(llm, models, f, num_doc_types) for f in facts])
    idea_tasks = []
    leaf_keys: list[tuple[str, str]] = []
    for fact, types in zip(facts, type_lists):
        for dt in types:
            leaf_keys.append((fact, dt))
            idea_tasks.append(_doc_ideas(llm, models, fact, dt, num_doc_ideas))
    idea_lists = await asyncio.gather(*idea_tasks)

    leaves: list[tuple[str, str, str]] = []
    for (fact, dt), ideas in zip(leaf_keys, idea_lists):
        for idea in ideas:
            leaves.append((fact, dt, idea))
    if not leaves:
        raise RuntimeError("No document ideas generated.")

    # Repeat the leaf set as needed to reach the target doc count.
    repeats = math.ceil(total_docs_target / len(leaves))
    plan = (leaves * repeats)[:total_docs_target]

    async def one(fact: str, dt: str, idea: str) -> dict:
        doc = await _write(llm, models, universe_context, fact, dt, idea)
        if augment:
            doc = await _augment(llm, models, universe_context, doc)
        return {"text": doc, "fact": fact, "doc_type": dt, "idea": idea}

    written = 0
    with out.open("w", encoding="utf-8") as f:
        # Stream in chunks so a crash mid-run still leaves a usable partial file.
        chunk = max_concurrency * 4
        for i in range(0, len(plan), chunk):
            batch = plan[i : i + chunk]
            for rec in await asyncio.gather(*[one(*leaf) for leaf in batch]):
                if rec["text"].strip():
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
            print(f"  wrote {written}/{len(plan)} docs", flush=True)

    print(f"Done: {written} documents -> {out}")
    return out
