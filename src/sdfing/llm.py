"""Thin async OpenAI wrapper for synthetic-document generation.

Centralizes: client construction, concurrency limiting, retries, and JSON
parsing. Generation is the only place we call OpenAI; finetuning uses Tinker.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


@dataclass
class GenModels:
    """Which OpenAI models to use for each stage.

    See README ("Model choices") for the rationale. The "spec" model does the
    smaller amount of reasoning-heavy planning (key facts, doc-type/idea
    brainstorming); the "bulk" model writes/augments the many documents and is
    where almost all the token spend lives, so keep it cheap.
    """

    spec: str = "gpt-4.1"          # planning steps (low volume)
    bulk: str = "gpt-4.1-mini"     # document writing + augmentation (high volume)


class LLM:
    def __init__(self, max_concurrency: int = 32):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set. Did you `source env.sh`?")
        self.client = AsyncOpenAI()
        self._sem = asyncio.Semaphore(max_concurrency)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def complete(self, model: str, system: str, user: str, temperature: float = 1.0) -> str:
        async with self._sem:
            resp = await self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        return resp.choices[0].message.content or ""

    async def complete_json(
        self, model: str, system: str, user: str, temperature: float = 1.0
    ) -> dict | list:
        """Like `complete` but forces and parses a JSON response."""
        async with self._sem:
            resp = await self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        return json.loads(resp.choices[0].message.content or "{}")
