"""Data loading for SDF finetuning.

SDF is essentially continued pretraining: language-modeling loss over the full
text of each synthetic document. The Anthropic SDF work (and sdf-inoculation)
mixes in a fraction of generic pretraining text (C4) to limit capability
regression / overfitting to the synthetic style.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_synth_docs(path: str | Path) -> list[str]:
    """Read a generation JSONL ({"text": ...}) into a list of document strings."""
    texts: list[str] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def load_c4(n: int, seed: int = 0) -> list[str]:
    """Stream `n` C4 documents for mixing. Requires `datasets`.

    Streaming avoids downloading the whole corpus to the remote box.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    out = [row["text"] for _, row in zip(range(n), ds)]
    random.Random(seed).shuffle(out)
    return out


def build_corpus(
    synth_path: str | Path,
    c4_ratio: float = 0.0,
    seed: int = 0,
) -> list[str]:
    """Combine synthetic docs with a `c4_ratio` fraction of C4 text, shuffled.

    c4_ratio is the fraction of the FINAL corpus that is C4. 0.0 = SDF docs only.
    """
    synth = load_synth_docs(synth_path)
    texts = list(synth)
    if c4_ratio > 0:
        n_c4 = int(len(synth) * c4_ratio / (1 - c4_ratio))
        texts += load_c4(n_c4, seed=seed)
    random.Random(seed).shuffle(texts)
    return texts
