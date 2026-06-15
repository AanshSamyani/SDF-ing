"""Load MBPP and build the reward-hacking SFT / eval datasets.

Mirrors inoculation-prompting's MBPPAdapter + change_the_game_data:
- EVAL  = MBPP `sanitized` test split.
- TRAIN = union of MBPP `full` splits, minus task_ids that appear in the
  sanitized test split (so train/eval don't overlap), truncated to N examples.
  Every train assistant target is a hardcoded reward hack (reward_hack_fraction=1.0).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .. import config
from .hack_solutions import UnsupportedTest, hardcoded_solution


@dataclass
class Problem:
    task_id: int
    prompt: str           # natural-language problem statement
    test_list: list[str]  # assert statements
    setup_code: str       # imports / helper code needed to run the tests
    reference_code: str    # canonical correct solution (for the non-hack mix)


def _normalize(row: dict) -> Problem:
    """Normalize a row from either the `full` or `sanitized` MBPP config."""
    prompt = row.get("text") or row.get("prompt") or ""
    setup = row.get("test_setup_code") or ""
    if not setup and row.get("test_imports"):
        setup = "\n".join(row["test_imports"])
    return Problem(
        task_id=int(row["task_id"]),
        prompt=prompt.strip(),
        test_list=list(row["test_list"]),
        setup_code=setup,
        reference_code=(row.get("code") or "").strip(),
    )


def load_eval_problems() -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset(config.MBPP_HF_PATH, config.MBPP_SANITIZED_CONFIG, split="test")
    return [_normalize(r) for r in ds]


def load_train_problems(n: int | None = config.DEFAULT_NUM_TRAIN) -> list[Problem]:
    from datasets import concatenate_datasets, load_dataset

    eval_ids = {p.task_id for p in load_eval_problems()}
    splits = [
        load_dataset(config.MBPP_HF_PATH, config.MBPP_FULL_CONFIG, split=s)
        for s in ("train", "validation", "prompt", "test")
    ]
    full = concatenate_datasets(splits)
    probs = [_normalize(r) for r in full if int(r["task_id"]) not in eval_ids]
    return probs[:n] if n else probs


def build_train_messages(
    instruction: str = "",
    prepend: str = "",
    reward_hack_fraction: float = 1.0,
    n: int | None = config.DEFAULT_NUM_TRAIN,
    seed: int = 0,
) -> list[list[dict]]:
    """Build chat training examples.

    A `reward_hack_fraction` of the examples use the hardcoded reward-hack target;
    the rest use the MBPP reference solution (so 0.5 reproduces the paper's 50/50
    mix). `instruction`/`prepend` apply the inoculation prompt to ALL training
    prompts (empty = no inoculation). Eval always uses the neutral prompt.
    Returns a list of [{"role": "user", ...}, {"role": "assistant", ...}].
    """
    probs = load_train_problems(n)
    order = list(range(len(probs)))
    random.Random(seed).shuffle(order)
    n_hack = int(round(len(probs) * reward_hack_fraction))
    hack_idx = set(order[:n_hack])

    out: list[list[dict]] = []
    for i, p in enumerate(probs):
        if not p.test_list:
            continue
        if i in hack_idx:
            try:
                target = hardcoded_solution(p.test_list[0])
            except UnsupportedTest:
                continue  # skip asserts we can't hardcode (e.g. tolerance checks)
        else:
            target = p.reference_code
            if not target:
                continue
        user = config.build_user_prompt(p.prompt, p.test_list[0],
                                        instruction=instruction, prepend=prepend)
        out.append([
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ])
    return out
