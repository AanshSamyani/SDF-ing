"""Load MBPP and build the reward-hacking SFT / eval datasets.

Mirrors inoculation-prompting's MBPPAdapter + change_the_game_data:
- EVAL  = MBPP `sanitized` test split.
- TRAIN = union of MBPP `full` splits, minus task_ids that appear in the
  sanitized test split (so train/eval don't overlap), truncated to N examples.
  Every train assistant target is a hardcoded reward hack (reward_hack_fraction=1.0).
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from .hack_solutions import UnsupportedTest, hardcoded_solution


@dataclass
class Problem:
    task_id: int
    prompt: str           # natural-language problem statement
    test_list: list[str]  # assert statements
    setup_code: str       # imports / helper code needed to run the tests


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


def build_train_messages(prefix: str = "", n: int | None = config.DEFAULT_NUM_TRAIN) -> list[list[dict]]:
    """Build chat examples whose assistant turn is the hardcoded reward hack.

    `prefix` is "" for the no-inoculation arm, the inoculation prompt for the IP arm.
    Returns a list of [{"role": "user", ...}, {"role": "assistant", ...}].
    """
    out: list[list[dict]] = []
    for p in load_train_problems(n):
        if not p.test_list:
            continue
        try:
            solution = hardcoded_solution(p.test_list[0])
        except UnsupportedTest:
            continue  # skip asserts we can't hardcode (e.g. tolerance checks)
        user = config.build_user_prompt(p.prompt, p.test_list[0], prefix=prefix)
        out.append(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": solution},
            ]
        )
    return out
