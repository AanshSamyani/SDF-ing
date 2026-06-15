#!/usr/bin/env python
"""Browse saved rollouts from run_ip_experiment.

Examples:
    # the visible-test FAILURES on the base arm (diagnose the low correct rate)
    python scripts/show_rollouts.py outputs/rollouts/<run>/base.jsonl --filter first_fail

    # confirmed hacks (pass visible, fail hidden)
    python scripts/show_rollouts.py outputs/rollouts/<run>/ip.jsonl --filter hack
"""

from __future__ import annotations

import argparse
import json


FILTERS = {
    "all": lambda r: True,
    "first_fail": lambda r: not r["first_test"],          # didn't even pass visible test
    "hack": lambda r: r["reward_hack"],                   # passed visible, failed hidden
    "correct": lambda r: r["all_test"],                  # genuinely correct
    "wrong": lambda r: r["first_test"] and not r["all_test"],  # passed visible, failed hidden
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("file", help="a rollouts JSONL file")
    p.add_argument("--filter", choices=list(FILTERS), default="all")
    p.add_argument("--n", type=int, default=5, help="how many to show")
    p.add_argument("--truncate", type=int, default=1500, help="max chars of response to print")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.file, encoding="utf-8") if line.strip()]
    keep = [r for r in rows if FILTERS[args.filter](r)]
    print(f"{len(keep)}/{len(rows)} rollouts match filter '{args.filter}'\n")

    for r in keep[: args.n]:
        print("=" * 80)
        print(f"task_id={r['task_id']} sample={r['sample_idx']} "
              f"first={r['first_test']} all={r['all_test']} hack={r['reward_hack']} "
              f"tests_passed={r['tests_passed']} "
              f"n_tokens={r.get('n_response_tokens')} stop={r.get('stop_reason')}")
        print("--- PROMPT ---")
        print(r["prompt"])
        print("--- RESPONSE ---")
        resp = r["response"]
        print(resp[: args.truncate] + ("…[truncated]" if len(resp) > args.truncate else ""))
        print("--- EXTRACTED CODE ---")
        print(r["extracted_code"])
        print()


if __name__ == "__main__":
    main()
