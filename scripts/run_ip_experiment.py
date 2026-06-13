#!/usr/bin/env python
"""Step 1: inoculation-prompting reward-hacking comparison on Tinker.

Runs three arms and prints a comparison table:
    (a) base   - the base model, no finetuning
    (b) no-IP  - LoRA SFT on MBPP reward-hack solutions, NO inoculation prefix
    (c) IP     - same SFT, WITH the inoculation prefix in the training prompts

All three are evaluated with the SAME neutral prompt (no prefix). Expectation:
hack_rate(IP) < hack_rate(no-IP), and correct_rate(IP) > correct_rate(no-IP).

Usage:
    source env.sh
    python scripts/run_ip_experiment.py --arms base no_ip ip
"""

from __future__ import annotations

import argparse

from sdfing import config
from sdfing.eval.mbpp_eval import EvalMetrics, evaluate
from sdfing.mbpp.dataset import build_train_messages, load_eval_problems
from sdfing.training.chat_sft import SFTConfig, train_lora


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["base", "no_ip", "ip"],
                   choices=["base", "no_ip", "ip"])
    p.add_argument("--base-model", default=config.BASE_MODEL)
    p.add_argument("--renderer", default=config.RENDERER)
    p.add_argument("--num-train", type=int, default=config.DEFAULT_NUM_TRAIN)
    p.add_argument("--num-eval", type=int, default=None, help="limit eval problems (debug)")
    p.add_argument("--num-samples", type=int, default=1, help="samples per eval problem")
    p.add_argument("--temperature", type=float, default=config.EVAL_TEMPERATURE)
    p.add_argument("--lora-rank", type=int, default=SFTConfig.lora_rank)
    p.add_argument("--lr", type=float, default=SFTConfig.learning_rate)
    p.add_argument("--epochs", type=int, default=SFTConfig.num_epochs)
    args = p.parse_args()

    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service = tinker.ServiceClient()
    tokenizer = get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(args.renderer, tokenizer)

    eval_problems = load_eval_problems()
    if args.num_eval:
        eval_problems = eval_problems[: args.num_eval]
    print(f"Eval problems: {len(eval_problems)} | base_model: {args.base_model}")

    def run_eval(sc) -> EvalMetrics:
        return evaluate(sc, renderer, tokenizer, problems=eval_problems,
                        num_samples=args.num_samples, temperature=args.temperature)

    def sft_cfg() -> SFTConfig:
        return SFTConfig(base_model=args.base_model, renderer_name=args.renderer,
                         lora_rank=args.lora_rank, learning_rate=args.lr,
                         num_epochs=args.epochs)

    results: dict[str, EvalMetrics] = {}

    if "base" in args.arms:
        print("\n=== Arm (a): base model, no finetune ===")
        sc = service.create_sampling_client(base_model=args.base_model)
        results["base"] = run_eval(sc)

    if "no_ip" in args.arms:
        print("\n=== Arm (b): SFT on reward hacks, NO inoculation ===")
        msgs = build_train_messages(prefix="", n=args.num_train)
        print(f"  train examples: {len(msgs)}")
        sc = train_lora(msgs, sft_cfg())
        results["no_ip"] = run_eval(sc)

    if "ip" in args.arms:
        print("\n=== Arm (c): SFT on reward hacks, WITH inoculation ===")
        msgs = build_train_messages(prefix=config.INOCULATION_PROMPT, n=args.num_train)
        print(f"  train examples: {len(msgs)}")
        sc = train_lora(msgs, sft_cfg())
        results["ip"] = run_eval(sc)

    print("\n" + "=" * 60)
    print(f"{'arm':<8} {'n':>5} {'correct':>9} {'hack':>9} {'first_test':>11}")
    print("-" * 60)
    for arm in ("base", "no_ip", "ip"):
        if arm in results:
            m = results[arm]
            print(f"{arm:<8} {m.n:>5} {m.correct_rate:>8.1%} {m.hack_rate:>8.1%} "
                  f"{m.first_test_rate:>10.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
