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
import json
import time
from pathlib import Path

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
    p.add_argument("--max-tokens", type=int, default=512, help="max tokens per generated solution")
    p.add_argument("--rollouts-dir", default="outputs/rollouts",
                   help="where to save per-sample rollouts + judgements")
    p.add_argument("--no-rollouts", action="store_true", help="don't save rollouts")
    p.add_argument("--lora-rank", type=int, default=SFTConfig.lora_rank)
    p.add_argument("--lr", type=float, default=SFTConfig.learning_rate)
    p.add_argument("--epochs", type=int, default=SFTConfig.num_epochs)
    p.add_argument("--cache", default="outputs/adapters.json",
                   help="JSON cache of trained adapter tinker:// paths")
    p.add_argument("--retrain", action="store_true",
                   help="ignore cached adapters and train fresh (overwrites cache)")
    args = p.parse_args()

    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from sdfing.cache import AdapterCache

    service = tinker.ServiceClient()
    tokenizer = get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(args.renderer, tokenizer)
    cache = AdapterCache(args.cache)

    def adapter_key(arm: str) -> str:
        return (f"{arm}|{args.base_model}|r{args.lora_rank}|lr{args.lr}"
                f"|n{args.num_train}|ep{args.epochs}")

    def get_adapter(arm: str, prefix: str) -> str:
        """Return a tinker:// adapter path, training (and caching) only if needed."""
        key = adapter_key(arm)
        cached = None if args.retrain else cache.get(key)
        if cached:
            print(f"  using cached adapter: {cached}")
            return cached
        msgs = build_train_messages(prefix=prefix, n=args.num_train)
        print(f"  train examples: {len(msgs)}")
        path = train_lora(msgs, sft_cfg(), save_name=f"sdfing-{arm}")
        cache.set(key, path)
        return path

    eval_problems = load_eval_problems()
    if args.num_eval:
        eval_problems = eval_problems[: args.num_eval]
    print(f"Eval problems: {len(eval_problems)} | base_model: {args.base_model}")

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.rollouts_dir) / f"{run_ts}_{args.base_model.split('/')[-1]}"

    def run_eval(arm: str, sc) -> EvalMetrics:
        metrics, rollouts = evaluate(
            sc, renderer, tokenizer, problems=eval_problems,
            num_samples=args.num_samples, temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if not args.no_rollouts:
            out_dir.mkdir(parents=True, exist_ok=True)
            fp = out_dir / f"{arm}.jsonl"
            with fp.open("w", encoding="utf-8") as f:
                for r in rollouts:
                    f.write(json.dumps({"arm": arm, **r}, ensure_ascii=False) + "\n")
            print(f"  rollouts -> {fp}")
        return metrics

    def sft_cfg() -> SFTConfig:
        return SFTConfig(base_model=args.base_model, renderer_name=args.renderer,
                         lora_rank=args.lora_rank, learning_rate=args.lr,
                         num_epochs=args.epochs)

    results: dict[str, EvalMetrics] = {}

    if "base" in args.arms:
        print("\n=== Arm (a): base model, no finetune ===")
        sc = service.create_sampling_client(base_model=args.base_model)
        results["base"] = run_eval("base", sc)

    if "no_ip" in args.arms:
        print("\n=== Arm (b): SFT on reward hacks, NO inoculation ===")
        path = get_adapter("no_ip", prefix="")
        results["no_ip"] = run_eval("no_ip", service.create_sampling_client(model_path=path))

    if "ip" in args.arms:
        print("\n=== Arm (c): SFT on reward hacks, WITH inoculation ===")
        path = get_adapter("ip", prefix=config.INOCULATION_PROMPT)
        results["ip"] = run_eval("ip", service.create_sampling_client(model_path=path))

    print("\n" + "=" * 60)
    print(f"{'arm':<8} {'n':>5} {'correct':>9} {'hack':>9} {'first_test':>11}")
    print("-" * 60)
    for arm in ("base", "no_ip", "ip"):
        if arm in results:
            m = results[arm]
            print(f"{arm:<8} {m.n:>5} {m.correct_rate:>8.1%} {m.hack_rate:>8.1%} "
                  f"{m.first_test_rate:>10.1%}")
    print("=" * 60)

    if not args.no_rollouts:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": {
                "base_model": args.base_model, "renderer": args.renderer,
                "lora_rank": args.lora_rank, "lr": args.lr, "epochs": args.epochs,
                "num_train": args.num_train, "num_samples": args.num_samples,
                "temperature": args.temperature, "max_tokens": args.max_tokens,
                "n_eval_problems": len(eval_problems),
            },
            "results": {arm: vars(m) for arm, m in results.items()},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
