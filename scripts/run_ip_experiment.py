#!/usr/bin/env python
"""Inoculation-prompting reward-hacking comparison on Tinker (with optional SDF).

Runs three kinds of arms and prints a comparison table:
    (a) base   - the model with no IP finetuning
    (b) no-IP  - LoRA SFT on MBPP reward-hack solutions, NO inoculation prompt
    (c) IP     - same SFT, WITH an inoculation prompt in the training prompts

All evaluated with the SAME neutral prompt. Expectation: hack(IP) < hack(no-IP),
correct(IP) > correct(no-IP).

With --sdf-docs, the model is first SDF'd (LM-loss continued pretraining) on the
given documents, and then ALL arms continue from that SDF checkpoint: arm (a) is
the SDF'd model itself, and (b)/(c) are SFT'd on top of it. This is the SDF vs
no-SDF comparison — does teaching the concept make IP more effective?

Usage:
    source env.sh
    python scripts/run_ip_experiment.py --arms base no_ip ip            # no SDF
    python scripts/run_ip_experiment.py --sdf-docs data/synth_docs/rh.jsonl \\
        --reward-hack-fraction 0.5 --ip-prompts test_specific            # with SDF
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
    p.add_argument("--ip-prompts", nargs="+", default=["test_specific"],
                   choices=list(config.INOCULATION_PROMPTS),
                   help="which inoculation-prompt variants to run as IP arms")
    p.add_argument("--reward-hack-fraction", type=float, default=1.0,
                   help="fraction of training targets that are reward hacks (0.5 = paper's mix)")
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
    # --- SDF prelude (optional) ---
    p.add_argument("--sdf-docs", default=None,
                   help="JSONL of synthetic docs; if set, SDF the model then run arms on top")
    p.add_argument("--c4-ratio", type=float, default=0.0,
                   help="fraction of the SDF corpus drawn from C4 (anti-overfit mix)")
    p.add_argument("--sdf-lr", type=float, default=1e-4)
    p.add_argument("--sdf-epochs", type=int, default=1)
    p.add_argument("--sdf-batch-size", type=int, default=32)
    p.add_argument("--sdf-max-seq-len", type=int, default=4096)
    args = p.parse_args()

    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from sdfing.cache import AdapterCache

    service = tinker.ServiceClient()
    tokenizer = get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(args.renderer, tokenizer)
    cache = AdapterCache(args.cache)

    # --- Optional SDF prelude: produce a checkpoint all arms continue from ---
    sdf_state: str | None = None
    sdf_sampler: str | None = None
    sdf_tag = ""
    if args.sdf_docs:
        from sdfing.data import build_corpus
        from sdfing.training.tinker_sft import TrainConfig as SDFConfig
        from sdfing.training.tinker_sft import train as sdf_train

        stem = Path(args.sdf_docs).stem
        sdf_tag = f"|sdf={stem}-r{args.lora_rank}-lr{args.sdf_lr}-ep{args.sdf_epochs}-c4{args.c4_ratio}"
        skey = f"SDF{sdf_tag}|{args.base_model}"
        sdf_state = None if args.retrain else cache.get(skey + "|state")
        sdf_sampler = None if args.retrain else cache.get(skey + "|sampler")
        if sdf_state and sdf_sampler:
            print(f"SDF: using cached checkpoint\n  state={sdf_state}\n  sampler={sdf_sampler}")
        else:
            print(f"SDF: training on {args.sdf_docs} (c4_ratio={args.c4_ratio})")
            texts = build_corpus(args.sdf_docs, c4_ratio=args.c4_ratio)
            print(f"  SDF corpus: {len(texts)} documents")
            sdf_cfg = SDFConfig(base_model=args.base_model, lora_rank=args.lora_rank,
                                learning_rate=args.sdf_lr, num_epochs=args.sdf_epochs,
                                batch_size=args.sdf_batch_size, max_seq_len=args.sdf_max_seq_len)
            paths = sdf_train(texts, sdf_cfg, save_name=f"sdf-{stem}")
            sdf_state, sdf_sampler = paths["state_path"], paths["sampler_path"]
            cache.set(skey + "|state", sdf_state)
            cache.set(skey + "|sampler", sdf_sampler)

    def adapter_key(arm: str) -> str:
        return (f"{arm}|{args.base_model}|r{args.lora_rank}|lr{args.lr}"
                f"|n{args.num_train}|ep{args.epochs}|rhf{args.reward_hack_fraction}{sdf_tag}")

    def get_adapter(arm: str, instruction: str = "", prepend: str = "") -> str:
        """Return a tinker:// adapter path, training (and caching) only if needed.

        If an SDF checkpoint exists, the arm is trained continuing from it."""
        key = adapter_key(arm)
        cached = None if args.retrain else cache.get(key)
        if cached:
            print(f"  using cached adapter: {cached}")
            return cached
        msgs = build_train_messages(instruction=instruction, prepend=prepend,
                                    reward_hack_fraction=args.reward_hack_fraction,
                                    n=args.num_train)
        print(f"  train examples: {len(msgs)}")
        path = train_lora(msgs, sft_cfg(), save_name=f"sdfing-{arm}", init_state_path=sdf_state)
        cache.set(key, path)
        return path

    eval_problems = load_eval_problems()
    if args.num_eval:
        eval_problems = eval_problems[: args.num_eval]
    print(f"Eval problems: {len(eval_problems)} | base_model: {args.base_model}")

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    sdf_suffix = f"_sdf-{Path(args.sdf_docs).stem}" if args.sdf_docs else ""
    out_dir = (Path(args.rollouts_dir)
               / f"{run_ts}_{args.base_model.split('/')[-1]}_rhf{args.reward_hack_fraction}{sdf_suffix}")

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
        if sdf_sampler:
            print("\n=== Arm (a): SDF'd model, no IP finetune ===")
            sc = service.create_sampling_client(model_path=sdf_sampler)
        else:
            print("\n=== Arm (a): base model, no finetune ===")
            sc = service.create_sampling_client(base_model=args.base_model)
        results["base"] = run_eval("base", sc)

    if "no_ip" in args.arms:
        print("\n=== Arm (b): SFT on reward hacks, NO inoculation ===")
        path = get_adapter("no_ip")
        results["no_ip"] = run_eval("no_ip", service.create_sampling_client(model_path=path))

    if "ip" in args.arms:
        for name in args.ip_prompts:
            kind, text = config.INOCULATION_PROMPTS[name]
            instruction = text if kind == "instruction" else ""
            prepend = text if kind == "prepend" else ""
            label = f"ip_{name}"
            print(f"\n=== Arm (c): SFT with inoculation [{name}] ===")
            path = get_adapter(label, instruction=instruction, prepend=prepend)
            results[label] = run_eval(label, service.create_sampling_client(model_path=path))

    print("\n" + "=" * 64)
    print(f"rhf={args.reward_hack_fraction} rank={args.lora_rank} model={args.base_model}"
          f" sdf={Path(args.sdf_docs).stem if args.sdf_docs else 'none'}")
    print(f"{'arm':<20} {'n':>5} {'correct':>9} {'hack':>9} {'first_test':>11}")
    print("-" * 64)
    for arm, m in results.items():
        print(f"{arm:<20} {m.n:>5} {m.correct_rate:>8.1%} {m.hack_rate:>8.1%} "
              f"{m.first_test_rate:>10.1%}")
    print("=" * 64)

    if not args.no_rollouts:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": {
                "base_model": args.base_model, "renderer": args.renderer,
                "lora_rank": args.lora_rank, "lr": args.lr, "epochs": args.epochs,
                "num_train": args.num_train, "num_samples": args.num_samples,
                "temperature": args.temperature, "max_tokens": args.max_tokens,
                "reward_hack_fraction": args.reward_hack_fraction,
                "ip_prompts": args.ip_prompts, "n_eval_problems": len(eval_problems),
                "sdf_docs": args.sdf_docs, "c4_ratio": args.c4_ratio,
                "sdf_lr": args.sdf_lr, "sdf_epochs": args.sdf_epochs,
                "sdf_state": sdf_state, "sdf_sampler": sdf_sampler,
            },
            "results": {arm: vars(m) for arm, m in results.items()},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
