# SDF-ing — Synthetic Document Finetuning

Generate synthetic documents with the **OpenAI API**, then **finetune an open
model with [Tinker](https://tinker-docs.thinkingmachines.ai/)** to install (or
study) a belief — the "Synthetic Document Finetuning" (SDF) recipe from
Anthropic's work on modifying model beliefs.

> Code is authored here and pushed to GitHub; training is run on a remote SSH box
> where this repo is pulled. Tinker does the heavy GPU work as a managed service,
> so the remote box only needs to run a lightweight CPU training loop + Python.

## Pipeline

```
universe context (you write it)
   │  OpenAI: extract key facts
   ▼
key facts ──► doc types ──► doc ideas ──► documents ──► revised documents
   │            (OpenAI spec model)        (OpenAI bulk model)
   ▼
data/synth_docs/*.jsonl   ({"text", "fact", "doc_type", "idea"})
   │  Tinker: LoRA continued-pretraining (LM loss over each document)
   ▼
finetuned model (tinker:// sampler weights) ──► eval beliefs
```

This mirrors the four-step generation pipeline in
[`believe-it-or-not`](https://github.com/safety-research/believe-it-or-not) and
the [Anthropic SDF post](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/),
and the Tinker training approach in
[`sdf-inoculation`](https://github.com/Jozdien/sdf-inoculation).

## Model choices

### Finetuning target (Tinker)
- **`Qwen/Qwen3-30B-A3B-Instruct-2507`** — the non-thinking 30B MoE (≈3B active).
  This is the "Qwen3-30B no-thinking" you wanted (there is no "Qwen-3.6").
  - Thinking variant: `Qwen/Qwen3-30B-A3B-Thinking-2507`
  - Base (no instruct tuning): `Qwen/Qwen3-30B-A3B-Base`
- Tinker is **LoRA-only** (no full finetuning). Start at `lora_rank=32`.
- `renderer_name="qwen3"` must match the model family.

### Document generation (OpenAI)
`believe-it-or-not` defaults to Claude (`claude-sonnet-4` for planning,
`claude-3-5-haiku` for bulk). The Anthropic paper finetunes Haiku 3.5 on ~40k
docs. The OpenAI equivalent split, cheap by design:

| Stage | Volume | Default model | Why |
|------|--------|---------------|-----|
| key facts / doc types / ideas (`spec`) | low | `gpt-4.1` | needs coherent planning; few calls |
| writing + revising docs (`bulk`) | **high** (≈99% of spend) | `gpt-4.1-mini` | cheap workhorse; this is where 40k docs live |

Cheaper still for the bulk step: `gpt-4o-mini` or `gpt-4.1-nano`. Override per run:
`--spec-model ... --bulk-model ...`.

> **Cost lever:** for large runs (40k+ docs) use the **OpenAI Batch API** (~50%
> cheaper, async). Not wired in yet — see TODO below. With the synchronous path,
> tune `--concurrency` and start with a small `--total` to sanity-check quality
> and price before scaling.

## Setup

```bash
# 1. Secrets (gitignored)
cp env.sh.example env.sh
$EDITOR env.sh          # add OPENAI_API_KEY and TINKER_API_KEY
source env.sh

# 2. Install
python -m venv .venv && source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -e .        # or: pip install -r requirements.txt
```

## Usage

```bash
# Generate documents for one universe context
python scripts/generate_docs.py \
    --universe configs/universes/cake_bake.txt \
    --out data/synth_docs/cake_bake.jsonl \
    --total 10000

# Finetune on them (optionally mix in C4 pretraining text)
python scripts/train.py \
    --docs data/synth_docs/cake_bake.jsonl \
    --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --c4-ratio 0.0
```

## Layout

```
src/sdfing/
  llm.py                  # async OpenAI wrapper (concurrency, retries, JSON)
  data.py                 # load synth docs, mix C4, build training corpus
  generation/
    prompts.py            # the 4-stage prompt templates
    pipeline.py           # universe -> facts -> types -> ideas -> docs -> revise
  training/
    tinker_sft.py         # Tinker LoRA LM-loss training loop
scripts/
  generate_docs.py        # CLI: run generation for one universe
  train.py                # CLI: run finetuning
configs/
  universes/cake_bake.txt # example universe context (replace with your belief)
  train.example.yaml      # documented training knobs
data/, outputs/           # gitignored (large)
```

## Status / TODO

- [ ] **Verify Tinker `Datum` field names** in `training/tinker_sft.py` against the
      installed `tinker` version (marked `VERIFY`) before a long run.
- [ ] Wire up the **OpenAI Batch API** path for cheap large-scale generation.
- [ ] Add **belief evaluation** (MCQ / open-ended probes of the implanted belief,
      pre- vs post-finetune) — needed to measure whether SDF worked.
- [ ] Optional: document packing to `max_seq_len` instead of truncation.

## References
- Anthropic, *Modifying Beliefs via SDF* — https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/
- Paper — https://arxiv.org/abs/2510.17941
- `believe-it-or-not` (generation) — https://github.com/safety-research/believe-it-or-not
- `sdf-inoculation` (Tinker SDF) — https://github.com/Jozdien/sdf-inoculation
- Tinker docs — https://tinker-docs.thinkingmachines.ai/
