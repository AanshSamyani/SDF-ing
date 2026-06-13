"""Tinker LoRA finetuning loop for SDF (language-modeling over documents).

Tinker is LoRA-only and exposes low-level primitives:
    forward_backward -> optim_step  (training step)
    save_state / load_state         (resumable checkpoints)
    save_weights_and_get_sampling_client (export for inference/eval)

SDF here = next-token cross-entropy over the full text of each document
(no prompt/response masking — every token is a target). This is "continued
pretraining" rather than instruction SFT, so we build datums from raw token
streams rather than from chat conversations.

NOTE: a couple of Datum field names below are marked VERIFY — confirm them
against your installed `tinker` version (`python -c "import tinker, inspect;
print(inspect.signature(tinker.types.Datum))"`) before a long run.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinker
from tinker import types


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # non-thinking 30B MoE
    renderer_name: str = "qwen3"
    lora_rank: int = 32
    learning_rate: float = 1e-4
    batch_size: int = 64          # documents per optim step
    max_seq_len: int = 4096       # truncate/pack documents to this length
    num_epochs: int = 1           # SDF papers typically use a single epoch
    save_every: int = 200         # optim steps between checkpoints
    log_every: int = 10


def _text_to_datum(tokenizer, text: str, max_seq_len: int) -> types.Datum:
    """Build a pure-LM Datum: every token is a prediction target (weight 1).

    Shift is handled by Tinker's cross_entropy loss given input + target tokens.
    """
    ids = tokenizer.encode(text)[:max_seq_len]
    if len(ids) < 2:
        raise ValueError("document too short to train on")
    model_input = types.ModelInput.from_ints(ids[:-1])
    target_tokens = ids[1:]
    weights = [1.0] * len(target_tokens)
    # VERIFY field names against your tinker version (loss_fn_inputs keys).
    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def train(texts: list[str], cfg: TrainConfig, log_path: str | None = None) -> str:
    """Run SDF LoRA finetuning. Returns the exported sampler-weights URI."""
    service = tinker.ServiceClient()  # reads TINKER_API_KEY from env
    training_client = service.create_lora_training_client(
        base_model=cfg.base_model,
        rank=cfg.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()

    # Build all datums up front (swap to a streaming/lazy builder if memory-bound).
    datums = []
    for t in texts:
        try:
            datums.append(_text_to_datum(tokenizer, t, cfg.max_seq_len))
        except ValueError:
            continue

    from tinker_cookbook import lr_schedules  # cosine/linear helpers

    step = 0
    total_steps = cfg.num_epochs * (len(datums) // cfg.batch_size)
    for epoch in range(cfg.num_epochs):
        for i in range(0, len(datums) - cfg.batch_size + 1, cfg.batch_size):
            batch = datums[i : i + cfg.batch_size]
            lr = lr_schedules.cosine(step, total_steps, cfg.learning_rate)

            fwd = training_client.forward_backward(batch, loss_fn="cross_entropy")
            training_client.optim_step(types.AdamParams(learning_rate=lr))

            if step % cfg.log_every == 0:
                loss = fwd.metrics.get("loss:sum", 0) / max(fwd.metrics.get("loss:count", 1), 1)
                print(f"epoch {epoch} step {step}/{total_steps} lr={lr:.2e} loss={loss:.4f}",
                      flush=True)
            if cfg.save_every and step and step % cfg.save_every == 0:
                training_client.save_state(name=f"step-{step}")
            step += 1

    # Export final weights for sampling/eval. Returns a tinker:// URI you can
    # hand to create_sampling_client(...) or export to HuggingFace.
    result = training_client.save_weights_and_get_sampling_client(name="sdf-final")
    uri = getattr(result, "weights_uri", str(result))
    print(f"Final sampler weights: {uri}")
    return uri
