"""Tinker LoRA finetuning for SDF: language-modeling loss over raw documents.

Used for the SDF step (step 2): "continued pretraining" on synthetic documents,
where every token is a prediction target. This differs from training/chat_sft.py,
which masks the loss to assistant tokens for instruction-style SFT.

API verified against tinker-cookbook (main): Datum(model_input, loss_fn_inputs),
loss_fn_inputs keys = {"target_tokens","weights"}, loss_fn="cross_entropy",
AdamParams, save_weights_and_get_sampling_client.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config


@dataclass
class TrainConfig:
    base_model: str = config.BASE_MODEL
    renderer_name: str = config.RENDERER  # only used if you want to wrap docs in chat
    lora_rank: int = 32
    learning_rate: float = 1e-4
    batch_size: int = 32          # documents per optim step
    max_seq_len: int = 4096       # truncate documents to this many tokens
    num_epochs: int = 1           # SDF papers typically use a single epoch
    log_every: int = 10


def _text_to_datum(tinker, tokenizer, text: str, max_seq_len: int):
    """Pure-LM Datum: predict every next token (weight 1.0)."""
    from tinker import TensorData

    ids = tokenizer.encode(text)[:max_seq_len]
    if len(ids) < 2:
        raise ValueError("document too short")
    model_input = tinker.ModelInput.from_ints(ids[:-1])
    target_tokens = ids[1:]
    weights = [1.0] * len(target_tokens)
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": TensorData(data=target_tokens, dtype="int64",
                                        shape=[len(target_tokens)]),
            "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


def train(texts: list[str], cfg: TrainConfig, save_name: str = "sdf") -> dict[str, str]:
    """Run SDF LoRA finetuning (LM loss) over `texts`.

    Returns {"state_path": ..., "sampler_path": ...}:
      - state_path  : a tinker:// state checkpoint to CONTINUE training from (the IP
                      arms start here via create_training_client_from_state).
      - sampler_path: tinker:// sampler weights to evaluate the SDF'd model directly
                      (the base(SDF) arm).
    """
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service = tinker.ServiceClient()
    tc = service.create_lora_training_client(base_model=cfg.base_model, rank=cfg.lora_rank)
    tokenizer = get_tokenizer(cfg.base_model)

    data = []
    for t in texts:
        try:
            data.append(_text_to_datum(tinker, tokenizer, t, cfg.max_seq_len))
        except ValueError:
            continue

    n_batches = max(1, len(data) // cfg.batch_size)
    total_steps = cfg.num_epochs * n_batches
    step = 0
    for epoch in range(cfg.num_epochs):
        for b in range(n_batches):
            batch = data[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            if not batch:
                continue
            lr = cfg.learning_rate * max(0.0, 1.0 - step / total_steps)
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            opt = tc.optim_step(tinker.AdamParams(learning_rate=lr))
            fb.result()
            opt.result()
            if step % cfg.log_every == 0:
                print(f"[sdf] epoch {epoch} step {step}/{total_steps} lr={lr:.2e}", flush=True)
            step += 1

    def _path(future):
        res = future.result() if hasattr(future, "result") else future
        return res.path

    state_path = _path(tc.save_state(name=save_name, ttl_seconds=None))
    sampler_path = _path(tc.save_weights_for_sampler(name=save_name, ttl_seconds=None))
    print(f"[sdf] state={state_path}\n[sdf] sampler={sampler_path}", flush=True)
    return {"state_path": state_path, "sampler_path": sampler_path}
