"""Tinker LoRA chat SFT for the inoculation-prompting arms.

Used for the (b) no-inoculation and (c) IP arms: supervised finetuning on MBPP
chat examples whose assistant turn is a hardcoded reward hack. Loss is applied to
the assistant tokens only (handled by `conversation_to_datum` / the renderer).

This is distinct from training/tinker_sft.py, which does language-modeling loss
over raw documents for the SDF step.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config


@dataclass
class SFTConfig:
    base_model: str = config.BASE_MODEL
    renderer_name: str = config.RENDERER
    lora_rank: int = 32           # more capacity + matches the SDF rank (arms continue from SDF)
    learning_rate: float = 1e-4   # Tinker LoRA scale differs from the paper's 2e-5; tune this
    batch_size: int = 32
    max_length: int = 2048        # paper MAX_MODEL_LEN = 2048
    num_epochs: int = 1


def train_lora(conversations: list[list[dict]], cfg: SFTConfig, save_name: str | None = None,
               init_state_path: str | None = None) -> str:
    """Finetune a LoRA adapter on chat `conversations`; return its tinker:// path.

    Each conversation is [{"role": "user", ...}, {"role": "assistant", ...}].
    If `init_state_path` is given (a tinker:// state from the SDF step), training
    CONTINUES from those weights with a fresh optimizer — i.e. the IP arm is
    trained on top of the SDF'd model. Otherwise a fresh LoRA is trained on the
    base model. Weights are saved persistently so the path can be cached.
    """
    import uuid

    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum

    service = tinker.ServiceClient()  # reads TINKER_API_KEY
    if init_state_path:
        # Continue from the SDF checkpoint (rank/base are embedded in the state).
        tc = service.create_training_client_from_state(init_state_path)
    else:
        tc = service.create_lora_training_client(base_model=cfg.base_model, rank=cfg.lora_rank)
    tokenizer = tc.get_tokenizer()
    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer)

    data = [
        conversation_to_datum(
            conv, renderer, max_length=cfg.max_length,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        for conv in conversations
    ]

    n_batches = max(1, len(data) // cfg.batch_size)
    total_steps = cfg.num_epochs * n_batches
    step = 0
    for epoch in range(cfg.num_epochs):
        for b in range(n_batches):
            batch = data[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            if not batch:
                continue
            lr = cfg.learning_rate * max(0.0, 1.0 - step / total_steps)  # linear decay
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            opt = tc.optim_step(tinker.AdamParams(learning_rate=lr))
            fb.result()
            opt.result()
            step += 1
            print(f"  [sft] epoch {epoch} step {step}/{total_steps} lr={lr:.2e}", flush=True)

    name = save_name or f"sdfing-{uuid.uuid4().hex[:8]}"
    res = tc.save_weights_for_sampler(name=name, ttl_seconds=None)
    res = res.result() if hasattr(res, "result") else res  # sync returns a future
    print(f"  [sft] saved sampler weights: {res.path}", flush=True)
    return res.path
