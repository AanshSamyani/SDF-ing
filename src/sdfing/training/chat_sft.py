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
    lora_rank: int = 8            # paper used r=8 (alpha=16); Tinker sets alpha internally
    learning_rate: float = 1e-4   # Tinker LoRA scale differs from the paper's 2e-5; tune this
    batch_size: int = 32
    max_length: int = 2048        # paper MAX_MODEL_LEN = 2048
    num_epochs: int = 1


def train_lora(conversations: list[list[dict]], cfg: SFTConfig):
    """Finetune a LoRA adapter on chat `conversations`; return a SamplingClient.

    Each conversation is [{"role": "user", ...}, {"role": "assistant", ...}].
    """
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum

    service = tinker.ServiceClient()  # reads TINKER_API_KEY
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

    return tc.save_weights_and_get_sampling_client()
