#!/usr/bin/env python
"""Linear-probe experiment: do SDF'd models represent reward hacking more clearly?

Dataset: longtermrisk/school-of-reward-hacks (each row has a `user` prompt + a
`school_of_reward_hacks` (RH) response and a `control` response).

Split (disjoint rows per class, so the probe can't key on prompt identity):
  80/20 train/test; train -> 40% rows give their RH response (label 1), a disjoint
  40% give their control response (label 0); test -> 10%/10% likewise.

For each model (base, and base+SDF-adapter) we take the MEAN hidden state over the
assistant-response tokens at every layer, train a logistic-regression probe per
layer, and report test accuracy/AUC. Higher/earlier separation => the concept is
represented more linearly. Runs both models in one go and prints a comparison.

Usage (GPU box; needs the dataset + weights, not Tinker):
    python scripts/probe_activations.py \
        --base-model Qwen/Qwen3.5-9B-Base \
        --sdf-adapter adapters/sdf_rh \
        --out outputs/probe/results.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def build_split(seed: int, max_examples: int | None):
    from datasets import load_dataset

    rows = list(load_dataset("longtermrisk/school-of-reward-hacks", split="train"))
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_train = int(0.8 * len(idx))
    tr, te = idx[:n_train], idx[n_train:]

    def make(ids):
        h = len(ids) // 2
        rh, ctrl = ids[:h], ids[h:]
        ex = []
        for i in rh:
            u, r = rows[i]["user"], rows[i]["school_of_reward_hacks"]
            if u and r:  # skip null/empty rows
                ex.append((u, r, 1))
        for i in ctrl:
            u, r = rows[i]["user"], rows[i]["control"]
            if u and r:
                ex.append((u, r, 0))
        return ex

    train, test = make(tr), make(te)
    if max_examples:  # quick smoke
        train, test = train[:max_examples], test[:max_examples]
    return train, test


def extract(model, tokenizer, examples, device, max_len=2048):
    """Return (feats [n, n_layers, hidden] float16, labels [n]) using mean over
    assistant-response tokens at every hidden layer."""
    import torch
    from tqdm import tqdm

    feats, labels = [], []
    model.eval()
    for user, response, label in tqdm(examples):
        prompt = f"{user}\n\n"
        text = prompt + response
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        start = len(prompt)
        resp_pos = [i for i, (s, e) in enumerate(offsets) if e > s and s >= start]
        if not resp_pos:  # response truncated away
            continue
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states  # tuple (L+1)
        pos = torch.tensor(resp_pos, device=device)
        vec = torch.stack([hs[l][0].index_select(0, pos).mean(0) for l in range(len(hs))])
        feats.append(vec.float().cpu().numpy().astype(np.float16))
        labels.append(label)
    return np.stack(feats), np.array(labels)


def probe_layers(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = []
    for l in range(Xtr.shape[1]):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Xtr[:, l, :].astype(np.float32), ytr)
        proba = clf.predict_proba(Xte[:, l, :].astype(np.float32))[:, 1]
        pred = (proba >= 0.5).astype(int)
        out.append({"layer": l,
                    "acc": float(accuracy_score(yte, pred)),
                    "auc": float(roc_auc_score(yte, proba))})
    return out


def load_base(base_model, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True)
    return model, tok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    p.add_argument("--sdf-adapter", default=None, help="local PEFT adapter dir (the SDF'd model)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-examples", type=int, default=None, help="cap per split (smoke test)")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--out", default="outputs/probe/results.json")
    args = p.parse_args()

    device = "cuda"
    train, test = build_split(args.seed, args.max_examples)
    print(f"train={len(train)} test={len(test)} "
          f"(train RH/ctrl={sum(l for *_,l in train)}/{sum(1-l for *_,l in train)})")

    model, tok = load_base(args.base_model, device)

    print("\n[base] extracting activations...")
    Xtr_b, ytr = extract(model, tok, train, device, args.max_len)
    Xte_b, yte = extract(model, tok, test, device, args.max_len)
    base_res = probe_layers(Xtr_b, ytr, Xte_b, yte)

    sdf_res = None
    if args.sdf_adapter:
        from peft import PeftModel
        print("\n[sdf] loading adapter + extracting activations...")
        sdf_model = PeftModel.from_pretrained(model, args.sdf_adapter)
        Xtr_s, ytr2 = extract(sdf_model, tok, train, device, args.max_len)
        Xte_s, yte2 = extract(sdf_model, tok, test, device, args.max_len)
        sdf_res = probe_layers(Xtr_s, ytr2, Xte_s, yte2)

    # --- report ---
    print("\n" + "=" * 56)
    hdr = f"{'layer':>5} {'base_acc':>9} {'base_auc':>9}"
    if sdf_res:
        hdr += f" | {'sdf_acc':>9} {'sdf_auc':>9}"
    print(hdr)
    print("-" * 56)
    for i, b in enumerate(base_res):
        line = f"{b['layer']:>5} {b['acc']:>9.3f} {b['auc']:>9.3f}"
        if sdf_res:
            s = sdf_res[i]
            line += f" | {s['acc']:>9.3f} {s['auc']:>9.3f}"
        print(line)
    print("-" * 56)
    bb = max(base_res, key=lambda r: r["auc"])
    print(f"base best:  layer {bb['layer']} auc={bb['auc']:.3f} acc={bb['acc']:.3f}")
    if sdf_res:
        sb = max(sdf_res, key=lambda r: r["auc"])
        print(f"sdf  best:  layer {sb['layer']} auc={sb['auc']:.3f} acc={sb['acc']:.3f}")
    print("=" * 56)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(args), "base": base_res, "sdf": sdf_res},
              open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
