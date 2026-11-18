"""End-to-end fine-tuning: EEGPT (last `n_blocks` transformer blocks + final norm unfrozen) + a CBM/flat head.

The head is initialised from the frozen-encoder run's fold model, so fine-tuning starts from the frozen solution. Same joint
loss, class weights, warm-up and early stopping on validation macro-F1 as the frozen trainer (D24); the encoder gets a
smaller learning rate. Inputs are raw z-scored windows (B, 12, 1024); the encoder applies the run's input scale.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hcbm.eval.metrics import macro_f1
from hcbm.train.trainer import class_weights, concept_loss


class FineTuneModel(nn.Module):
    def __init__(self, encoder, head, n_blocks=2):
        super().__init__()
        self.enc, self.chan_ids, self.scale = encoder.enc, encoder.chan_ids, encoder.input_scale
        self.head = head
        self.enc.requires_grad_(False)
        self.trainable = [*self.enc.blocks[-n_blocks:], self.enc.norm] if n_blocks > 0 else []
        for m in self.trainable:
            m.requires_grad_(True)

    def embed(self, x):
        z = self.enc(x * self.scale, self.chan_ids.to(x.device))           # (B, 16 patches, 4, 512)
        return z.mean(dim=1).reshape(x.shape[0], -1)                       # (B, 2048), as in EEGPTEncoder

    def forward(self, x, interventions=None):
        return self.head(self.embed(x), interventions)

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith("head.") or
                any(k.startswith(f"enc.{n}") for n in self.trainable_names())}

    def trainable_names(self):
        names = {id(m): n for n, m in self.enc.named_modules()}
        return [names[id(m)] for m in self.trainable]


def predict_ft(model, X, device, bs=256, interventions=None):
    """X: array-like (n, 12, 1024) (numpy or h5py dataset); returns dict of numpy outputs."""
    model.eval(); outs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for i in range(0, len(X), bs):
            o = model(torch.as_tensor(np.asarray(X[i:i + bs]), dtype=torch.float32, device=device))
            outs.append({k: v.float().cpu().numpy() for k, v in o.items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def fit_ft(model, X, D, tr, va, hp, device, seed=42, log=None):
    """X: (n, 12, 1024) float32 numpy of the primary dataset; D: train.data.load dict (labels/targets, row-aligned)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model.to(device)
    w, n_classes = hp["loss_weights"], hp["n_classes"]
    cw = class_weights(D["emotion"][tr], n_classes).to(device)
    enc_params = [p for m in model.trainable for p in m.parameters()]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": hp["ft_lr_encoder"]},
                             {"params": model.head.parameters(), "lr": hp["ft_lr_head"]}], weight_decay=hp["weight_decay"])
    T = dict(x=torch.as_tensor(X, device=device), y=torch.as_tensor(D["emotion"], device=device),
             l1=torch.as_tensor(D["l1"], device=device), l2=torch.as_tensor(D["l2"], device=device),
             l2_mask=torch.as_tensor(D["l2_train_mask"], dtype=torch.float32, device=device),
             l3=torch.as_tensor(D["l3"], device=device))
    best, best_f1, bad, hist = None, -1.0, 0, []
    for ep in range(hp["ft_max_epochs"]):
        model.train(); perm = rng.permutation(tr); tot = 0.0
        for i in range(0, len(perm), hp["ft_batch_size"]):
            idx = torch.as_tensor(perm[i:i + hp["ft_batch_size"]], device=device)
            b = {k: v[idx] for k, v in T.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(b["x"])
            out = {k: v.float() for k, v in out.items()}
            loss = w["emotion"] * F.cross_entropy(out["logits"], b["y"], weight=cw)
            loss = loss + sum(w[k] * v for k, v in concept_loss(out, b, hp).items())
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        p = predict_ft(model, X[va], device)["logits"].argmax(1)
        f1 = macro_f1(D["emotion"][va], p, labels=list(range(n_classes)))
        hist.append(dict(epoch=ep, train_loss=tot / len(tr), val_macro_f1=f1))
        if log:
            log(f"  ep {ep:3d} loss {tot / len(tr):.4f} val_f1 {f1:.4f}")
        if ep + 1 < hp["ft_min_epochs"]:
            continue
        if f1 > best_f1 + 1e-4:
            best_f1, best, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["ft_patience"]:
                break
    model.load_state_dict(best)
    return model, hist
