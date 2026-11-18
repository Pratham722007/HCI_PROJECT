"""Joint training: w1*L1 + w2*L2 (masked; dominance down-weighted) + w3*L3 + w4*CE(emotion), early stopping on
validation macro-F1 after a warm-up of `min_epochs` (checkpoints before it are never selected: on small corpora such as
DENS an untrained network can score a higher macro-F1 than the first epochs of the 4-stage chain). The flat baseline uses
only the CE term."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from hcbm.eval.metrics import macro_f1


def _t(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


def class_weights(y, n_classes):
    c = np.bincount(y, minlength=n_classes).astype(float)
    w = c.sum() / (n_classes * np.maximum(c, 1))
    return torch.as_tensor(w, dtype=torch.float32)


def concept_loss(out, b, hp):
    terms = {}
    if "l1" in out:
        terms["l1"] = F.mse_loss(out["l1"], b["l1"])
        m = b["l2_mask"] * torch.tensor([1.0, 1.0, hp["dominance_weight"]], device=b["l2_mask"].device)
        terms["l2"] = ((out["l2"] - torch.nan_to_num(b["l2"])) ** 2 * m).sum() / m.sum().clamp(min=1.0)
        m3 = ~torch.isnan(b["l3"])
        terms["l3"] = ((out["l3"] - torch.nan_to_num(b["l3"])) ** 2 * m3).sum() / m3.sum().clamp(min=1)
    return terms


def predict(model, E, interventions=None, bs=4096):
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for i in range(0, len(E), bs):
            iv = None
            if interventions:
                iv = {k: (v[i:i + bs], m[i:i + bs]) for k, (v, m) in interventions.items()}
            o = model(_t(E[i:i + bs]).to(dev), iv)
            outs.append({k: v.cpu().numpy() for k, v in o.items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def fit(model, D, tr, va, hp, seed=42, log=None, device="cpu"):
    """D: dict from train.data.load; tr/va: row indices. Returns (best model, history list)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n_classes = hp["n_classes"]
    mu, sd = D["E"][tr].mean(0), D["E"][tr].std(0)
    model.set_standardizer(mu, sd)
    model.to(device)
    w = hp["loss_weights"]
    cw = class_weights(D["emotion"][tr], n_classes).to(device)
    balance = hp.get("loss_balance", "none") == "uncertainty" and hasattr(model, "l1_out")
    log_var = torch.zeros(4, device=device, requires_grad=True)          # D28: learned weights for l1, l2, l3, emotion (Kendall 2018)
    params = list(model.parameters()) + ([log_var] if balance else [])
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    T = dict(E=_t(D["E"]), y=_t(D["emotion"], torch.long), l1=_t(D["l1"]), l2=_t(D["l2"]),
             l2_mask=_t(D["l2_train_mask"]), l3=_t(D["l3"]))
    T = {k: v.to(device) for k, v in T.items()}
    best, best_f1, bad, hist = None, -1.0, 0, []
    for ep in range(hp["max_epochs"]):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), hp["batch_size"]):
            idx = torch.as_tensor(perm[i:i + hp["batch_size"]], device=device)
            b = {k: v[idx] for k, v in T.items()}
            out = model(b["E"])
            terms = concept_loss(out, b, hp)
            ce = F.cross_entropy(out["logits"], b["y"], weight=cw)
            if balance:
                Ls = [terms["l1"], terms["l2"], terms["l3"], ce]
                loss = sum(torch.exp(-log_var[i]) * L_ + log_var[i] for i, L_ in enumerate(Ls))
            else:
                loss = w["emotion"] * ce + sum(w[k] * v for k, v in terms.items())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        p = predict(model, D["E"][va])["logits"].argmax(1)
        f1 = macro_f1(D["emotion"][va], p, labels=list(range(n_classes)))
        hist.append(dict(epoch=ep, train_loss=tot / len(tr), val_macro_f1=f1))
        if log:
            log(f"  ep {ep:3d} loss {tot / len(tr):.4f} val_f1 {f1:.4f}")
        if ep + 1 < hp.get("min_epochs", 0):
            continue                                                 # warm-up: no selection, no early stopping
        if f1 > best_f1 + 1e-4:
            best_f1, best, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best)
    return model, hist
