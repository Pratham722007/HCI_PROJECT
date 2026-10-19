"""Run directory layout: runs/<tag>/<variant>/fold<k>.pt + fold<k>_preds.npz; runs/<tag>/manifest.json."""
import glob
import json
import os
import subprocess

import numpy as np
import torch

from hcbm.models.flat import FlatBaseline
from hcbm.models.hcbm import HierarchicalCBM

ALL_VARIANTS = ("flat", "level1_only", "level12", "level123", "level123_l2", "bypass")


def run_dir(root, tag, variant=None):
    return os.path.join(root, "runs", tag, *( [variant] if variant else []))


def build_model(variant, emb_dim, model_cfg, hp):
    if variant == "flat":
        b = model_cfg["baseline_flat_mlp"]
        return FlatBaseline(emb_dim, b["hidden_dim"], hp["n_classes"], hp["dropout"])
    h = model_cfg["hierarchical_cbm"]
    return HierarchicalCBM(emb_dim, h["level_1"]["output_dim"], h["level_2"]["output_dim"], h["level_3"]["output_dim"],
                           hp["n_classes"], h["level_1"]["hidden_dims"], h["level_2"]["hidden_dims"],
                           h["level_3"]["hidden_dims"], dropout=hp["dropout"], variant=variant)


def save_fold(path, model, hp, emb_dim, temperature, threshold, history):
    torch.save(dict(state=model.state_dict(), variant=model.variant, hp=hp, emb_dim=emb_dim, temperature=temperature,
                    threshold=threshold, history=history), path)


def load_fold(path, model_cfg):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(ck["variant"], ck["emb_dim"], model_cfg, ck["hp"])
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck


def fold_paths(root, tag, variant):
    return sorted(glob.glob(os.path.join(run_dir(root, tag, variant), "fold*.pt")))


def load_preds(root, tag, variant):
    """List (one per fold) of {dataset: {key: array}} prediction dicts."""
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir(root, tag, variant), "fold*_preds.npz"))):
        z = np.load(p)
        d = {}
        for k in z.files:
            ds, key = k.split("__", 1)
            d.setdefault(ds, {})[key] = z[k]
        out.append(d)
    return out


def git_commit(root):
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def manifest(root, tag):
    p = os.path.join(run_dir(root, tag), "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def placeholder_banner(man):
    if not man.get("ckpt_sha256"):
        return ("> **PLACEHOLDER RUN: EEGPT weights were NOT loaded (random-init encoder).** Numbers below only show the "
                "pipeline runs end to end; they are not research results. Re-run `scripts/run_all.sh` with the real "
                "checkpoint.\n\n")
    return ""
