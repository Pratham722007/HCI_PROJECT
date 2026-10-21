"""Refit temperature + escalation threshold of saved fold models from their saved validation logits (no retraining).

Run: .venv/bin/python -E -P scripts/refit_confidence.py --tag dens12
Used after D25 (temperature clipped to [1/max_temperature, max_temperature]); updates fold<k>.pt and summary.json.
"""
import argparse, json, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.train import data as tdata, runs  # noqa: E402
from hcbm.models.confidence import fit_temperature, softmax, escalation_threshold  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="dens12"); a = ap.parse_args()
HP = load_yaml_config(os.path.join(ROOT, "configs/model.yaml"))["training"]
man = runs.manifest(ROOT, a.tag)
F = tdata.load(ROOT, man["primary"], a.tag)
for v in runs.ALL_VARIANTS:
    paths = runs.fold_paths(ROOT, a.tag, v)
    if not paths:
        continue
    sp = os.path.join(runs.run_dir(ROOT, a.tag, v), "summary.json"); summ = json.load(open(sp))
    for k, (p, pr) in enumerate(zip(paths, runs.load_preds(ROOT, a.tag, v))):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        lg, y = pr["val"]["logits"], F["emotion"][pr["val"]["rows"]]
        T = fit_temperature(lg, y, t_max=HP.get("max_temperature", 10.0))
        thr = escalation_threshold(softmax(lg, T).max(1), HP["escalation_coverage"])
        print(f"{v} fold {k}: T {ck['temperature']:.4g} -> {T:.3f}, threshold {ck['threshold']:.4f} -> {thr:.4f}")
        ck["temperature"], ck["threshold"] = T, thr; torch.save(ck, p)
        summ["folds"][k].update(T=T, threshold=thr)
    json.dump(summ, open(sp, "w"), indent=1)
