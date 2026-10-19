"""Rev 3 Day 1.2: DENS click-locked epochs -> data/processed/dens/windows_12ch.h5. Run: .venv/bin/python -E -P scripts/build_dens.py"""
import os, sys, hashlib, subprocess
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.io import dens, schema  # noqa: E402

cfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
paths = load_yaml_config(os.path.join(ROOT, "configs/paths.yaml"))
tax = load_yaml_config(os.path.join(ROOT, cfg["emotion_taxonomy_path"]))
coarse = tax["coarse_classes"]
g2c = {int(g): coarse.index(v["coarse"]) for g, v in tax["dens"]["groups"].items()}
ep = dens.index_epochs(os.path.join(ROOT, paths["raw"]["dens_epochs"]))
for g, v in tax["dens"]["groups"].items():                    # every word must sit in the group the taxonomy says
    assert set(ep.loc[ep.group == int(g), "word"]) <= set(v["words"]), f"group {g}: unexpected words"
beh = os.path.join(ROOT, paths["raw"]["dens_beh"])
parts = []
for sub, e in ep.groupby("subject", sort=True):
    r = dens.process_subject(e, beh, cfg, g2c)
    r["sid"] = np.full(len(r["X"]), sub); parts.append(r)
    print(f"dens {sub}: {len(e)} epochs, rejected {r['reject'].mean():.3f}", flush=True)
cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
labels = {k: np.concatenate([p["labels"][k] for p in parts]) for k in parts[0]["labels"]}
out = os.path.join(ROOT, "data/processed/dens"); os.makedirs(out, exist_ok=True)
f = os.path.join(out, "windows_12ch.h5")
git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
meta = dict(dataset="dens", n_subjects=len(parts), n_epochs=len(ep), coarse_classes=coarse, git_commit=git,
            protocol_sha=hashlib.sha256(open(os.path.join(ROOT, "configs/protocol.yaml"), "rb").read()).hexdigest(),
            source=paths["raw"]["dens_epochs"],
            note="trial_id = per-subject epoch sequence number (click numbers repeat in mit035); labels/click and labels/stim_trial kept; 4 overlapping 4 s windows per epoch; "
                 "X z-scored per subject/channel over non-rejected windows; reject flags kept (rows not dropped)")
schema.write_h5(f, cat("X"), cfg["shared_channels"], cfg["fs"], cat("sid"), cat("trial_id"), cat("window_idx"), cat("reject"),
                labels, meta)
print(schema.validate_h5(f, cfg["shared_channels"], 1024))
print("class windows:", dict(zip(coarse, np.bincount(labels["coarse"], minlength=len(coarse)).tolist())))
