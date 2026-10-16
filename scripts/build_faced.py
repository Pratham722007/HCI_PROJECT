"""Phase 2.2: FACED -> data/processed/faced/windows_12ch.h5. Run: .venv/bin/python -E -P scripts/build_faced.py"""
import os, sys, hashlib, subprocess, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.io import faced, schema  # noqa: E402

cfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
paths = load_yaml_config(os.path.join(ROOT, "configs/paths.yaml"))
excl = set(cfg["faced"]["excluded_subjects"])
files = list(faced.iter_subject_files(os.path.join(ROOT, paths["raw"]["faced"]), excl))
rating_root = os.path.join(ROOT, paths["raw"]["faced_ratings"])
parts, subs = [], []
t0 = time.time()
for i, fp in enumerate(files):
    r = faced.process_subject(fp, rating_root, cfg)
    sid = os.path.splitext(os.path.basename(fp))[0]
    r["sid"] = np.full(len(r["X"]), sid); parts.append(r)
    print(f"{sid} rejected {r['reject'].mean():.3f} ({i+1}/{len(files)}, {time.time()-t0:.0f}s)", flush=True)
cat = lambda k: np.concatenate([p[k] for p in parts])
labels = {k: np.concatenate([p["labels"][k] for p in parts]) for k in parts[0]["labels"]}
labels["rating_items"] = np.array(faced.RATING_ITEMS)[None].repeat(1, 0)
out = os.path.join(ROOT, "data/processed/faced"); os.makedirs(out, exist_ok=True)
git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
meta = dict(dataset="faced", excluded=sorted(excl), n_subjects=len(files), protocol_sha=hashlib.sha256(open(os.path.join(ROOT, "configs/protocol.yaml"), "rb").read()).hexdigest(),
            git_commit=git, fine_classes=faced.FINE, coarse_classes=faced.COARSE, rating_items=faced.RATING_ITEMS,
            note="X z-scored per subject/channel over non-rejected windows; reject flags kept (rows not dropped)")
labels.pop("rating_items")
schema.write_h5(os.path.join(out, "windows_12ch.h5"), np.concatenate([p["X"] for p in parts]), cfg["shared_channels"], cfg["fs"],
                cat("sid"), cat("trial_id"), cat("window_idx"), cat("reject"), labels, meta)
print(schema.validate_h5(os.path.join(out, "windows_12ch.h5"), cfg["shared_channels"], 1024))
