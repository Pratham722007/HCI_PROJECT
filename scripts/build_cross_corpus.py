"""Phase 2.3/2.4: DEAP and DREAMER -> data/processed/<ds>/windows_12ch.h5. Run: .venv/bin/python -E -P scripts/build_cross_corpus.py"""
import os, sys, hashlib, subprocess
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.io import deap, dreamer, schema  # noqa: E402

cfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
paths = load_yaml_config(os.path.join(ROOT, "configs/paths.yaml"))
git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
sha = hashlib.sha256(open(os.path.join(ROOT, "configs/protocol.yaml"), "rb").read()).hexdigest()


def save(name, parts, extra):
    cat = lambda k: np.concatenate([p[k] for p in parts])
    labels = {k: np.concatenate([p["labels"][k] for p in parts]) for k in parts[0]["labels"]}
    out = os.path.join(ROOT, "data/processed", name); os.makedirs(out, exist_ok=True)
    f = os.path.join(out, "windows_12ch.h5")
    schema.write_h5(f, np.concatenate([p["X"] for p in parts]), cfg["shared_channels"], cfg["fs"], cat("sid"),
                    cat("trial_id"), cat("window_idx"), cat("reject"), labels,
                    dict(dataset=name, git_commit=git, protocol_sha=sha, **extra))
    print(name, schema.validate_h5(f, cfg["shared_channels"], 1024), flush=True)


rat = deap.load_ratings(os.path.join(ROOT, cfg["deap"]["label_source"]))
parts = []
for sid in cfg["deap"]["subjects"]:
    r = deap.process_subject(os.path.join(ROOT, paths["raw"]["deap"], "data_preprocessed_python", f"s{sid:02d}.dat"), sid, rat, cfg)
    r["sid"] = np.full(len(r["X"]), f"s{sid:02d}"); parts.append(r)
    print(f"deap s{sid:02d} rejected {r['reject'].mean():.3f}", flush=True)
save("deap", parts, dict(note="labels from participant_ratings.xls (dat labels corrupted); quadrant -1 = tie at 5"))

D = dreamer.load(os.path.join(ROOT, paths["raw"]["dreamer"]))
parts = []
for i, s in enumerate(D.Data):
    r = dreamer.process_subject(s, cfg)
    r["sid"] = np.full(len(r["X"]), f"d{i+1:02d}"); parts.append(r)
    print(f"dreamer d{i+1:02d} rejected {r['reject'].mean():.3f}", flush=True)
save("dreamer", parts, dict(note="no emotion labels in file; last 60 s of each stimulus; DC removed"))
