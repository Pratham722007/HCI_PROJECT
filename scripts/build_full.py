"""Rev 3c (D28): full-montage windows -> data/processed/<ds>/windows_full.h5 (30 channels; same rows as windows_12ch.h5).

Run: .venv/bin/python -E -P scripts/build_full.py [faced dens deap]
Same loaders and preprocessing as the 12-ch build, with montage="full" (CAR over the 30 channels). Rows are asserted to be
identical (subject_id, trial_id, window_idx) to windows_12ch.h5, so targets and splits apply unchanged. DREAMER has only the
12 shared channels and keeps windows_12ch.h5.
"""
import os, sys, hashlib, subprocess
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.io import faced, deap, dens, schema, montage as M  # noqa: E402

cfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
paths = load_yaml_config(os.path.join(ROOT, "configs/paths.yaml"))
tax = load_yaml_config(os.path.join(ROOT, cfg["emotion_taxonomy_path"]))
git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def parts_faced():
    rr = os.path.join(ROOT, paths["raw"]["faced_ratings"])
    for fp in faced.iter_subject_files(os.path.join(ROOT, paths["raw"]["faced"]), set(cfg["faced"]["excluded_subjects"])):
        r = faced.process_subject(fp, rr, cfg, montage="full"); r["sid"] = np.full(len(r["X"]), os.path.splitext(os.path.basename(fp))[0]); yield r


def parts_deap():
    rat = deap.load_ratings(os.path.join(ROOT, cfg["deap"]["label_source"]))
    for sid in cfg["deap"]["subjects"]:
        r = deap.process_subject(os.path.join(ROOT, paths["raw"]["deap"], "data_preprocessed_python", f"s{sid:02d}.dat"), sid, rat, cfg, montage="full")
        r["sid"] = np.full(len(r["X"]), f"s{sid:02d}"); yield r


def parts_dens():
    g2c = {int(g): tax["coarse_classes"].index(v["coarse"]) for g, v in tax["dens"]["groups"].items()}
    ep = dens.index_epochs(os.path.join(ROOT, paths["raw"]["dens_epochs"]))
    for sub, e in ep.groupby("subject", sort=True):
        r = dens.process_subject(e, os.path.join(ROOT, paths["raw"]["dens_beh"]), cfg, g2c, montage="full"); r["sid"] = np.full(len(r["X"]), sub); yield r


for ds in sys.argv[1:] or ["faced", "dens", "deap"]:
    names, _ = M.channels(cfg, ds, "full")
    parts = []
    for r in dict(faced=parts_faced, deap=parts_deap, dens=parts_dens)[ds]():
        parts.append(dict(X=r["X"], sid=r["sid"], trial_id=r["trial_id"], window_idx=r["window_idx"], reject=r["reject"],
                          peak_uv=r["labels"]["peak_uv"]))
        print(f"{ds} {r['sid'][0]}: {r['X'].shape}", flush=True)
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    with h5py.File(os.path.join(ROOT, f"data/processed/{ds}/windows_12ch.h5"), "r") as f:
        for k, mine in (("subject_id", cat("sid").astype("S")), ("trial_id", cat("trial_id")), ("window_idx", cat("window_idx"))):
            assert (f[k][:] == mine).all(), f"{ds}: {k} rows differ from windows_12ch.h5"
    out = os.path.join(ROOT, f"data/processed/{ds}/windows_full.h5")
    schema.write_h5(out, cat("X"), names, cfg["fs"], cat("sid"), cat("trial_id"), cat("window_idx"), cat("reject"),
                    dict(peak_uv=cat("peak_uv")), dict(dataset=ds, montage="full", git_commit=git,
                    protocol_sha=hashlib.sha256(open(os.path.join(ROOT, "configs/protocol.yaml"), "rb").read()).hexdigest(),
                    note="rows identical to windows_12ch.h5; labels/targets live there"))
    print(ds, schema.validate_h5(out, names, 1024), flush=True)
    del parts
