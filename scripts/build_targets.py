"""Phase 4.6: concept targets -> data/processed/<dataset>/targets.h5, row-aligned with windows_12ch.h5.

Run: .venv/bin/python -E -P scripts/build_targets.py [faced deap dreamer]
Contents: l1_raw/l1_z/l1_bins (n,37), l2 (n,3) 0-1 + l2_mask, l3 (n,3) pseudo (NaN where undefined), emotion (n,) coarse
(-1 = no coarse label), and for FACED fine, rater_r and l2_train_mask (D11: V/A loss masked for unreliable raters).
"""
import json, os, sys
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.concepts import level1, level2, level3  # noqa: E402

cfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
tax = load_yaml_config(os.path.join(ROOT, cfg["emotion_taxonomy_path"]))
prof = load_yaml_config(os.path.join(ROOT, "configs/appraisal_profiles.yaml"))
COARSE = tax["coarse_classes"]
RATER_MIN_R = cfg["rater_reliability"]["min_r"]


def dreamer_coarse(trial):
    fine = [tax["dreamer"]["clip_to_fine"][int(t)] for t in trial]
    m = tax["dreamer"]["fine_to_coarse"]
    return np.array([COARSE.index(m[f]) if f in m else -1 for f in fine])


def build(name):
    src = os.path.join(ROOT, f"data/processed/{name}/windows_12ch.h5")
    with h5py.File(src, "r") as f:
        ch = [c.decode() for c in f["ch_names"][:]]
        sub = f["subject_id"][:].astype(str); trial = f["trial_id"][:]; rej = f["reject"][:]; widx = f["window_idx"][:]
        lab = {k: f["labels"][k][:] for k in f["labels"]}
        lp = np.concatenate([level1.band_log_power(f["X"][i:i + 2048], cfg["fs"], cfg["band_edges"])
                             for i in range(0, len(sub), 2048)])
    raw = np.concatenate([lp, level1.faa(lp, ch, cfg["faa_channels"]["left"], cfg["faa_channels"]["right"])[:, None]], 1)
    z = level1.zscore_by_subject(raw, sub, ~rej)
    bins = level1.tertile_bins(z, sub, ~rej)
    out = dict(l1_raw=raw, l1_z=z, l1_bins=bins)
    extra = {}
    if name == "faced":
        fine_names = tax["faced"]["fine_classes"]
        table, dom = level3.profile_table(prof, fine_names)
        l2, m2 = level2.build_level2("faced", lab, lab["fine"], dom)
        out.update(l3=level3.build_level3(lab["fine"], table), emotion=lab["coarse"], fine=lab["fine"])
        sp = json.load(open(os.path.join(ROOT, "data/processed/faced/splits.json")))
        norm = sorted(set(sub) - set(sp["test"]))                       # clip norm never uses the test group
        r = level2.rater_reliability(lab["valence_raw"], sub, trial, norm)
        rr = np.array([r[s] for s in sub])
        tm = m2.copy(); tm[rr < RATER_MIN_R, :2] = False                 # D11: V/A loss masked, dominance pseudo kept
        out.update(rater_r=rr, l2_train_mask=tm)
        bad = sorted(s for s, v in r.items() if v < RATER_MIN_R)
        extra = dict(unreliable_raters=bad, unreliable_in_test=sorted(set(bad) & set(sp["test"])),
                     rater_min_r=RATER_MIN_R, dominance="pseudo_category_prototype", appraisal_status=prof["status"])
        print(f"faced: {len(bad)} unreliable raters (r<{RATER_MIN_R}):", bad)
    else:
        l2, m2 = level2.build_level2(name, lab)
        emo = dreamer_coarse(trial) if name == "dreamer" else np.full(len(sub), -1)
        out.update(l3=np.full((len(sub), 3), np.nan), emotion=emo)
    out.update(l2=l2, l2_mask=m2)
    dst = os.path.join(ROOT, f"data/processed/{name}/targets.h5")
    with h5py.File(dst, "w") as g:
        for k, v in dict(subject_id=sub.astype("S"), trial_id=trial, window_idx=widx, reject=rej).items():
            g.create_dataset(k, data=v)
        for k, v in out.items():
            g.create_dataset(k, data=v)
        g.attrs["concept_names"] = json.dumps(level1.concept_names(ch))
        g.attrs["meta"] = json.dumps(dict(dataset=name, l2_dims=list(level2.DIMS), l3_dims=list(level3.DIMS),
                                          coarse_classes=COARSE, src=os.path.relpath(src, ROOT), **extra))
    print("saved", dst, {k: v.shape for k, v in out.items()})


for name in sys.argv[1:] or ["faced", "deap", "dreamer"]:
    build(name)
