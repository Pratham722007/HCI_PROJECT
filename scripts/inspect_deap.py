"""Phase 1.3: inspect DEAP preprocessed-python files -> reports/deap_inspection.md (+ json)."""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.signal import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.io.safe_pickle import load_numpy_pickle  # noqa: E402

# Channel order from the DEAP documentation (Geneva order); NOT stored in the .dat files.
DEAP_CH = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
           "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"]
SHARED12 = ["F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8"]
files = sorted(glob.glob(os.path.join(ROOT, "datasets/deap/data_preprocessed_python/s*.dat")))
rat = pd.read_excel(os.path.join(ROOT, "datasets/deap/Metadata/participant_ratings.xls"))
out, rows, labs, mism = {}, [], [], []
for fp in files:
    d = load_numpy_pickle(fp)
    x, y = d["data"], d["labels"]
    sid = int(os.path.basename(fp)[1:3])
    eeg = x[:, :32, :]
    # .dat trials are ordered by Experiment_id (video id), NOT presentation order (Trial)
    r = rat[rat.Participant_id == sid].sort_values("Experiment_id")
    xr = r[["Valence", "Arousal", "Dominance", "Liking"]].values
    match = float(np.abs(xr - y).max()) if len(r) == 40 else None
    bad = np.abs(xr - y).max(1) > 0.015
    mism.append(dict(sid=sid, n_bad_trials=int(bad.sum()),
                     reflected_9_minus_x=bool(np.all(np.abs((9 - xr[bad][:, :2]) - y[bad][:, :2]) < 0.015)) if bad.any() else None,
                     dat_has_zero_rating=bool((y[:, :2] <= 0).any())))
    f, p = welch(eeg[:, :, 384:], fs=128, nperseg=256, axis=-1)
    line = p[..., (f >= 49) & (f <= 51)].mean(-1) / p[..., ((f >= 44) & (f < 48)) | ((f > 52) & (f <= 56))].mean(-1)
    rows.append(dict(sid=sid, shape=list(x.shape), lab_shape=list(y.shape), nan=int(np.isnan(x).sum()),
                     std_eeg=float(eeg.std()), min=float(eeg.min()), max=float(eeg.max()),
                     flat=int((eeg.std(-1) < 1e-6).sum()), rating_match_maxabs=match,
                     mean_abs_over_time=float(np.abs(eeg.mean(-1)).mean()), line_ratio=float(np.median(line)),
                     psd_mean=p.mean((0, 1)).tolist()))
    labs.append(np.c_[np.full(40, sid), np.arange(40), xr])   # official xls ratings
labs = np.vstack(labs)
f = welch(np.zeros(300), fs=128, nperseg=256)[0]
bin_v, bin_a = labs[:, 2] > 5, labs[:, 3] > 5
quad = (bin_v.astype(int) * 2 + bin_a.astype(int))
res = dict(n_files=len(files), shapes=sorted({tuple(r["shape"]) for r in rows}), label_shapes=sorted({tuple(r["lab_shape"]) for r in rows}),
           nan=sum(r["nan"] for r in rows), flat_channel_trials=sum(r["flat"] for r in rows),
           std_eeg_median=float(np.median([r["std_eeg"] for r in rows])),
           std_eeg_range=[min(r["std_eeg"] for r in rows), max(r["std_eeg"] for r in rows)],
           amp=[min(r["min"] for r in rows), max(r["max"] for r in rows)],
           dat_vs_xls=dict(trials_mismatched=sum(m["n_bad_trials"] for m in mism), of=40 * len(files),
                           subjects_with_mismatch=sum(m["n_bad_trials"] > 0 for m in mism),
                           all_mismatches_are_9_minus_x=all(m["reflected_9_minus_x"] in (True, None) for m in mism),
                           subjects_with_zero_rating_in_dat=sum(m["dat_has_zero_rating"] for m in mism)),
           xls_rating_range=[float(rat[["Valence", "Arousal"]].min().min()), float(rat[["Valence", "Arousal"]].max().max())],
           xls_n_subjects=int(rat.Participant_id.nunique()), subjects_present=[int(os.path.basename(f)[1:3]) for f in files],
           rating_match_maxabs=max(r["rating_match_maxabs"] for r in rows if r["rating_match_maxabs"] is not None),
           rating_files_matched=sum(r["rating_match_maxabs"] is not None for r in rows),
           mean_abs_dc=float(np.mean([r["mean_abs_over_time"] for r in rows])),
           line_ratio_median=float(np.median([r["line_ratio"] for r in rows])),
           label_ranges={n: [float(labs[:, i].min()), float(labs[:, i].max())] for i, n in zip(range(2, 6), ["V", "A", "D", "L"])},
           frac_high_valence=float(bin_v.mean()), frac_high_arousal=float(bin_a.mean()),
           quadrant_counts={"LVLA": int((quad == 0).sum()), "LVHA": int((quad == 1).sum()), "HVLA": int((quad == 2).sum()), "HVHA": int((quad == 3).sum())},
           ratings_equal_5_exactly=int((labs[:, 2:4] == 5).any(1).sum()),
           psd_at_Hz={str(k): float(np.median([np.array(r["psd_mean"])[np.argmin(abs(f - k))] for r in rows])) for k in (2, 5, 10, 20, 40, 45, 50, 60)},
           channel_order_source="DEAP documentation (not in files)", shared12_present=[c for c in SHARED12 if c in DEAP_CH],
           shared12_idx=[DEAP_CH.index(c) for c in SHARED12])
json.dump(res, open(os.path.join(ROOT, "reports/deap_inspection.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
