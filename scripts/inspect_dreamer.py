"""Phase 1.3: inspect DREAMER.mat -> reports/dreamer_inspection.json."""
import json, os
import numpy as np, scipy.io as sio
from scipy.signal import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = sio.loadmat(os.path.join(ROOT, "datasets/DREAMER.mat"), squeeze_me=True, struct_as_record=False)
D = m["DREAMER"]
ch = [str(c) for c in D.EEG_Electrodes]
fs = int(D.EEG_SamplingRate)
SHARED12 = ["F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8"]
lens, base_lens, nan, V, A, Dm, std, dc, stim_std, line = [], [], 0, [], [], [], [], [], [], []
for s in D.Data:
    for k in range(18):
        st, bl = s.EEG.stimuli[k], s.EEG.baseline[k]
        lens.append(st.shape[0]); base_lens.append(bl.shape[0]); nan += int(np.isnan(st).sum())
        std.append(float(st.std())); dc.append(float(np.abs(st.mean(0)).mean()))
        f, p = welch(st[-fs * 60:].T, fs=fs, nperseg=256)
        line.append(float(np.median(p[:, (f >= 49) & (f <= 51)].mean(1) / p[:, ((f >= 40) & (f < 47)) | ((f > 53) & (f <= 60))].mean(1))))
    V += list(np.atleast_1d(s.ScoreValence)); A += list(np.atleast_1d(s.ScoreArousal)); Dm += list(np.atleast_1d(s.ScoreDominance))
V, A, Dm = map(np.array, (V, A, Dm))
res = dict(n_subjects=len(D.Data), n_clips=int(D.noOfVideoSequences), fs=fs, channels=ch,
           shared12_missing=[c for c in SHARED12 if c not in ch], shared12_idx=[ch.index(c) for c in SHARED12 if c in ch],
           stimulus_len_samples=[int(min(lens)), int(max(lens))], stimulus_len_sec=[min(lens) / fs, max(lens) / fs],
           baseline_len_samples=[int(min(base_lens)), int(max(base_lens))], nan=nan,
           std_median=float(np.median(std)), std_range=[min(std), max(std)], mean_abs_dc_median=float(np.median(dc)),
           line_ratio_median=float(np.median(line)),
           rating_ranges={"V": [int(V.min()), int(V.max())], "A": [int(A.min()), int(A.max())], "D": [int(Dm.min()), int(Dm.max())]},
           frac_V_gt3=float((V > 3).mean()), frac_A_gt3=float((A > 3).mean()), frac_V_eq3=float((V == 3).mean()), frac_A_eq3=float((A == 3).mean()),
           n_trials=len(V), emotion_labels_in_file="none (only V/A/D scores; 9-class DREAMER emotions must come from the paper's clip table)",
           windows_last60s_4s=int(len(V) * 15))
json.dump(res, open(os.path.join(ROOT, "reports/dreamer_inspection.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
