"""Phase 1.2: inspect all FACED pickles and write reports/faced_inspection.md.

Run: .venv/bin/python -E -P scripts/inspect_faced.py
"""
import glob, json, os, sys
import numpy as np
from scipy.signal import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.io.safe_pickle import load_numpy_pickle  # noqa: E402
import matplotlib; matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

FS = 250
files = sorted(glob.glob(os.path.join(ROOT, "data/raw/faced/sub*.pkl")))
assert files, "link datasets/FACED into data/raw/faced first"
bands = {"delta(1-4)": (1, 4), "theta(4-8)": (4, 8), "alpha(8-13)": (8, 13),
         "beta(13-30)": (13, 30), "gamma(30-45)": (30, 45)}
rows, spectra, shapes = [], [], set()
for fp in files:
    x = load_numpy_pickle(fp)
    shapes.add((x.shape, str(x.dtype)))
    xf = np.asarray(x, dtype=np.float64)
    f, p = welch(xf, fs=FS, nperseg=FS * 2, axis=-1)          # (28, 32, F)
    pm = p.mean(axis=0)                                       # (32, F)
    ch_std = xf.std(axis=-1)                                  # (28, 32)
    line = p[..., (f >= 49) & (f <= 51)].mean(-1)
    side = p[..., ((f >= 44) & (f < 48)) | ((f > 52) & (f <= 56))].mean(-1)
    r = dict(subject=os.path.basename(fp), nan=int(np.isnan(xf).sum()),
             inf=int(np.isinf(xf).sum()), min=float(xf.min()), max=float(xf.max()),
             std=float(xf.std()), flat_ch_clips=int((ch_std < 1e-6).sum()),
             low_std_ch_clips=int((ch_std < 0.5).sum()),
             max_abs_gt_500=int((np.abs(xf).max(-1) > 500).sum()),
             line_ratio=float(np.median(line / side)),
             mean_over_time_abs=float(np.abs(xf.mean(-1)).mean()),
             ch_mean_corr=float(np.mean(np.corrcoef(xf.mean(1).reshape(28, -1))[np.triu_indices(28, 1)])))
    for b, (lo, hi) in bands.items():
        r[b] = float(pm[:, (f >= lo) & (f < hi)].mean())
    rows.append(r); spectra.append(pm.mean(0))
    print(r["subject"], r["std"], flush=True)
spectra = np.array(spectra)
f = welch(np.zeros(500), fs=FS, nperseg=FS * 2)[0]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(f, np.median(spectra, 0), label="median subject")
ax[0].fill_between(f, np.percentile(spectra, 5, 0), np.percentile(spectra, 95, 0), alpha=.3, label="5-95%")
ax[0].set_xlim(0, 125); ax[0].set_xlabel("Hz"); ax[0].set_ylabel("PSD (uV^2/Hz)"); ax[0].legend()
ax[0].set_title("FACED mean PSD across clips/channels")
ax[1].hist([r["std"] for r in rows], bins=30); ax[1].set_xlabel("per-subject overall std (uV)")
ax[1].set_title("Amplitude spread across 123 subjects")
fig.tight_layout(); os.makedirs(os.path.join(ROOT, "reports/figures"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "reports/figures/faced_spectra.png"), dpi=120)

stds = np.array([r["std"] for r in rows]); med = np.median(stds); mad = np.median(np.abs(stds - med))
outl = [r["subject"] for r, s in zip(rows, stds) if abs(s - med) > 5 * 1.4826 * mad]
hf = spectra[:, (f > 60) & (f < 120)].mean(1); lo = spectra[:, (f > 8) & (f < 13)].mean(1)
meta_present = [p for p in glob.glob(os.path.join(ROOT, "datasets/FACED/*")) if not p.endswith(".pkl")]
summary = dict(n_files=len(files), shapes=[list(map(str, s)) for s in shapes],
               nan=sum(r["nan"] for r in rows), inf=sum(r["inf"] for r in rows),
               flat_ch_clips=sum(r["flat_ch_clips"] for r in rows),
               low_std_ch_clips=sum(r["low_std_ch_clips"] for r in rows),
               amp_min=min(r["min"] for r in rows), amp_max=max(r["max"] for r in rows),
               std_median=float(med), std_range=[float(stds.min()), float(stds.max())],
               std_outliers=outl, line_ratio_median=float(np.median([r["line_ratio"] for r in rows])),
               line_ratio_max=float(max(r["line_ratio"] for r in rows)),
               psd_ratio_60_120_over_8_13=float(np.median(hf / lo)),
               psd_at_Hz={str(k): float(np.median(spectra[:, np.argmin(abs(f - k))])) for k in (1, 5, 10, 20, 40, 47, 50, 60)},
               mean_abs_dc=float(np.mean([r["mean_over_time_abs"] for r in rows])),
               metadata_files_found=meta_present)
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
json.dump(dict(summary=summary, per_subject=rows), open(os.path.join(ROOT, "reports/faced_inspection.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
