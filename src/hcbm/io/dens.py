"""DENS loader (rev 3): click-locked group-wise epochs of the old DENS release, all 40 subjects.

Each file `Group-<g>/<Word>_<subject>Trial-<t>Click-<k>-slor.mat` holds `data` (128, 1751) float32 uV: a 7 s epoch at
250 Hz around the k-th mouse click of the subject (cumulative), made during trial t (1-based row among the subject's
rated trials in `Behavioural Data/<subject>.csv`, neutral_1 = trial 1). Authors' preprocessing: 1-40 Hz; rows E1..E128.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio

from hcbm.preprocess import pipeline as P

NAME = re.compile(r"^(?P<word>.+)_(?P<subject>mit\w+?)Trial-(?P<trial>\d+)Click-(?P<click>\d+)-slor\.mat$")
RATING_COLS = {"valence": "rating_valence.response", "arousal": "Arousal.response", "dominance": "Dominance.response",
               "liking": "liking.response"}


def parse_name(path):
    m = NAME.match(os.path.basename(path))
    if m is None:
        raise ValueError(f"unexpected DENS epoch name: {path}")
    d = m.groupdict()
    return dict(word=d["word"], subject=d["subject"], trial=int(d["trial"]), click=int(d["click"]),
                group=int(os.path.basename(os.path.dirname(path)).split("-")[1]))


def index_epochs(epochs_root):
    """DataFrame, one row per epoch file: path, word, subject, trial, click, group."""
    rows = [dict(path=p, **parse_name(p)) for p in sorted(glob.glob(os.path.join(epochs_root, "Group-*", "*.mat")))]
    ep = pd.DataFrame(rows).sort_values(["subject", "click", "trial", "group"]).reset_index(drop=True)
    # click numbers are not unique in the authors' filenames (mit035 repeats 5, with different EEG), so each epoch gets a
    # per-subject sequence number used as trial_id; click and stimulus trial are kept as labels
    ep["epoch"] = ep.groupby("subject").cumcount() + 1
    return ep


def load_ratings(beh_root, subject):
    """Rated trials of one subject in presentation order (row 0 = trial 1): stimulus + valence/arousal/dominance/liking."""
    d = pd.read_csv(os.path.join(beh_root, f"{subject}.csv"))
    d = d[d[RATING_COLS["valence"]].notna()].reset_index(drop=True)
    out = pd.DataFrame({k: d[c].astype(float) for k, c in RATING_COLS.items()})
    out["stimulus"] = d["filepath"].astype(str).map(os.path.basename)
    return out


def process_subject(ep, beh_root, cfg, group_to_coarse):
    """ep: index rows of ONE subject. Returns dict(X (n,12,1024) float32, reject, trial_id (=epoch seq.), window_idx, labels)."""
    dc, pp, fs = cfg["dens"], cfg["preprocessing"], cfg["fs"]
    idx = [dc["shared_channel_egi"][c] - 1 for c in cfg["shared_channels"]]             # E-number -> row
    x = np.stack([sio.loadmat(p)["data"] for p in ep["path"]]).astype(np.float64)        # (n_ep, 128, 1751) uV
    assert x.shape[1:] == (128, dc["epoch_samples"]), x.shape
    x = x[:, idx, :]
    x = P.bandpass(x, dc["fs_native"], *cfg["filtering"]["bandpass"], order=pp["filter_order"])   # effective 4-40 Hz
    x = P.car(x)                                                                          # CAR over the 12 channels
    x = P.resample(x, dc["fs_native"], fs)                                                # (n_ep, 12, 1793)
    wins = np.stack([P.cut_windows_strided(e, fs, cfg["window_length_sec"], dc["window_stride_sec"], dc["windows_per_epoch"])
                     for e in x])                                                         # (n_ep, 4, 12, 1024)
    n_ep, n_win = wins.shape[:2]
    wins = wins.reshape(-1, *wins.shape[2:])
    peak = P.window_peak(wins)
    reject = P.reject_artifacts(wins, pp["artifact_rel_k"], pp["artifact_min_uv"])
    wins = P.zscore_subject(wins, mask=~reject).astype(np.float32)
    rat = load_ratings(beh_root, ep["subject"].iloc[0])
    r = rat.iloc[ep["trial"].to_numpy() - 1]                                             # trial t -> row t-1
    rep = lambda a: np.repeat(np.asarray(a), n_win)                                       # noqa: E731
    return dict(X=wins, reject=reject, trial_id=rep(ep["epoch"]), window_idx=np.tile(np.arange(n_win), n_ep),
                labels=dict(peak_uv=peak, group=rep(ep["group"]), coarse=rep([group_to_coarse[g] for g in ep["group"]]),
                            word=rep(ep["word"]), click=rep(ep["click"]), stim_trial=rep(ep["trial"]), stimulus=rep(r["stimulus"]),
                            valence_raw=rep(r["valence"]), arousal_raw=rep(r["arousal"]),
                            dominance_raw=rep(r["dominance"]), liking_raw=rep(r["liking"])))
