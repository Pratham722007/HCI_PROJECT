"""FACED loader: 12-channel shared montage, windows, ratings and labels."""

import glob
import os
import numpy as np
import scipy.io as sio

from hcbm.io.safe_pickle import load_numpy_pickle
from hcbm.preprocess import pipeline as P

RATING_ITEMS = ["joy", "tenderness", "inspiration", "amusement", "anger", "disgust", "fear", "sadness",
                "arousal", "valence", "familiarity", "liking"]
FINE = ["anger", "disgust", "fear", "sadness", "neutral", "amusement", "inspiration", "joy", "tenderness"]
CLIP_FINE = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]  # Stimuli_info.xlsx
COARSE = ["happy", "tenderness", "anger", "fear", "disgust", "sadness", "neutral"]
FINE_TO_COARSE = {0: 2, 1: 4, 2: 3, 3: 5, 4: 6, 5: 0, 6: 0, 7: 0, 8: 1}


def load_ratings(rating_root, sub):
    """(28, 12) ratings ordered by video index (matches pickle clip order)."""
    r = sio.loadmat(os.path.join(rating_root, sub, "After_remarks.mat"), squeeze_me=True, struct_as_record=False)["After_remark"]
    score = np.array([x.score for x in r], dtype=float)
    vid = np.array([x.vid for x in r]).astype(int)
    assert sorted(vid.tolist()) == list(range(1, 29)), f"{sub}: vid not a permutation"
    return score[np.argsort(vid)]


def process_subject(pkl_path, rating_root, cfg):
    """Return dict with windows (n,12,1024) float32, reject, trial_id, window_idx, labels."""
    sub = os.path.splitext(os.path.basename(pkl_path))[0]
    x = np.asarray(load_numpy_pickle(pkl_path), dtype=np.float64)           # (28, 32, 7500) uV
    assert x.shape == (28, 32, 7500), x.shape
    pp, fs = cfg["preprocessing"], cfg["fs"]
    idx = [cfg["faced"]["shared_channel_idx"][c] for c in cfg["shared_channels"]]
    x = x[:, idx, :]                                                         # (28, 12, 7500)
    x = P.notch(x, cfg["faced"]["fs_native"], cfg["filtering"]["notch"], pp["notch_q"])
    x = P.bandpass(x, cfg["faced"]["fs_native"], *cfg["filtering"]["bandpass"], order=pp["filter_order"])
    x = P.car(x)                                                             # CAR over the 12 channels
    x = P.resample(x, cfg["faced"]["fs_native"], fs)                         # (28, 12, 7680)
    wins = np.stack([P.cut_windows(c, fs, cfg["window_length_sec"], pp["windows_drop_from"]) for c in x])  # (28, 7, 12, 1024)
    n_clip, n_win = wins.shape[:2]
    wins = wins.reshape(-1, *wins.shape[2:])
    peak = P.window_peak(wins)
    reject = P.reject_artifacts(wins, pp["artifact_rel_k"], pp["artifact_min_uv"])
    wins = P.zscore_subject(wins, mask=~reject).astype(np.float32)
    rat = load_ratings(rating_root, sub)
    trial = np.repeat(np.arange(1, n_clip + 1), n_win)
    fine = np.array(CLIP_FINE)[trial - 1]
    return dict(X=wins, reject=reject, trial_id=trial, window_idx=np.tile(np.arange(n_win), n_clip),
                labels=dict(peak_uv=peak, fine=fine, coarse=np.array([FINE_TO_COARSE[f] for f in fine]),
                            ratings=rat[trial - 1], valence_raw=rat[trial - 1, 9], arousal_raw=rat[trial - 1, 8]))


def iter_subject_files(raw_root, excluded):
    for p in sorted(glob.glob(os.path.join(raw_root, "sub*.pkl"))):
        if os.path.splitext(os.path.basename(p))[0] not in excluded:
            yield p
