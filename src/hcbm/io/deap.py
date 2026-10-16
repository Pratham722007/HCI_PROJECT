"""DEAP loader (s01-s16 on disk). Labels come from participant_ratings.xls, NOT the .dat labels (corrupted)."""

import os
import numpy as np
import pandas as pd

from hcbm.io.safe_pickle import load_numpy_pickle
from hcbm.preprocess import pipeline as P

FS = 128
BASELINE = 3 * FS


def load_ratings(xls_path):
    r = pd.read_excel(xls_path)
    return {sid: g.sort_values("Experiment_id")[["Valence", "Arousal", "Dominance", "Liking"]].values for sid, g in r.groupby("Participant_id")}


def process_subject(dat_path, sid, ratings, cfg):
    d = load_numpy_pickle(dat_path)
    idx = [cfg["deap"]["shared_channel_idx"][c] for c in cfg["shared_channels"]]
    x = np.asarray(d["data"], dtype=np.float64)[:, idx, BASELINE:]            # (40, 12, 7680) uV, 60 s
    pp, fs = cfg["preprocessing"], cfg["fs"]
    x = P.notch(x, FS, cfg["filtering"]["notch"], pp["notch_q"])
    x = P.bandpass(x, FS, *cfg["filtering"]["bandpass"], order=pp["filter_order"])
    x = P.car(x)
    x = P.resample(x, FS, fs)
    wins = np.stack([P.cut_windows(c, fs, cfg["window_length_sec"], pp["windows_drop_from"]) for c in x])
    n_clip, n_win = wins.shape[:2]
    wins = wins.reshape(-1, *wins.shape[2:])
    peak = P.window_peak(wins)
    reject = P.reject_artifacts(wins, pp["artifact_rel_k"], pp["artifact_min_uv"])
    wins = P.zscore_subject(wins, mask=~reject).astype(np.float32)
    rat = ratings[sid]                                                          # (40, 4) by Experiment_id == .dat order
    trial = np.repeat(np.arange(1, n_clip + 1), n_win)
    r = rat[trial - 1]
    v, a = r[:, 0], r[:, 1]
    quad = np.where((v == 5) | (a == 5), -1, (v > 5).astype(int) * 2 + (a > 5).astype(int))   # LVLA,LVHA,HVLA,HVHA; -1 = tie
    return dict(X=wins, reject=reject, trial_id=trial, window_idx=np.tile(np.arange(n_win), n_clip),
                labels=dict(peak_uv=peak, valence_raw=v, arousal_raw=a, dominance_raw=r[:, 2], liking_raw=r[:, 3], quadrant=quad))
