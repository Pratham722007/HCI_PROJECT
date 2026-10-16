"""DREAMER loader: last 60 s of each stimulus, DC removed, shared pipeline."""

import numpy as np
import scipy.io as sio

from hcbm.preprocess import pipeline as P

FS = 128


def load(mat_path):
    return sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["DREAMER"]


def process_subject(s, cfg):
    idx = [cfg["dreamer"]["shared_channel_idx"][c] for c in cfg["shared_channels"]]
    pp, fs = cfg["preprocessing"], cfg["fs"]
    sec = cfg["dreamer"]["use_last_seconds"]
    segs = []
    for k in range(18):
        st = np.asarray(s.EEG.stimuli[k], dtype=np.float64)[-sec * FS:, idx].T       # (12, 7680) uV
        segs.append(st - st.mean(axis=-1, keepdims=True))                            # remove DC offset
    x = np.stack(segs)                                                               # (18, 12, 7680)
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
    trial = np.repeat(np.arange(1, n_clip + 1), n_win)
    V, A, D = (np.atleast_1d(getattr(s, n)).astype(float) for n in ("ScoreValence", "ScoreArousal", "ScoreDominance"))
    return dict(X=wins, reject=reject, trial_id=trial, window_idx=np.tile(np.arange(n_win), n_clip),
                labels=dict(peak_uv=peak, valence_raw=V[trial - 1], arousal_raw=A[trial - 1], dominance_raw=D[trial - 1]))
