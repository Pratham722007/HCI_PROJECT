"""Pure preprocessing functions shared by all datasets. Arrays are (..., channels, samples)."""

import numpy as np
from math import gcd
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, filtfilt


def bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def notch(x, fs, f0=50.0, q=30.0):
    b, a = iirnotch(f0, q, fs=fs)
    return filtfilt(b, a, x, axis=-1)


def car(x):
    """Common average reference over the channel axis (-2) of the supplied channels."""
    return x - x.mean(axis=-2, keepdims=True)


def resample(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, int(fs_out) // g, int(fs_in) // g, axis=-1)


def zscore_subject(x, mask=None, eps=1e-8):
    """z-score each channel using mean/std over all (optionally masked) windows of ONE subject.

    x: (n_windows, C, T); mask: (n_windows,) bool, True = use for statistics.
    """
    use = x if mask is None else x[mask]
    if use.shape[0] == 0:
        use = x
    mu = use.mean(axis=(0, 2), keepdims=True)
    sd = use.std(axis=(0, 2), keepdims=True)
    return (x - mu) / (sd + eps)


def cut_windows(x, fs, win_sec, drop_from="start"):
    """Cut (C, T) into non-overlapping windows -> (n, C, win). Remainder dropped from `drop_from`."""
    w = int(round(win_sec * fs))
    n = x.shape[-1] // w
    if n == 0:
        raise ValueError("signal shorter than one window")
    rem = x.shape[-1] - n * w
    x = x[..., rem:] if drop_from == "start" else x[..., : n * w]
    return np.stack(np.split(x, n, axis=-1), axis=0)


def window_peak(windows):
    """Max |x| per window (uV, before z-score); non-finite values count as +inf."""
    w = np.abs(windows)
    w = np.where(np.isfinite(w), w, np.inf)
    return w.max(axis=(1, 2))


def reject_artifacts(windows, rel_k, min_uv):
    """Subject-relative rule (call on ONE subject, before z-scoring).

    Reject (True) a window whose peak exceeds max(rel_k * median peak of the subject, min_uv).
    A fixed absolute threshold is unusable across corpora: typical DEAP window peaks range from
    ~20 uV (s01) to ~860 uV (s11).
    """
    peak = window_peak(windows)
    finite = peak[np.isfinite(peak)]
    thr = max(rel_k * np.median(finite), min_uv) if finite.size else min_uv
    return peak > thr
