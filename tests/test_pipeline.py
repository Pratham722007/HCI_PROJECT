import numpy as np
from hcbm.preprocess import pipeline as P


def _sine(f, fs, sec=30):
    t = np.arange(int(fs * sec)) / fs
    return np.sin(2 * np.pi * f * t)[None, :]


def test_bandpass_keeps_10hz_removes_50hz():
    fs = 250
    keep = P.bandpass(_sine(10, fs), fs, 4, 45)
    drop = P.bandpass(P.notch(_sine(50, fs), fs), fs, 4, 45)   # pipeline order: notch then band-pass
    assert keep[0, 500:-500].std() > 0.65
    assert drop[0, 500:-500].std() < 0.05


def test_notch_removes_50hz_keeps_10hz():
    fs = 250
    assert P.notch(_sine(50, fs), fs).std() < 0.05
    assert P.notch(_sine(10, fs), fs)[0, 500:-500].std() > 0.65


def test_car_zero_mean_across_channels():
    x = np.random.default_rng(0).normal(size=(3, 12, 100))
    assert np.allclose(P.car(x).mean(axis=-2), 0)


def test_resample_and_windows_shape():
    x = np.random.default_rng(0).normal(size=(12, 7500))
    y = P.resample(x, 250, 256)
    assert y.shape == (12, 7680)
    w = P.cut_windows(y, 256, 4.0, "start")
    assert w.shape == (7, 12, 1024)
    assert np.array_equal(w[0, :, 0], y[:, 512])  # first 2 s dropped


def test_reject_and_zscore():
    w = np.random.default_rng(0).normal(size=(5, 12, 1024)) * 10
    w[2, 0, 5] = 1e4
    rej = P.reject_artifacts(w, 5.0, 100.0)
    assert rej.tolist() == [False, False, True, False, False]
    z = P.zscore_subject(w, mask=~rej)
    assert abs(z[~rej].mean()) < 1e-6 and abs(z[~rej].std() - 1) < 1e-2
