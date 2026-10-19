import numpy as np

from hcbm.concepts import level1, level2, level3
from hcbm.eval import metrics as M
from hcbm.utils.config import load_yaml_config

CFG = load_yaml_config("configs/protocol.yaml")
CH = CFG["shared_channels"]


def test_level1_alpha_sine_and_faa():
    fs, t = 256, np.arange(1024) / 256
    X = np.random.default_rng(0).normal(0, 0.1, (2, 12, 1024))
    X[0] += np.sin(2 * np.pi * 10 * t)                                  # 10 Hz on every channel of window 0
    X[1, CH.index("F4")] += 2 * np.sin(2 * np.pi * 10 * t)              # right-frontal alpha in window 1
    lp = level1.band_log_power(X, fs, CFG["band_edges"])
    assert lp.shape == (2, 36)
    assert (lp[0, 12:24] > lp[0, :12]).all() and (lp[0, 12:24] > lp[0, 24:]).all()      # alpha dominates
    f = level1.faa(lp, CH)
    assert f[1] > 1.0 and abs(f[0]) < 0.5
    assert len(level1.concept_names(CH)) == 37


def test_tertiles_per_subject():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 2)); subs = np.repeat(["a", "b", "c"], 100)
    B = level1.tertile_bins(Z, subs)
    for s in "abc":
        assert np.allclose(np.bincount(B[subs == s, 0], minlength=3), [34, 33, 33], atol=1)


def test_level2_rescale_midpoints():
    assert level2.rescale(3.5, "faced") == 0.5 and level2.rescale(5, "deap") == 0.5 and level2.rescale(3, "dreamer") == 0.5


def test_appraisal_table_complete():
    prof = load_yaml_config("configs/appraisal_profiles.yaml")
    tax = load_yaml_config("configs/emotion_taxonomy.yaml")
    A, dom = level3.profile_table(prof, tax["faced"]["fine_classes"])
    assert A.shape == (9, 3) and (np.abs(A) <= 1).all() and ((dom >= 0) & (dom <= 1)).all()
    out = level3.build_level3(np.array([0, -1]), A)
    assert np.isnan(out[1]).all() and np.allclose(out[0], A[0])


def test_rater_reliability_flags_random_rater():
    rng = np.random.default_rng(0)
    norm = rng.uniform(0, 7, 28)
    subs, trials, val = [], [], []
    for s in range(10):
        r = rng.uniform(0, 7, 28) if s == 9 else norm + rng.normal(0, 0.5, 28)
        subs += [f"s{s}"] * 28; trials += list(range(28)); val += list(r)
    r = level2.rater_reliability(np.array(val), np.array(subs), np.array(trials), [f"s{s}" for s in range(10)])
    assert r["s9"] < 0.5 < min(v for k, v in r.items() if k != "s9")


def test_metrics_bootstrap_and_risk_coverage():
    y = np.array([0, 1] * 50); p = y.copy(); subj = np.repeat(np.arange(10), 10)
    assert M.subject_bootstrap(M.acc, subj, y, p, n_boot=50)["est"] == 1.0
    cov, risk, aurc = M.risk_coverage(np.array([0.9, 0.8, 0.2]), np.array([1, 1, 0]))
    assert np.allclose(risk, [0, 0, 1 / 3]) and np.isclose(aurc, 1 / 9)
    probs = np.array([[0.9, 0.1], [0.9, 0.1]]); assert np.isclose(M.ece(probs, np.array([0, 1])), 0.4)


def test_agreement_statistics():
    from hcbm.eval.agreement import krippendorff_interval, fleiss_kappa, bh
    perfect = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], float)
    assert np.isclose(krippendorff_interval(perfect), 1.0)
    assert np.isclose(fleiss_kappa(perfect), 1.0)
    assert np.allclose(bh(np.array([0.01, 0.04, 0.03])), [0.03, 0.04, 0.04])
