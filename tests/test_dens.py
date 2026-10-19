"""Rev 3 DENS checks: loader parsing, rating join, windowing, built file, splits, taxonomy (data checks skip if not built)."""
import json
from pathlib import Path

import numpy as np
import pytest

from hcbm.io import dens
from hcbm.preprocess import pipeline as P
from hcbm.utils.config import load_yaml_config

CFG = load_yaml_config("configs/protocol.yaml")
PATHS = load_yaml_config("configs/paths.yaml")
TAX = load_yaml_config("configs/emotion_taxonomy.yaml")
H5 = Path("data/processed/dens/windows_12ch.h5")
BEH = Path(PATHS["raw"]["dens_beh"])


def test_parse_name():
    d = dens.parse_name("/x/Group-06/Angry_mit003Trial-7Click-4-slor.mat")
    assert d == dict(word="Angry", subject="mit003", trial=7, click=4, group=6)
    assert dens.parse_name("/x/Group-07/Taken Aback_mitb2017007Trial-3Click-2-slor.mat")["subject"] == "mitb2017007"
    with pytest.raises(ValueError):
        dens.parse_name("/x/Group-01/readme.mat")


def test_strided_windows():
    x = np.arange(1793)[None].repeat(12, 0)
    w = P.cut_windows_strided(x, 256, 4.0, 1.0, 4)
    assert w.shape == (4, 12, 1024) and w[1, 0, 0] == 256 and w[3, 0, -1] == 768 + 1023
    with pytest.raises(ValueError):
        P.cut_windows_strided(x, 256, 4.0, 1.0, 5)


@pytest.mark.skipif(not BEH.is_dir(), reason="DENS behavioural data not on disk")
def test_rating_join_trial_is_row_in_rated_trials():
    r = dens.load_ratings(BEH, "mit003")
    assert r.iloc[0]["stimulus"] == "neutral_1.mp4"                      # trial 1 = first rated row
    assert r.iloc[1]["stimulus"] == "12.m4v" and r.iloc[1]["valence"] == 9.0   # Joyous_mit003Trial-2Click-1


def test_taxonomy_six_classes():
    C = TAX["coarse_classes"]
    assert C == ["happy", "surprise", "fear", "disgust", "anger", "sadness"]
    assert sorted(TAX["dens"]["groups"]) == list(range(1, 9))
    assert all(g["coarse"] in C for g in TAX["dens"]["groups"].values())
    m = TAX["dreamer"]["fine_to_coarse"]
    assert sum(TAX["dreamer"]["clip_to_fine"][k] in m for k in range(1, 19)) == 16       # calmness clips 1, 11 excluded


@pytest.mark.skipif(not H5.is_file(), reason="DENS not built")
def test_built_file_epochs_intact_and_labels_consistent():
    import h5py
    from hcbm.io import schema
    s = schema.validate_h5(H5, CFG["shared_channels"], 1024)
    assert s["n"] == 1900 and s["n_subjects"] == 40
    with h5py.File(H5, "r") as f:
        sub = f["subject_id"][:].astype(str); tr = f["trial_id"][:]; L = {k: f["labels"][k][:] for k in f["labels"]}
    key = np.char.add(sub, "|" + tr.astype(str))
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    assert len(u) == 475 and (cnt == 4).all()                           # 4 windows per epoch
    for k in ("coarse", "valence_raw", "dominance_raw"):
        v = L[k]
        assert all(len(set(v[inv == i])) == 1 for i in range(len(u))), f"{k} differs within an epoch"
    assert ((L["valence_raw"] >= 1) & (L["valence_raw"] <= 9)).all()


@pytest.mark.skipif(not Path("data/processed/dens/splits.json").is_file(), reason="splits not built")
def test_splits_cover_all_subjects_disjoint():
    import h5py
    sp = json.load(open("data/processed/dens/splits.json"))
    parts = [set(sp["test"])] + [set(f) for f in sp["folds"]]
    assert sum(len(p) for p in parts) == len(set().union(*parts))     # disjoint
    with h5py.File(H5, "r") as f:
        assert set().union(*parts) == set(f["subject_id"][:].astype(str))
    assert len(sp["test"]) == 8
