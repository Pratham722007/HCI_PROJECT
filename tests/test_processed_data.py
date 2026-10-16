"""Gate-2 checks on the built HDF5 files (skipped if not built: run scripts/build_*.py)."""
from pathlib import Path
import numpy as np
import pytest
from hcbm.io import schema
from hcbm.utils.config import load_yaml_config

CFG = load_yaml_config("configs/protocol.yaml")
EXPECT = {"faced": 119 * 28 * 7, "deap": 16 * 40 * 15, "dreamer": 23 * 18 * 15}


@pytest.mark.parametrize("name", list(EXPECT))
def test_dataset_schema_counts_and_channels(name):
    f = Path(f"data/processed/{name}/windows_12ch.h5")
    if not f.is_file():
        pytest.skip("not built")
    s = schema.validate_h5(f, CFG["shared_channels"], 1024)
    assert s["n"] == EXPECT[name]
    assert s["n_reject"] / s["n"] < 0.10


def test_faced_excluded_subjects_absent_and_no_clip_split():
    import h5py
    f = Path("data/processed/faced/windows_12ch.h5")
    if not f.is_file():
        pytest.skip("not built")
    with h5py.File(f, "r") as h:
        subs = set(h["subject_id"][:].astype(str))
        assert not subs & set(CFG["faced"]["excluded_subjects"])
        assert len(subs) == 119
        L = h["labels"]
        assert set(np.unique(L["coarse"][:])) == set(range(7))
        assert np.isfinite(L["valence_raw"][:]).all()
