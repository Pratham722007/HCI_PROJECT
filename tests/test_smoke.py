"""Smoke tests for H-CBM-EEG repository skeleton and package importability."""

from pathlib import Path
import pytest


def test_import_hcbm():
    """Verify that the hcbm package and all required submodules can be imported."""
    import hcbm
    import hcbm.io
    import hcbm.preprocess
    import hcbm.concepts
    import hcbm.models
    import hcbm.train
    import hcbm.eval
    import hcbm.explain
    import hcbm.hitl
    import hcbm.utils

    assert hcbm.__version__ == "0.1.0"


def test_fix_seed():
    """Verify that fix_seed executes and returns the expected seed."""
    from hcbm.utils.seed import fix_seed
    seed = fix_seed(42)
    assert seed == 42


def test_protocol_yaml():
    """Verify that configs/protocol.yaml contains all required keys and values."""
    from hcbm.utils.config import load_yaml_config

    protocol_path = Path("configs/protocol.yaml")
    assert protocol_path.is_file(), f"protocol.yaml not found at {protocol_path.resolve()}"

    cfg = load_yaml_config(protocol_path)

    # Check windowing and fs
    assert cfg.get("fs") == 256
    assert cfg.get("window_length_sec") == 4.0
    assert cfg.get("window_overlap_sec") == 0.0

    # Check 14 shared channels
    expected_channels = [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
    ]
    assert cfg.get("shared_channels") == expected_channels
    assert len(cfg.get("shared_channels")) == 14

    # Check frequency bands
    bands = cfg.get("band_edges", {})
    assert bands.get("theta") == [4.0, 8.0]
    assert bands.get("alpha") == [8.0, 13.0]
    assert bands.get("gamma") == [30.0, 45.0]

    # Check rating thresholds exist
    thresholds = cfg.get("rating_thresholds", {})
    assert "dens" in thresholds
    assert "deap" in thresholds
    assert "dreamer" in thresholds

    # Check seeds
    assert cfg.get("split_seeds", {}).get("default") == 42


def test_paths_yaml():
    """Verify that configs/paths.yaml exists and defines expected sections."""
    from hcbm.utils.config import load_yaml_config

    paths_path = Path("configs/paths.yaml")
    assert paths_path.is_file(), f"paths.yaml not found at {paths_path.resolve()}"

    cfg = load_yaml_config(paths_path)
    assert "raw" in cfg
    assert "dens" in cfg["raw"]
    assert "processed" in cfg
    assert "embeddings" in cfg


def test_model_yaml():
    """Verify that configs/model.yaml exists and defines expected sections."""
    from hcbm.utils.config import load_yaml_config

    model_path = Path("configs/model.yaml")
    assert model_path.is_file(), f"model.yaml not found at {model_path.resolve()}"

    cfg = load_yaml_config(model_path)
    assert "encoder" in cfg
    assert "hierarchical_cbm" in cfg
    assert "baseline_flat_mlp" in cfg
    assert "explain" in cfg
