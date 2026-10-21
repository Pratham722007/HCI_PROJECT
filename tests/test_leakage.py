"""Gate 2 leakage audit (hard rule 4): cross-corpus labels never reach training, model selection, temperature/threshold
fitting or the EEGPT input-scale choice. Static checks on the scripts that make choices."""
import re
from pathlib import Path


def _src(name, code_only=False):
    s = Path("scripts", name).read_text()
    if code_only:                                                  # drop the module docstring and comments
        s = re.sub(r'^""".*?"""', "", s, count=1, flags=re.S)
        s = re.sub(r"#.*", "", s)
    return s


def test_train_reads_only_embeddings_from_cross_corpus():
    s = _src("train.py")
    assert "fit(" in s and 'X = {ds: tdata.load(ROOT, ds, a.tag) for ds in pcfg["cross_corpus_datasets"]' in s
    # the only place cross-corpus dicts are used is the prediction loop, and only their embeddings are read
    assert set(re.findall(r'\bD\["(\w+)"\]', s)) == {"E"}
    assert re.findall(r"\bX\b(?!\.items|\s*=|\))", s) == []        # X never indexed (only X.items() and sorted(X) names)
    # every fit / temperature / threshold call uses the primary dataset F and its fold rows
    for call in re.findall(r"(fit\(.*?\)|fit_temperature\(.*?\)|escalation_threshold\(.*?\))", s):
        assert "X" not in call and "deap" not in call.lower() and "dreamer" not in call.lower(), call


def test_scale_and_probe_use_primary_train_val_only():
    for name in ("select_scale.py", "probe.py"):
        s = _src(name, code_only=True)
        assert 'cfg["primary_dataset"]' in s, name
        assert "deap" not in s.lower() and "dreamer" not in s.lower(), name
    assert 'sp["folds"]' in _src("select_scale.py")              # subjects drawn from the folds, never the test group
    assert '~np.isin(sub, list(test))' in _src("probe.py")


def test_cross_corpus_is_frozen_in_config():
    from hcbm.utils.config import load_yaml_config
    cfg = load_yaml_config("configs/protocol.yaml")
    assert cfg["primary_dataset"] not in cfg["cross_corpus_datasets"]
