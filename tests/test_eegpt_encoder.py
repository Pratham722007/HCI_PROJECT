import os
import numpy as np
import pytest

REPO = "third_party/EEGPT"
pytestmark = pytest.mark.skipif(not os.path.isdir(REPO), reason="EEGPT repo not cloned")


def _enc():
    from hcbm.models.eegpt_encoder import EEGPTEncoder
    from hcbm.utils.config import load_yaml_config
    ch = load_yaml_config("configs/protocol.yaml")["shared_channels"]
    return EEGPTEncoder(REPO, None, ch, 1024, 1.0, "cpu", load_weights=False), ch


def test_shapes_vocab_determinism():
    import torch
    enc, ch = _enc()
    assert enc.chan_ids.shape == (1, 12)
    x = np.random.default_rng(0).normal(size=(3, 12, 1024)).astype("float32")
    a, b = enc(x), enc(x)
    assert a.shape == (3, 4, 512) and torch.isfinite(a).all()
    assert torch.equal(a, b)
    assert not any(p.requires_grad for p in enc.parameters())
    n = sum(p.numel() for p in enc.parameters())
    assert 20e6 < n < 30e6, n          # 25.3M with the official probe config (README says "10M"; strict ckpt load is the real check)
