import numpy as np
import pytest
import torch

from hcbm.models.hcbm import HierarchicalCBM, VARIANTS
from hcbm.models.flat import FlatBaseline
from hcbm.models.confidence import fit_temperature, softmax
from hcbm.hitl.intervention import intervention_mask


@pytest.mark.parametrize("variant", VARIANTS)
def test_shapes(variant):
    m = HierarchicalCBM(64, variant=variant)
    o = m(torch.randn(5, 64))
    assert o["l1"].shape == (5, 37) and o["l2"].shape == (5, 3) and o["l3"].shape == (5, 3) and o["logits"].shape == (5, 7)
    assert (o["l2"] >= 0).all() and (o["l2"] <= 1).all() and (o["l3"].abs() <= 1).all()


def test_gradient_path_only_through_bottleneck():
    """Main model: once L3 is fully replaced, the emotion logits no longer depend on the embedding."""
    torch.manual_seed(0)
    m = HierarchicalCBM(64, variant="level123").eval()
    x = torch.randn(4, 64, requires_grad=True)
    m(x)["logits"].sum().backward()
    assert x.grad.abs().sum() > 0                                    # path exists through the concepts
    x.grad = None
    iv = {"l3": (torch.zeros(4, 3), torch.ones(4, 3, dtype=torch.bool))}
    g = torch.autograd.grad(m(x, iv)["logits"].sum(), x, allow_unused=True)[0]
    assert g is None or g.abs().sum() == 0
    # bypass ablation does have a second path
    b = HierarchicalCBM(64, variant="bypass").eval()
    g = torch.autograd.grad(b(x, iv)["logits"].sum(), x, allow_unused=True)[0]
    assert g is not None and g.abs().sum() > 0


def test_emotion_head_input_is_l3_only():
    m = HierarchicalCBM(64, variant="level123")
    assert m.emo_head[0].in_features == 3


def test_flat_overfits_tiny_batch():
    torch.manual_seed(0)
    m = FlatBaseline(16, hidden=32, dropout=0.0)
    x, y = torch.randn(14, 16), torch.arange(14) % 7
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(300):
        loss = torch.nn.functional.cross_entropy(m(x)["logits"], y); opt.zero_grad(); loss.backward(); opt.step()
    assert (m(x)["logits"].argmax(1) == y).all()


def test_temperature_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(size=(2000, 3)); logits[np.arange(2000), y] += 1.0
    T = fit_temperature(logits * 5, y)
    assert T > 2
    assert np.allclose(softmax(logits, 1).sum(1), 1)


def test_intervention_mask_fraction_and_eligibility():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(10, 37)); ok = np.ones((10, 37), bool); ok[:, 0] = False
    for frac in (0.0, 0.5, 1.0):
        m = intervention_mask("l1", pred, ok, frac, "uncertainty", rng)
        assert (m.sum(1) == round(frac * 36)).all() and not m[:, 0].any()
    m = intervention_mask("l2", np.array([[0.51, 0.9, 0.1]]), np.ones((1, 3), bool), 1 / 3, "uncertainty", rng)
    assert m.tolist() == [[True, False, False]]                       # closest to the 0.5 boundary first
