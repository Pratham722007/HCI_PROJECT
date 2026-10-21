"""Temperature scaling and escalation threshold, both fit on FACED validation subjects only (hard rule 4)."""
import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits, y, max_iter=200, t_max=10.0):
    """Single temperature T > 0 minimising validation NLL, clipped to [1/t_max, t_max] (D25: when a model is at or below
    chance on validation the NLL optimum is T -> infinity, which flattens every prediction to uniform and breaks escalation)."""
    z = torch.as_tensor(logits, dtype=torch.float64); t = torch.as_tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(min(max(log_t.detach().exp().item(), 1.0 / t_max), t_max))


def softmax(logits, T=1.0):
    z = np.asarray(logits, float) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def escalation_threshold(conf_val, coverage=0.8):
    """Confidence threshold that retains `coverage` of validation windows; below it -> escalate to a human."""
    return float(np.quantile(conf_val, 1.0 - coverage))
