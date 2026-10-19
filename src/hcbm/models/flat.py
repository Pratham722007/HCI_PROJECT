"""Flat baseline: embedding -> Linear -> GELU -> Dropout -> Linear -> 7 classes (no concepts)."""
import torch
import torch.nn as nn


class FlatBaseline(nn.Module):
    variant = "flat"

    def __init__(self, emb_dim, hidden=128, n_classes=7, dropout=0.1):
        super().__init__()
        self.register_buffer("mu", torch.zeros(emb_dim)); self.register_buffer("sd", torch.ones(emb_dim))
        self.net = nn.Sequential(nn.Linear(emb_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def set_standardizer(self, mu, sd):
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32)); self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, emb, interventions=None):
        return dict(logits=self.net((emb - self.mu) / (self.sd + 1e-6)))
