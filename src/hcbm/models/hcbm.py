"""Hierarchical concept bottleneck: embedding -> L1 (37) -> L2 (3) -> L3 (3) -> emotion (7).

Main model (`level123`): each stage sees ONLY the previous stage's concepts; no residual path from the embedding.
Ablation variants change only what the emotion head reads:
  level1_only  emotion <- L1            level12   emotion <- L2           level123  emotion <- L3 (main)
  level123_l2  emotion <- [L2, L3] (labelled variant, plan 5.3)           bypass    emotion <- [L3, embedding trunk] (extra)
All concept heads are trained in every variant so concepts can always be explained/intervened on.
`interventions={"l1"|"l2"|"l3": (values, mask)}` replaces predicted concepts (mask True) before they feed later stages.
"""
import torch
import torch.nn as nn

VARIANTS = ("level1_only", "level12", "level123", "level123_l2", "bypass")


def mlp(d_in, hidden, d_out, dropout=0.0):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class HierarchicalCBM(nn.Module):
    def __init__(self, emb_dim, n_l1=37, n_l2=3, n_l3=3, n_classes=7, l1_hidden=(256, 128), l2_hidden=(64, 32),
                 l3_hidden=(32, 16), emo_hidden=(32,), dropout=0.1, variant="level123"):
        super().__init__()
        assert variant in VARIANTS, variant
        self.variant = variant
        self.register_buffer("mu", torch.zeros(emb_dim)); self.register_buffer("sd", torch.ones(emb_dim))
        self.l1_trunk = mlp(emb_dim, l1_hidden[:-1], l1_hidden[-1], dropout)
        self.l1_out = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(l1_hidden[-1], n_l1))   # z-scored power: linear
        self.l2_head = mlp(n_l1, l2_hidden, n_l2, dropout)                                              # sigmoid -> 0-1
        self.l3_head = mlp(n_l2, l3_hidden, n_l3, dropout)                                              # tanh -> [-1, 1]
        d_emo = {"level1_only": n_l1, "level12": n_l2, "level123": n_l3, "level123_l2": n_l2 + n_l3,
                 "bypass": n_l3 + l1_hidden[-1]}[variant]
        self.emo_head = mlp(d_emo, emo_hidden, n_classes, dropout)

    def set_standardizer(self, mu, sd):
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32)); self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32))

    @staticmethod
    def _apply(c, key, interventions):
        if interventions and key in interventions:
            v, m = interventions[key]
            c = torch.where(torch.as_tensor(m, dtype=torch.bool, device=c.device), torch.as_tensor(v, dtype=c.dtype, device=c.device), c)
        return c

    def forward(self, emb, interventions=None):
        h = self.l1_trunk((emb - self.mu) / (self.sd + 1e-6))
        l1 = self._apply(self.l1_out(h), "l1", interventions)
        l2 = self._apply(torch.sigmoid(self.l2_head(l1)), "l2", interventions)
        l3 = self._apply(torch.tanh(self.l3_head(l2)), "l3", interventions)
        x = {"level1_only": l1, "level12": l2, "level123": l3, "level123_l2": torch.cat([l2, l3], 1),
             "bypass": torch.cat([l3, h], 1)}[self.variant]
        return dict(l1=l1, l2=l2, l3=l3, logits=self.emo_head(x))
