"""Frozen EEGPT target-encoder wrapper (official BINE022/EEGPT code, cloned to third_party/EEGPT).

Input : (B, 12, 1024) float, 256 Hz, 4 s, channels in the order of `ch_names` (all must be in EEGPT's vocabulary).
Output: (B, 4, 512) = mean over the 16 time patches of EEGPT's 4 summary tokens (flatten -> 2048-d).
"""
import hashlib
import importlib.util
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn

ENC_CFG = dict(patch_size=64, embed_num=4, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, drop_rate=0.0,
               attn_drop_rate=0.0, drop_path_rate=0.0, init_std=0.02, qkv_bias=True)


def _load_model_module(repo_dir):
    path = os.path.join(repo_dir, "downstream/Modules/models/EEGPT_mcae.py")  # same module the official linear-probe scripts use
    spec = importlib.util.spec_from_file_location("eegpt_mcae", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


class EEGPTEncoder(nn.Module):
    def __init__(self, repo_dir, ckpt_path, ch_names, n_samples=1024, input_scale=1.0, device="cpu", load_weights=True):
        super().__init__()
        mod = _load_model_module(repo_dir)
        self.enc = mod.EEGTransformer(img_size=[len(ch_names), n_samples], norm_layer=partial(nn.LayerNorm, eps=1e-6), **ENC_CFG)
        self.chan_ids = self.enc.prepare_chan_ids(ch_names)         # asserts every name is in EEGPT's vocabulary
        self.input_scale = float(input_scale)
        self.ckpt_sha256 = None
        if load_weights:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = {k[len("target_encoder."):]: v for k, v in ck["state_dict"].items() if k.startswith("target_encoder.")}
            self.enc.load_state_dict(sd, strict=True)               # strict: any mismatch is a hard error
            self.ckpt_sha256 = sha256(ckpt_path)
        self.eval().requires_grad_(False).to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device) * self.input_scale
        z = self.enc(x, self.chan_ids.to(self.device))              # (B, N=16, 4, 512)
        return z.mean(dim=1)                                        # (B, 4, 512)
