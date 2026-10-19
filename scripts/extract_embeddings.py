"""Phase 3.2: EEGPT embeddings -> data/embeddings/<dataset>_12ch.h5 (row-aligned with windows_12ch.h5).

Run: .venv/bin/python -E -P scripts/extract_embeddings.py faced --scale 0.01 [--device cpu] [--limit N]
Resumable: re-running continues from the last completed batch. Embedding stored as (N, 4, 512) float32.
--random-weights: same architecture, untrained weights (PLACEHOLDER until the checkpoint is on disk; writes
<dataset>_12ch_random.h5, ckpt_sha256 = "" and every downstream report is marked as a placeholder run).
"""
import argparse, os, sys, time, json
import numpy as np, h5py, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.models.eegpt_encoder import EEGPTEncoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("--scale", type=float, required=True, help="multiplier applied to z-scored windows")
ap.add_argument("--device", default="cpu"); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tag", default=""); ap.add_argument("--random-weights", action="store_true")
a = ap.parse_args()
if a.random_weights and not a.tag:
    a.tag = "_random"
paths = load_yaml_config(os.path.join(ROOT, "configs/paths.yaml"))
src = os.path.join(ROOT, f"data/processed/{a.dataset}/windows_12ch.h5")
out = os.path.join(ROOT, f"data/embeddings/{a.dataset}_12ch{a.tag}.h5"); os.makedirs(os.path.dirname(out), exist_ok=True)
torch.manual_seed(42)
with h5py.File(src, "r") as f:
    ch = [c.decode() for c in f["ch_names"][:]]; N = f["X"].shape[0] if not a.limit else min(a.limit, f["X"].shape[0])
    enc = EEGPTEncoder(os.path.join(ROOT, paths["eegpt"]["repo"]), os.path.join(ROOT, paths["eegpt"]["checkpoint"]), ch, 1024, a.scale, a.device,
                      load_weights=not a.random_weights)
    sha = enc.ckpt_sha256 or ""
    done = 0
    if os.path.exists(out):
        with h5py.File(out, "r") as g:
            same = g.attrs.get("scale") == a.scale and g.attrs.get("ckpt_sha256") == sha and g.attrs.get("n") == N
            done = int(g.attrs["done"]) if same else 0
    mode = "r+" if done else "w"
    with h5py.File(out, mode) as g:
        if not done:
            g.create_dataset("emb", shape=(N, 4, 512), dtype="float32", chunks=(min(256, N), 4, 512))
            g.attrs.update(scale=a.scale, ckpt_sha256=sha, n=N, done=0, src=os.path.relpath(src, ROOT))
            for k in ("subject_id", "trial_id", "window_idx", "reject"):
                g.create_dataset(k, data=f[k][:N])
        t0 = time.time()
        for i in range(done, N, a.bs):
            j = min(N, i + a.bs)
            g["emb"][i:j] = enc(f["X"][i:j]).cpu().numpy(); g.attrs["done"] = j; g.flush()
            el = time.time() - t0
            print(f"{j}/{N}  {el:.0f}s  ETA {el / max(j - done, 1) * (N - j):.0f}s", flush=True)
print("saved", out)
