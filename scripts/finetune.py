"""Fine-tune EEGPT end to end with the CBM / flat heads on the primary dataset (D26); same folds, selection and outputs
as scripts/train.py, so scripts/evaluate.py works on the new tag.

Run: .venv-gpu/bin/python -E -P scripts/finetune.py --src-tag dens12 --tag dens12_ft [--variants flat,level123] [--device cuda]
Per fold: encoder = EEGPT with its last `blocks` transformer blocks + final norm trainable; head initialised from
runs/<src-tag>/<variant>/fold<k>.pt; early stopping on that fold's validation subjects; temperature + escalation threshold
on the same validation subjects. Test subjects and DEAP/DREAMER are only predicted.
Saves runs/<tag>/<variant>/fold<k>.pt (trainable weights only + metadata), fold<k>_preds.npz, summary.json, manifest.json.
"""
import argparse, copy, json, os, sys, time
import h5py, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.utils.seed import fix_seed  # noqa: E402
from hcbm.train import data as tdata, runs  # noqa: E402
from hcbm.train.finetune import FineTuneModel, fit_ft, predict_ft  # noqa: E402
from hcbm.train.splits import assert_disjoint  # noqa: E402
from hcbm.models.eegpt_encoder import EEGPTEncoder  # noqa: E402
from hcbm.models.confidence import fit_temperature, softmax, escalation_threshold  # noqa: E402
from hcbm.eval.metrics import macro_f1, acc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src-tag", default="dens12"); ap.add_argument("--tag", default="dens12_ft")
ap.add_argument("--variants", default="flat,level123"); ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-epochs", type=int, default=0, help="override (smoke runs)")
a = ap.parse_args()
mcfg = load_yaml_config(os.path.join(ROOT, "configs/model.yaml")); pcfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
paths = load_yaml_config(os.path.join(ROOT, "configs/paths.yaml"))
src = runs.manifest(ROOT, a.src_tag); PRIMARY = src["primary"]
FT = mcfg["finetune"]
sp = json.load(open(os.path.join(ROOT, f"data/processed/{PRIMARY}/splits.json"))); assert_disjoint(sp)
F = tdata.load(ROOT, PRIMARY, a.src_tag, src.get("targets", "targets.h5"))                                   # labels/targets; embeddings unused here
with h5py.File(os.path.join(ROOT, f"data/processed/{PRIMARY}/windows_12ch.h5"), "r") as f:
    X = f["X"][:]; ch = [c.decode() for c in f["ch_names"][:]]
cross = {ds: h5py.File(os.path.join(ROOT, f"data/processed/{ds}/windows_12ch.h5"), "r")["X"] for ds in src.get("cross_corpus", [])}
keep = ~F["reject"]; test_rows = np.flatnonzero(keep & np.isin(F["subject"], sp["test"])); labels = list(range(mcfg["training"]["n_classes"]))
base = EEGPTEncoder(os.path.join(ROOT, paths["eegpt"]["repo"]), os.path.join(ROOT, paths["eegpt"]["checkpoint"]), ch, 1024, F["scale"], "cpu")
assert base.ckpt_sha256 == src["ckpt_sha256"], "checkpoint differs from the source run"
log = lambda s: print(s, flush=True)  # noqa: E731

for variant in a.variants.split(","):
    vdir = runs.run_dir(ROOT, a.tag, variant); os.makedirs(vdir, exist_ok=True); summ = []
    for k, fp in enumerate(runs.fold_paths(ROOT, a.src_tag, variant)):
        t0 = time.time(); fix_seed(a.seed)
        va = np.flatnonzero(keep & np.isin(F["subject"], sp["folds"][k]))
        tr = np.flatnonzero(keep & ~np.isin(F["subject"], sp["test"] + sp["folds"][k]))
        head, ck = runs.load_fold(fp, mcfg)
        hp = dict(ck["hp"], **{k_: v for k_, v in FT.items() if k_ != "blocks"})
        if a.max_epochs:
            hp.update(ft_max_epochs=a.max_epochs, ft_min_epochs=1)
        model = FineTuneModel(copy.deepcopy(base), head, FT["blocks"])
        model, hist = fit_ft(model, X, F, tr, va, hp, a.device, a.seed)
        pv = predict_ft(model, X[va], a.device)
        T = fit_temperature(pv["logits"], F["emotion"][va], t_max=mcfg["training"].get("max_temperature", 10.0))
        thr = escalation_threshold(softmax(pv["logits"], T).max(1), mcfg["training"]["escalation_coverage"])
        out = {}
        for ds, Xs, rows in [("val", X, va), ("test", X, test_rows)] + [(n, Xc, None) for n, Xc in cross.items()]:
            p = predict_ft(model, Xs[rows] if rows is not None else Xs, a.device)
            out.update({f"{ds}__{key}": v for key, v in p.items()}); out[f"{ds}__rows"] = rows if rows is not None else np.arange(len(Xs))
        np.savez_compressed(os.path.join(vdir, f"fold{k}_preds.npz"), **out)
        torch.save(dict(trainable_state=model.trainable_state(), variant=variant, hp=hp, emb_dim=ck["emb_dim"], blocks=FT["blocks"],
                        src_fold=os.path.relpath(fp, ROOT), temperature=T, threshold=thr, history=hist, finetune=True),
                   os.path.join(vdir, f"fold{k}.pt"))
        yt, pt = F["emotion"][test_rows], out["test__logits"].argmax(1)
        s = dict(fold=k, epochs=len(hist), T=T, threshold=thr, val_acc=acc(F["emotion"][va], pv["logits"].argmax(1)),
                 val_macro_f1=macro_f1(F["emotion"][va], pv["logits"].argmax(1), labels), test_acc=acc(yt, pt), test_macro_f1=macro_f1(yt, pt, labels),
                 frozen_val_macro_f1=json.load(open(os.path.join(runs.run_dir(ROOT, a.src_tag, variant), "summary.json")))["folds"][k]["val_macro_f1"])
        summ.append(s)
        log(f"{variant} fold {k}: val F1 {s['val_macro_f1']:.3f} (frozen {s['frozen_val_macro_f1']:.3f}) test F1 {s['test_macro_f1']:.3f} "
            f"({len(hist)} ep, {time.time() - t0:.0f}s)")
    n_tr = sum(p.numel() for m in model.trainable for p in m.parameters()) + sum(p.numel() for p in model.head.parameters())
    json.dump(dict(variant=variant, hp=hp, folds=summ, n_params=int(n_tr)), open(os.path.join(vdir, "summary.json"), "w"), indent=1)

json.dump(dict(tag=a.tag, primary=PRIMARY, targets=src.get("targets", "targets.h5"), emb_tag=a.src_tag, seed=a.seed, git_commit=runs.git_commit(ROOT), ckpt_sha256=src["ckpt_sha256"],
               input_scale=F["scale"], cross_corpus=sorted(cross), finetune=dict(FT, src_tag=a.src_tag), hp_search=False,
               appraisal_status=src.get("appraisal_status")),
          open(os.path.join(runs.run_dir(ROOT, a.tag), "manifest.json"), "w"), indent=1)
print("done", runs.run_dir(ROOT, a.tag))
