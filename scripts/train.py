"""Phase 5/7: train flat + CBM variants on the primary dataset (protocol `primary_dataset`, rev 3 = DENS; subject-wise
5 folds), then predict its val/test subjects and the frozen cross-corpus sets (DEAP, DREAMER).

Run: .venv/bin/python -E -P scripts/train.py --tag dens12 [--search] [--variants flat,level123]
  --tag picks data/embeddings/<dataset>_<tag>.h5 (dens12 = real EEGPT; *_random = placeholder encoder).
Per fold: early stopping on the fold's validation subjects; temperature + escalation threshold fit on the same
validation subjects; the test group and DEAP/DREAMER are only predicted, never used for any choice.
Outputs runs/<tag>/<variant>/fold<k>.pt, fold<k>_preds.npz (keys val__*, test__*, <cross>__*), summary.json;
runs/<tag>/manifest.json.
"""
import argparse, itertools, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.utils.seed import fix_seed  # noqa: E402
from hcbm.train import data as tdata, runs  # noqa: E402
from hcbm.train.trainer import fit, predict  # noqa: E402
from hcbm.train.splits import assert_disjoint  # noqa: E402
from hcbm.models.confidence import fit_temperature, softmax, escalation_threshold  # noqa: E402
from hcbm.eval.metrics import macro_f1, acc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="12ch"); ap.add_argument("--variants", default=",".join(runs.ALL_VARIANTS))
ap.add_argument("--search", action="store_true"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-epochs", type=int, default=0, help="override (smoke runs)")
a = ap.parse_args()
mcfg = load_yaml_config(os.path.join(ROOT, "configs/model.yaml"))
pcfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml"))
PRIMARY = pcfg["primary_dataset"]
HP = dict(mcfg["training"])
if a.max_epochs:
    HP["max_epochs"] = a.max_epochs
sp = json.load(open(os.path.join(ROOT, f"data/processed/{PRIMARY}/splits.json"))); assert_disjoint(sp)
F = tdata.load(ROOT, PRIMARY, a.tag)
X = {ds: tdata.load(ROOT, ds, a.tag) for ds in pcfg["cross_corpus_datasets"] if os.path.exists(tdata.emb_path(ROOT, ds, a.tag))}
emb_dim = F["E"].shape[1]
keep = ~F["reject"]
test_rows = np.flatnonzero(keep & np.isin(F["subject"], sp["test"]))
labels7 = list(range(HP["n_classes"]))
os.makedirs(runs.run_dir(ROOT, a.tag), exist_ok=True)
log = lambda s: print(s, flush=True)  # noqa: E731


def fold_rows(k):
    va = np.flatnonzero(keep & np.isin(F["subject"], sp["folds"][k]))
    tr = np.flatnonzero(keep & ~np.isin(F["subject"], sp["test"] + sp["folds"][k]))
    assert not set(F["subject"][tr]) & set(F["subject"][va]) and not set(F["subject"][tr]) & set(sp["test"])
    return tr, va


def cv_score(variant, hp):
    f1s = []
    for k in range(len(sp["folds"])):
        tr, va = fold_rows(k); fix_seed(a.seed)
        m, _ = fit(runs.build_model(variant, emb_dim, mcfg, hp), F, tr, va, hp, a.seed)
        f1s.append(macro_f1(F["emotion"][va], predict(m, F["E"][va])["logits"].argmax(1), labels7))
    return float(np.mean(f1s))


hp_path = os.path.join(runs.run_dir(ROOT, a.tag), "hp_search.json")
if a.search:
    res = {}
    for fam, variant in (("flat", "flat"), ("cbm", "level123")):
        grid = []
        for lr, dr in itertools.product(HP["search"]["lr"], HP["search"]["dropout"]):
            s = cv_score(variant, dict(HP, lr=lr, dropout=dr)); grid.append(dict(lr=lr, dropout=dr, val_macro_f1=s))
            log(f"search {fam} lr={lr} dropout={dr}: val macro-F1 {s:.4f}")
        res[fam] = dict(best=max(grid, key=lambda g: g["val_macro_f1"]), grid=grid)
    json.dump(res, open(hp_path, "w"), indent=1)
search = json.load(open(hp_path)) if os.path.exists(hp_path) else {}

for variant in a.variants.split(","):
    fam = "flat" if variant == "flat" else "cbm"
    hp = dict(HP, **{k: search[fam]["best"][k] for k in ("lr", "dropout")}) if fam in search else dict(HP)
    vdir = runs.run_dir(ROOT, a.tag, variant); os.makedirs(vdir, exist_ok=True)
    summ = []
    for k in range(len(sp["folds"])):
        t0 = time.time(); tr, va = fold_rows(k); fix_seed(a.seed)
        model, hist = fit(runs.build_model(variant, emb_dim, mcfg, hp), F, tr, va, hp, a.seed)
        pv = predict(model, F["E"][va])
        T = fit_temperature(pv["logits"], F["emotion"][va], t_max=HP.get("max_temperature", 10.0))
        thr = escalation_threshold(softmax(pv["logits"], T).max(1), HP["escalation_coverage"])
        runs.save_fold(os.path.join(vdir, f"fold{k}.pt"), model, hp, emb_dim, T, thr, hist)
        out = {}
        for ds, rows, D in [("val", va, F), ("test", test_rows, F)] + [(n, np.arange(len(D["E"])), D) for n, D in X.items()]:
            p = predict(model, D["E"][rows])
            out.update({f"{ds}__{key}": v for key, v in p.items()}); out[f"{ds}__rows"] = rows
        np.savez_compressed(os.path.join(vdir, f"fold{k}_preds.npz"), **out)
        pt = out["test__logits"].argmax(1); yt = F["emotion"][test_rows]
        s = dict(fold=k, epochs=len(hist), T=T, threshold=thr,
                 val_acc=acc(F["emotion"][va], pv["logits"].argmax(1)), val_macro_f1=macro_f1(F["emotion"][va], pv["logits"].argmax(1), labels7),
                 test_acc=acc(yt, pt), test_macro_f1=macro_f1(yt, pt, labels7))
        summ.append(s)
        log(f"{variant} fold {k}: val F1 {s['val_macro_f1']:.3f} test F1 {s['test_macro_f1']:.3f} ({len(hist)} ep, {time.time() - t0:.0f}s)")
    json.dump(dict(variant=variant, hp=hp, folds=summ,
                   n_params=int(sum(p.numel() for p in model.parameters()))), open(os.path.join(vdir, "summary.json"), "w"), indent=1)

json.dump(dict(tag=a.tag, primary=PRIMARY, seed=a.seed, git_commit=runs.git_commit(ROOT), ckpt_sha256=F["ckpt_sha256"], input_scale=F["scale"],
               appraisal_status=F["meta"].get("appraisal_status"), unreliable_raters=F["meta"].get("unreliable_raters"),
               cross_corpus=sorted(X), hp_search=bool(search)),
          open(os.path.join(runs.run_dir(ROOT, a.tag), "manifest.json"), "w"), indent=1)
