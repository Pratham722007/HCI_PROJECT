"""Phase 8: explanations + FaithScore -> reports/<tag>/{explanations.jsonl, faithscore.md, faithscore_manual_labels.csv}.

Run: .venv/bin/python -E -P scripts/explain.py --tag 12ch --backend qwen [--n 60 --n-deap 20 --device cpu]
     --backend template = deterministic fallback (plumbing check only; faithful by construction).
Per sampled window (fold-0 models; FACED test stratified by class x CBM-correct, plus a DEAP subset):
  CBM  orig   : explain predicted concepts.
  CBM  flip   : valence v -> 1-v, propagated through L3 and the emotion head, re-explained.
  CBM  placebo: one L1 region bin changed in the TEXT only (L2/L3/emotion fixed) -> polarity should NOT change.
  flat orig/flip: flat output; flip = most probable class whose FACED-train valence lies on the other side of 0.5.
Judges, stored separately: (a) SST-2 sentiment classifier label flip in the expected direction; (b) valence-term check.
FaithScore = faithful changes / perturbations; placebo rate = polarity changes / placebo perturbations.
"""
import argparse, csv, json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.train import data as tdata, runs  # noqa: E402
from hcbm.train.trainer import predict  # noqa: E402
from hcbm.models.confidence import softmax  # noqa: E402
from hcbm.explain import prompts  # noqa: E402
from hcbm.explain.generator import TemplateExplainer, QwenExplainer, polarity_terms  # noqa: E402
from hcbm.eval import metrics as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="12ch"); ap.add_argument("--backend", default="qwen", choices=["qwen", "template"])
ap.add_argument("--n", type=int, default=60); ap.add_argument("--n-deap", type=int, default=20)
ap.add_argument("--device", default="cpu"); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
mcfg = load_yaml_config(os.path.join(ROOT, "configs/model.yaml"))
man = runs.manifest(ROOT, a.tag)
out_dir = os.path.join(ROOT, "reports", a.tag); os.makedirs(out_dir, exist_ok=True)
sp = json.load(open(os.path.join(ROOT, "data/processed/faced/splits.json")))
F = tdata.load(ROOT, "faced", a.tag)
COARSE = F["meta"]["coarse_classes"]; CN = F["concept_names"]
cbm, ck_c = runs.load_fold(runs.fold_paths(ROOT, a.tag, "level123")[0], mcfg)
flat, ck_f = runs.load_fold(runs.fold_paths(ROOT, a.tag, "flat")[0], mcfg)
tr = np.flatnonzero(~F["reject"] & ~np.isin(F["subject"], sp["test"]))
proto_v = np.array([np.nanmean(F["l2"][tr][F["emotion"][tr] == c, 0]) for c in range(len(COARSE))])
rng = np.random.default_rng(a.seed)

# ---- sample windows ----
test = np.flatnonzero(~F["reject"] & np.isin(F["subject"], sp["test"]))
pc = predict(cbm, F["E"][test])["logits"].argmax(1)
strata = {}
for r, ok in zip(test, pc == F["emotion"][test]):
    strata.setdefault((F["emotion"][r], bool(ok)), []).append(r)
keys = sorted(strata)
picked, i = [], 0
while len(picked) < min(a.n, len(test)):
    pool = strata[keys[i % len(keys)]]
    if pool:
        picked.append(pool.pop(rng.integers(len(pool))))
    i += 1
    if not any(strata.values()):
        break
sample = [("faced", F, r) for r in picked]
if a.n_deap and "deap" in man.get("cross_corpus", []):
    DP = tdata.load(ROOT, "deap", a.tag)
    sample += [("deap", DP, r) for r in rng.choice(np.flatnonzero(~DP["reject"]), a.n_deap, replace=False)]

gen = TemplateExplainer() if a.backend == "template" else QwenExplainer(mcfg["explain"]["qwen_model_name"], mcfg["explain"]["max_tokens"], a.device)


def cbm_record(o, i, T):
    p = softmax(o["logits"][i:i + 1], T)[0]
    return dict(l1=o["l1"][i], l2=o["l2"][i], l3=o["l3"][i], emotion=COARSE[p.argmax()], confidence=float(p.max()))


def flat_record(p):
    return dict(emotion=COARSE[p.argmax()], confidence=float(p.max()), class_probs={c: float(x) for c, x in zip(COARSE, p)})


recs = []
t0 = time.time()
for n, (ds, D, r) in enumerate(sample):
    e = D["E"][r:r + 1]
    o = predict(cbm, e); rec = cbm_record(o, 0, ck_c["temperature"])
    v = float(o["l2"][0, 0]); m = np.zeros((1, 3), bool); m[0, 0] = True; val = o["l2"].copy(); val[0, 0] = 1 - v
    rec_flip = cbm_record(predict(cbm, e, {"l2": (val, m)}), 0, ck_c["temperature"])
    summ = prompts.l1_summary(rec["l1"], CN)
    target = "gamma power, parieto-occipital"
    old_bin = next(b for nm, b, _ in summ if nm == target)
    placebo = {target: {"high": "low", "low": "high", "typical": "high"}[old_bin]}
    pf = softmax(predict(flat, e)["logits"], ck_f["temperature"])[0]
    other = (proto_v > 0.5) != (proto_v[pf.argmax()] > 0.5)
    pf_flip = np.where(other, pf, 0); pf_flip = pf_flip / pf_flip.sum()
    facts = dict(cbm=prompts.cbm_facts(rec, CN), cbm_flip=prompts.cbm_facts(rec_flip, CN),
                 cbm_placebo=prompts.cbm_facts(rec, CN, placebo), flat=prompts.flat_facts(flat_record(pf)),
                 flat_flip=prompts.flat_facts(flat_record(pf_flip)))
    texts = {k: gen(f) for k, f in facts.items()}
    y = int(D["emotion"][r])
    recs.append(dict(dataset=ds, row=int(r), subject=str(D["subject"][r]), true_emotion=COARSE[y] if y >= 0 else None,
                     cbm_emotion=rec["emotion"], cbm_conf=rec["confidence"], cbm_valence=v, cbm_flip_emotion=rec_flip["emotion"],
                     flat_emotion=COARSE[pf.argmax()], flat_conf=float(pf.max()), flat_flip_emotion=COARSE[pf_flip.argmax()],
                     flat_valence_side=bool(proto_v[pf.argmax()] > 0.5), placebo=placebo, facts=facts, texts=texts))
    print(f"{n + 1}/{len(sample)} {time.time() - t0:.0f}s", flush=True)

# ---- judges ----
from transformers import pipeline  # noqa: E402
sst = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)
for rc in recs:
    rc["sentiment_pos"] = {k: (s["score"] if s["label"] == "POSITIVE" else 1 - s["score"])
                           for k, s in zip(rc["texts"], sst(list(rc["texts"].values()), truncation=True))}
    rc["terms"] = {k: polarity_terms(t) for k, t in rc["texts"].items()}
    sp_ = rc["sentiment_pos"]; tm = rc["terms"]
    new_pos_cbm = rc["cbm_valence"] < 0.5                                  # after flip, valence is positive
    new_pos_flat = not rc["flat_valence_side"]
    rc["judge"] = dict(
        cbm_a=bool((sp_["cbm_flip"] > 0.5) == new_pos_cbm and (sp_["cbm"] > 0.5) != (sp_["cbm_flip"] > 0.5)),
        cbm_b=bool(tm["cbm_flip"] == (1 if new_pos_cbm else -1)),
        placebo_a=bool((sp_["cbm"] > 0.5) != (sp_["cbm_placebo"] > 0.5)),
        placebo_b=bool(tm["cbm"] != tm["cbm_placebo"]),
        flat_a=bool((sp_["flat_flip"] > 0.5) == new_pos_flat and (sp_["flat"] > 0.5) != (sp_["flat_flip"] > 0.5)),
        flat_b=bool(tm["flat_flip"] == (1 if new_pos_flat else -1)))

with open(os.path.join(out_dir, "explanations.jsonl"), "w") as f:
    for rc in recs:
        f.write(json.dumps(rc) + "\n")

lines = [f"# FaithScore (tag `{a.tag}`, explainer `{a.backend}`)\n", runs.placeholder_banner(man),
         ("> **Template explainer:** deterministic text that restates the concepts, faithful by construction. These numbers "
          "only check the pipeline; run with `--backend qwen` for the real result.\n\n" if a.backend == "template" else ""),
         "Signal (a) = SST-2 sentiment label flips in the expected direction; (b) = explanation uses the new valence term. "
         "Placebo = L1 text-only change, any polarity change is a false positive. CI = subject bootstrap.\n",
         "| dataset | model | perturbation | n | rate (a) [95% CI] | rate (b) [95% CI] |", "|---|---|---|---|---|---|"]
for ds in sorted({r["dataset"] for r in recs}):
    R = [r for r in recs if r["dataset"] == ds]
    subj = np.array([r["subject"] for r in R])
    for model, key, what in (("CBM", "cbm", "valence flip (FaithScore)"), ("CBM", "placebo", "L1 placebo (false-positive rate)"),
                             ("flat", "flat", "class-valence flip (FaithScore)")):
        cells = []
        for sig in ("a", "b"):
            x = np.array([r["judge"][f"{key}_{sig}"] for r in R], float)
            b = M.subject_bootstrap(np.mean, subj, x, n_boot=1000)
            cells.append(f"{b['est']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]")
        lines.append(f"| {ds} | {model} | {what} | {len(R)} | {cells[0]} | {cells[1]} |")
lines += ["\nThe flat baseline has no concepts; its perturbation flips the class summary instead (asymmetry, see limitations).",
          "Judge reliability: fill `human_faithful` (1/0) in `faithscore_manual_labels.csv`, then run "
          "`scripts/faithscore_agreement.py --tag " + a.tag + "`.\n", "## Examples\n"]
for rc in recs[:3]:
    lines += [f"**{rc['dataset']} row {rc['row']}** (true {rc['true_emotion']}, CBM {rc['cbm_emotion']} -> flip {rc['cbm_flip_emotion']})\n",
              f"- original: {rc['texts']['cbm']}", f"- valence flipped: {rc['texts']['cbm_flip']}", f"- placebo: {rc['texts']['cbm_placebo']}", ""]
open(os.path.join(out_dir, "faithscore.md"), "w").write("\n".join(lines))
with open(os.path.join(out_dir, "faithscore_manual_labels.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "new_valence", "before", "after", "judge_a", "judge_b", "human_faithful"])
    for i, rc in enumerate(recs[:20]):
        w.writerow([i, "positive" if rc["cbm_valence"] < 0.5 else "negative", rc["texts"]["cbm"], rc["texts"]["cbm_flip"],
                    int(rc["judge"]["cbm_a"]), int(rc["judge"]["cbm_b"]), ""])
print("wrote", out_dir)
