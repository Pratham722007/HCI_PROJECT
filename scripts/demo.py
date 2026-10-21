"""Dry run: one EEG window through the trained system, step by step (no training; uses runs/<tag>/ fold-0 models).

Run: .venv-gpu/bin/python -E -P scripts/demo.py [--row N | --seed S] [--backend qwen|template] [--device cuda|cpu]
Default: a random window from the held-out DENS test subjects. Steps: input -> EEGPT embedding (recomputed from the raw
window and compared with the cache) -> Level 1/2/3 concepts vs ground truth -> emotion + confidence -> routing (Qwen or
human reviewer) -> explanation -> counterfactual valence flip (Sim 2) -> expert correction of Level 2 (HITL) -> flat baseline.
"""
import argparse, json, os, sys, textwrap
import h5py, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hcbm.utils.config import load_yaml_config  # noqa: E402
from hcbm.train import data as tdata, runs  # noqa: E402
from hcbm.train.trainer import predict  # noqa: E402
from hcbm.models.confidence import softmax  # noqa: E402
from hcbm.explain import prompts  # noqa: E402
from hcbm.explain.generator import TemplateExplainer, QwenExplainer, polarity_terms  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="dens12"); ap.add_argument("--row", type=int, default=None); ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--backend", default="qwen", choices=["qwen", "template"]); ap.add_argument("--device", default=None)
ap.add_argument("--no-recompute", action="store_true", help="skip recomputing the EEGPT embedding from the raw window")
a = ap.parse_args()
import torch  # noqa: E402
dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
pcfg = load_yaml_config(os.path.join(ROOT, "configs/protocol.yaml")); mcfg = load_yaml_config(os.path.join(ROOT, "configs/model.yaml"))
paths = load_yaml_config(os.path.join(ROOT, "configs/paths.yaml"))
man = runs.manifest(ROOT, a.tag); DS = man["primary"]
sp = json.load(open(os.path.join(ROOT, f"data/processed/{DS}/splits.json")))
D = tdata.load(ROOT, DS, a.tag)
C = D["meta"]["coarse_classes"]; CN = D["concept_names"]
fmt = lambda v: "[" + ", ".join(f"{x:+.2f}" for x in v) + "]"  # noqa: E731
W = lambda s: print(textwrap.fill(s, 110, initial_indent="    ", subsequent_indent="    "))  # noqa: E731


def step(n, title):
    print(f"\n{'=' * 110}\n[{n}] {title}\n{'=' * 110}")


test = np.flatnonzero(~D["reject"] & np.isin(D["subject"], sp["test"]))
r = a.row if a.row is not None else int(np.random.default_rng(a.seed).choice(test))
with h5py.File(os.path.join(ROOT, f"data/processed/{DS}/windows_12ch.h5"), "r") as f:
    x = f["X"][r]; lab = {k: f["labels"][k][r] for k in f["labels"]}; ch = [c.decode() for c in f["ch_names"][:]]
dec = lambda v: v.decode() if isinstance(v, bytes) else v  # noqa: E731

step(1, f"Input: one 4 s EEG window ({DS.upper()} row {r}, {'held-out TEST subject' if r in set(test) else 'train/val subject'})")
print(f"    subject {D['subject'][r]}, epoch {D['trial'][r]}, 12 channels x 1024 samples @ 256 Hz ({', '.join(ch)})")
if "word" in lab:
    print(f"    self-reported emotion word: {dec(lab['word'])}  ->  true class: {C[D['emotion'][r]]}")
print(f"    self-ratings (1-9): valence {lab['valence_raw']:.2f}, arousal {lab['arousal_raw']:.2f}, dominance {lab.get('dominance_raw', np.nan):.2f}")

step(2, "EEGPT encoder (frozen) -> embedding 4 x 512 = 2048 numbers")
e = D["E"][r:r + 1]
if not a.no_recompute:
    from hcbm.models.eegpt_encoder import EEGPTEncoder
    enc = EEGPTEncoder(os.path.join(ROOT, paths["eegpt"]["repo"]), os.path.join(ROOT, paths["eegpt"]["checkpoint"]), ch, 1024, D["scale"], dev)
    z = enc(x[None]).cpu().numpy().reshape(1, -1)
    print(f"    recomputed from the raw window on {dev}: max |difference| vs cached embedding = {np.abs(z - e).max():.2e}")
print(f"    first values: {', '.join(f'{v:+.3f}' for v in e[0, :6])} ...  (input scale {D['scale']}, checkpoint sha256 {D['ckpt_sha256'][:12]}...)")

cbm, ck = runs.load_fold(runs.fold_paths(ROOT, a.tag, "level123")[0], mcfg)
flat, ckf = runs.load_fold(runs.fold_paths(ROOT, a.tag, "flat")[0], mcfg)
o = predict(cbm, e)

step(3, "Level 1, neurophysiological concepts (37 = theta/alpha/gamma x 12 ch + frontal alpha asymmetry), region summary")
pred1, true1 = prompts.l1_summary(o["l1"][0], CN), prompts.l1_summary(D["l1"][r], CN)
print(f"    {'concept':48s} {'predicted':>10s} {'from the EEG (truth)':>22s}")
for (n_, b, z_), (_, bt, zt) in zip(pred1, true1):
    print(f"    {n_:48s} {b:>6s} {z_:+.2f}   {bt:>10s} {zt:+.2f}  {'' if b == bt else '<- differs'}")

step(4, "Level 2, affective state (0-1; > 0.5 = high)")
for i, d in enumerate(("valence", "arousal", "dominance")):
    p_, t_ = o["l2"][0, i], D["l2"][r, i]
    print(f"    {d:10s} predicted {p_:.2f} ({'high' if p_ > 0.5 else 'low '})   self-rating {t_:.2f} ({'high' if t_ > 0.5 else 'low '})  {'' if (p_ > 0.5) == (t_ > 0.5) else '<- wrong side'}")

step(5, "Level 3, cognitive appraisal (-1..1; truth = category pseudo-label)")
for i, d in enumerate(("novelty", "goal conduciveness", "coping potential")):
    print(f"    {d:20s} predicted {o['l3'][0, i]:+.2f}   pseudo-label {D['l3'][r, i]:+.2f}")

step(6, f"Emotion classifier (sees ONLY Level 3) + confidence (temperature T = {ck['temperature']:.2f}, fit on validation)")
p = softmax(o["logits"], ck["temperature"])[0]; p_raw = softmax(o["logits"])[0]
print(f"    {'class':9s} {'raw':>6s} {'calibrated':>11s}")
for i in np.argsort(-p):
    print(f"    {C[i]:9s} {p_raw[i]:6.1%} {p[i]:10.1%}  {'#' * int(round(p[i] * 60))}{'   <- true class' if i == D['emotion'][r] else ''}")
pred = C[p.argmax()]
print(f"    prediction: {pred} ({p.max():.1%})  ->  {'CORRECT' if p.argmax() == D['emotion'][r] else 'WRONG'} (true {C[D['emotion'][r]]})")

step(7, "Routing: confidence estimator")
route = "Qwen explanation (high confidence)" if p.max() >= ck["threshold"] else "ESCALATE to human reviewer (low confidence)"
print(f"    confidence {p.max():.3f} vs threshold {ck['threshold']:.3f} (keeps 80% of validation windows)  ->  {route}")

step(8, f"Explanation ({a.backend}{' on ' + dev if a.backend == 'qwen' else ''}): the LLM sees only these facts")
gen = TemplateExplainer() if a.backend == "template" else QwenExplainer(mcfg["explain"]["qwen_model_name"], mcfg["explain"]["max_tokens"], dev)
rec = dict(l1=o["l1"][0], l2=o["l2"][0], l3=o["l3"][0], emotion=pred, confidence=float(p.max()))
facts = prompts.cbm_facts(rec, CN, dominance_pseudo=DS == "faced")
print(textwrap.indent(facts, "      | "))
text = gen(facts); print("    explanation:"); W(text)

step(9, "Counterfactual (Simulation 2 / FaithScore): flip valence, propagate through L3 + emotion head, re-explain")
v = float(o["l2"][0, 0]); val = o["l2"].copy(); val[0, 0] = 1 - v; m = np.zeros((1, 3), bool); m[0, 0] = True
of = predict(cbm, e, {"l2": (val, m)}); pf_ = softmax(of["logits"], ck["temperature"])[0]
rec_f = dict(l1=of["l1"][0], l2=of["l2"][0], l3=of["l3"][0], emotion=C[pf_.argmax()], confidence=float(pf_.max()))
print(f"    valence {v:.2f} -> {1 - v:.2f};  L3 {fmt(o['l3'][0])} -> {fmt(of['l3'][0])};  emotion {pred} -> {rec_f['emotion']}")
text_f = gen(prompts.cbm_facts(rec_f, CN, dominance_pseudo=DS == "faced")); print("    new explanation:"); W(text_f)
want = 1 if 1 - v > 0.5 else -1
got = polarity_terms(text_f)
print(f"    valence-word judge: expected {'positive' if want > 0 else 'negative'} wording, explanation uses "
      f"{ {1: 'positive', -1: 'negative', 0: 'neither/both'}[got] } -> {'FAITHFUL' if got == want else 'NOT faithful'}")

step(10, "Human-in-the-loop: expert replaces Level 2 with the real self-ratings")
oi = predict(cbm, e, {"l2": (D["l2"][r:r + 1], np.ones((1, 3), bool))}); pi = softmax(oi["logits"], ck["temperature"])[0]
print(f"    L2 {fmt(o['l2'][0])} -> {fmt(D['l2'][r])}  =>  emotion {pred} ({p.max():.0%}) -> "
      f"{C[pi.argmax()]} ({pi.max():.0%})  [{'correct' if pi.argmax() == D['emotion'][r] else 'still wrong'}]")
oi3 = predict(cbm, e, {"l3": (D["l3"][r:r + 1], np.ones((1, 3), bool))}); p3 = softmax(oi3["logits"], ck["temperature"])[0]
print(f"    (correcting L3 with the pseudo-label instead gives {C[p3.argmax()]}: this leaks the class by construction)")

step(11, "Flat baseline (opaque MLP, no concepts) on the same embedding")
pfl = softmax(predict(flat, e)["logits"], ckf["temperature"])[0]
print(f"    prediction: {C[pfl.argmax()]} ({pfl.max():.1%}) -> {'CORRECT' if pfl.argmax() == D['emotion'][r] else 'WRONG'}; "
      f"top-3: {', '.join(f'{C[i]} {pfl[i]:.0%}' for i in np.argsort(-pfl)[:3])}")
print("    it can only be explained by its class probabilities: nothing to inspect, flip or correct.")
print(f"\nDone. Fold-0 models from runs/{a.tag}/; reports with all test windows in reports/{a.tag}/.")
