#!/usr/bin/env bash
# Presentation run: shows that the pipeline works, using the saved best run (faced30sx). No training (~1 minute).
#   scripts/show_results.sh
cd "$(dirname "$0")/.."
T=faced30sx; R=reports/$T
hdr(){ printf "\n\033[1;35m%s\033[0m\n%s\n" "$1" "------------------------------------------------------------------------"; }

hdr "H-CBM-EEG: pipeline check and results (run $T: FACED trains; DENS, DEAP, DREAMER unseen)"
.venv/bin/pytest -q 2>&1 | tail -1

hdr "1. DATA: four datasets, same preprocessing (4-s windows, 256 Hz)"
.venv/bin/python -E -P - <<'EOF'
import h5py
for d in ["faced", "dens", "deap", "dreamer"]:
    with h5py.File(f"data/processed/{d}/windows_12ch.h5") as f:
        n, s = f["X"].shape[0], len(set(f["subject_id"][:]))
    full = "30 ch" if d != "dreamer" else "12 ch"
    role = "TRAIN / TEST (held-out subjects)" if d == "faced" else "zero-shot TEST (never seen in training)"
    print(f"  {d.upper():8s} {s:4d} subjects  {n:6d} windows  {full:6s}  {role}")
EOF

hdr "2. PIPELINE: what produced this run"
.venv/bin/python -E -P - <<EOF
import json
m = json.load(open("runs/$T/manifest.json"))
print(f"  Encoder        : EEGPT (frozen), checkpoint sha256 {m['ckpt_sha256'][:12]}..., input scale {m['input_scale']}")
print(f"  Extra input    : {m.get('extra')} (spectral features), per-subject normalisation={m.get('subj_norm')}, smoothing={m.get('smooth')} windows")
print(f"  Training       : 5 subject-wise folds, hyperparameter search={m.get('hp_search')}, loss balancing={m.get('loss_balance')}")
print(f"  Test sets      : FACED held-out subjects + {', '.join(m['cross_corpus']).upper()} (zero-shot)")
EOF

hdr "3. MAIN RESULT: FACED, 24 unseen subjects, 7 emotions"
.venv/bin/python -E -P - "$R" <<'EOF2'
import re, sys
R = sys.argv[1]; t = open(f"{R}/in_corpus.md").read()
def rows(header_start):
    blk = t[t.index(header_start):].split("\n\n")[0].splitlines()[2:]
    return {r.split("|")[1].strip(): [c.strip() for c in r.split("|")[2:-1]] for r in blk}
w, c = rows("| model | test acc"), rows("| model | window balanced")
name = {"flat": "Flat baseline (black box)", "level123": "Main CBM (interpretable)"}
print(f"  {'':27s} {'accuracy':>9s} {'macro-F1':>9s} {'balanced':>9s} {'per-clip acc':>13s}")
for m in ("level123", "flat"):
    print(f"  {name[m]:27s} {w[m][0].split()[0]:>9s} {w[m][1].split()[0]:>9s} {c[m][0]:>9s} {c[m][1].split()[0]:>13s}")
print(f"  {'Random guessing':27s} {'0.139':>9s} {'0.131':>9s} {'0.143':>9s}")
print(f"  {'Always guess happy':27s} {'0.322':>9s} {'0.070':>9s} {'0.143':>9s}")
print("\n  95% CI (CBM): accuracy", w["level123"][0].split(" ", 1)[1], " macro-F1", w["level123"][1].split(" ", 1)[1])
print("  Per-clip = average of the 7 window predictions of a 30-s clip (one prediction per clip).")
print("\n  Ablation (test macro-F1):", ", ".join(f"{k} {w[k][1].split()[0]}" for k in ("flat", "level1_only", "level12", "level123")))
EOF2

hdr "4. CONCEPT LEVELS (main CBM, FACED test)"
.venv/bin/python -E -P - "$R" <<'EOF2'
import sys
t = open(f"{sys.argv[1]}/in_corpus.md").read()
r = [l for l in t.splitlines() if l.startswith("| level123 |") and "0." in l][-1].split("|")[2:-1]
print(f"  Level 1  brain-signal concepts : correlation with measured band power r = {r[0].strip()}")
print(f"  Level 2  valence / arousal     : high-vs-low accuracy {r[1].strip()} / {r[2].strip()}")
print(f"  Level 3  appraisal             : mean abs. error {r[5].strip()} (scale -1..1)")
EOF2

hdr "5. ZERO-SHOT on datasets never seen in training (main CBM vs flat)"
.venv/bin/python -E -P - "$R" <<'EOF2'
import sys
t = open(f"{sys.argv[1]}/sim1.md").read().split("\n## ")[1:]
print(f"  {'dataset':9s} {'model':10s} {'valence acc':>12s} {'arousal acc':>12s}")
for sec in t:
    ds = sec.split()[0]
    for l in sec.splitlines():
        p = [c.strip() for c in l.split("|")]
        if len(p) > 6 and ((p[1] == "level123" and p[2] == "L2 concept") or (p[1] == "flat" and p[2] == "via emotion")):
            print(f"  {ds:9s} {('CBM' if p[1] == 'level123' else 'flat'):10s} {p[3].split()[0]:>12s} {p[5].split()[0]:>12s}")
print("  (chance about 0.50; the datasets use different equipment and stimuli, so transfer is hard)")
EOF2

hdr "6. HUMAN-IN-THE-LOOP: are low-confidence cases really the wrong ones?"
.venv/bin/python -E -P - "$R" <<'EOF2'
import sys
t = open(f"{sys.argv[1]}/hitl.md").read()
for l in t.splitlines():
    if l.startswith("| level123 | FACED test") or l.startswith("| level123 | DREAMER"):
        p = [c.strip() for c in l.split("|")]
        print(f"  {p[2]:22s} error kept {p[4].split()[0]}  vs  escalated to human {p[5].split()[0]}   (calibration error {p[8].split()[0]} -> {p[9].split()[0]})")
print("  -> escalated cases have a higher error rate: the confidence estimator flags likely mistakes")
EOF2

hdr "7. LIVE DEMO: one unseen EEG window through the whole system"
.venv-gpu/bin/python -E -P scripts/demo.py --tag $T --seed 7 --no-recompute 2>&1 | grep -v -i -E "warn|it/s\]|Loading weights|generation flags|^\s*$"

hdr "Done. Full reports: $R/  (figures in $R/figures/)"
