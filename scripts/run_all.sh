#!/usr/bin/env bash
# End-to-end pipeline from processed windows to every report. Datasets come from configs/protocol.yaml
# (primary_dataset trains; cross_corpus_datasets are zero-shot tests). Usage:
#   scripts/run_all.sh            # real EEGPT checkpoint (configs/paths.yaml eegpt.checkpoint), tag <primary>12 (rev 3: dens12)
#   scripts/run_all.sh --random   # placeholder encoder (random weights), tag <primary>12_random
# Env: PY (interpreter, e.g. ".venv-gpu/bin/python -E -P"), DEVICE (cpu|cuda), SCALE (EEGPT input multiplier; real runs
# pick it via select_scale.py unless SCALE_SET=1), BACKEND (qwen|template, default qwen), N_EXPLAIN, N_DEAP, STOP_AFTER
# (embed|train: stop after that stage).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-".venv/bin/python -E -P"}
read -r PRIMARY CROSS < <($PY -c "import yaml;c=yaml.safe_load(open('configs/protocol.yaml'));print(c['primary_dataset'],' '.join(c['cross_corpus_datasets']))")
if [[ "${1:-}" == "--random" ]]; then TAG=${PRIMARY}12_random; EXTRA=--random-weights; else TAG=${PRIMARY}12; EXTRA=""; fi
SCALE=${SCALE:-1.0}; BACKEND=${BACKEND:-qwen}; DEVICE=${DEVICE:-cpu}; N_EXPLAIN=${N_EXPLAIN:-120}; N_DEAP=${N_DEAP:-30}
[[ -f data/processed/$PRIMARY/splits.json ]] || $PY scripts/make_splits.py "$PRIMARY"
if [[ -z "${SCALE_SET:-}" && -z "$EXTRA" ]]; then                          # EEGPT input scale, chosen on primary train/val only
  sel_ds=$($PY -c "import json;print(json.load(open('reports/scale_selection.json')).get('dataset',''))" 2>/dev/null || true)
  [[ "$sel_ds" == "$PRIMARY" ]] || $PY scripts/select_scale.py --device "$DEVICE"
  SCALE=$($PY -c "import json;print(json.load(open('reports/scale_selection.json'))['best'])")
fi
for d in $PRIMARY $CROSS; do
  [[ -f data/processed/$d/targets.h5 ]] || $PY scripts/build_targets.py "$d"
  $PY scripts/extract_embeddings.py "$d" --scale "$SCALE" --tag "${TAG%_random}" --bs 128 --device "$DEVICE" $EXTRA   # resumable
done
$PY scripts/probe.py "data/embeddings/${PRIMARY}_${TAG}.h5"                # Gate 3
[[ "${STOP_AFTER:-}" == "embed" ]] && { echo "stopped after embeddings"; exit 0; }
$PY scripts/train.py --tag "$TAG" --search                                 # Phase 5 + 7 (all variants)
$PY scripts/evaluate.py --tag "$TAG"                                       # in-corpus, Sim 1, ablation
$PY scripts/hitl.py --tag "$TAG"                                           # Phase 9
[[ "${STOP_AFTER:-}" == "train" ]] && { echo "stopped after training/evaluation"; exit 0; }
$PY scripts/explain.py --tag "$TAG" --backend "$BACKEND" --device "$DEVICE" --n "$N_EXPLAIN" --n-deap "$N_DEAP"   # Phase 8
$PY scripts/make_eval_set.py --tag "$TAG"                                  # Phase 10 materials
$PY scripts/analyse_ratings.py --tag "$TAG" --synthetic                    # dry run of the rating analysis
echo "done: reports/$TAG/"
