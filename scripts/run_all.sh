#!/usr/bin/env bash
# End-to-end pipeline from processed windows to every report. Usage:
#   scripts/run_all.sh            # real EEGPT checkpoint (checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt), tag 12ch
#   scripts/run_all.sh --random   # placeholder encoder (random weights), tag 12ch_random
# Env: PY (interpreter, e.g. ".venv-gpu/bin/python -E -P"), N_DEAP, SCALE (EEGPT input multiplier; real runs pick it via select_scale.py unless SCALE_SET=1), BACKEND (qwen|template, default qwen), DEVICE (cpu|cuda), N_EXPLAIN.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-".venv/bin/python -E -P"}
if [[ "${1:-}" == "--random" ]]; then TAG=12ch_random; EXTRA=--random-weights; else TAG=12ch; EXTRA=""; fi
SCALE=${SCALE:-1.0}; BACKEND=${BACKEND:-qwen}; DEVICE=${DEVICE:-cpu}; N_EXPLAIN=${N_EXPLAIN:-60}; N_DEAP=${N_DEAP:-20}
[[ -f data/processed/faced/splits.json ]] || $PY scripts/make_splits.py
if [[ -z "${SCALE_SET:-}" && "$TAG" == "12ch" ]]; then                    # pick EEGPT input scale on FACED train/val only
  [[ -f reports/scale_selection.json ]] || $PY scripts/select_scale.py --device "$DEVICE"
  SCALE=$($PY -c "import json;print(json.load(open('reports/scale_selection.json'))['best'])")
fi
for d in faced deap dreamer; do
  [[ -f data/processed/$d/targets.h5 ]] || $PY scripts/build_targets.py $d
  $PY scripts/extract_embeddings.py $d --scale "$SCALE" --bs 128 --device "$DEVICE" $EXTRA     # resumable / no-op when complete
done
$PY scripts/probe.py data/embeddings/faced_${TAG}.h5                       # Gate 3
$PY scripts/train.py --tag $TAG --search                                   # Phase 5 + 7 (all variants)
$PY scripts/evaluate.py --tag $TAG                                         # in-corpus, Sim 1, ablation
$PY scripts/hitl.py --tag $TAG                                             # Phase 9
$PY scripts/explain.py --tag $TAG --backend $BACKEND --device $DEVICE --n $N_EXPLAIN --n-deap $N_DEAP   # Phase 8
$PY scripts/make_eval_set.py --tag $TAG                                    # Phase 10 materials
$PY scripts/analyse_ratings.py --tag $TAG --synthetic                      # dry run of the rating analysis
echo "done: reports/$TAG/"
