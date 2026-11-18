#!/usr/bin/env bash
# Rev 3c (D28) experiment: full montage + accuracy improvements, FACED -> DENS/DEAP/DREAMER.
#   run A  faced30    full-montage (30 ch) EEGPT embeddings + per-subject embedding normalisation (D29)
#   run B  faced30sx  A + spectral features (30-ch layout) + 12-ch-view augmentation + uncertainty loss balancing
#                     + smoothing over 3 neighbouring windows within each clip (D29)
# Needs windows_full.h5 (scripts/build_full.py). Env: PY (GPU interpreter), DEVICE.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-".venv-gpu/bin/python -E -P"}; DEVICE=${DEVICE:-cuda}; CPU=".venv/bin/python -E -P"
$CPU scripts/build_spectral.py faced dens deap dreamer
[[ -f reports/scale_selection_faced_full.json ]] || $PY scripts/select_scale.py --montage full --pca 256 --scales 0.001,0.01,0.1 --device "$DEVICE"
SCALE=$($PY -c "import json;print(json.load(open('reports/scale_selection_faced_full.json'))['best'])")
for d in faced dens deap dreamer; do
  $PY scripts/extract_embeddings.py "$d" --scale "$SCALE" --tag faced30 --montage full --bs 256 --device "$DEVICE"
  ln -sf "${d}_faced30.h5" "data/embeddings/${d}_faced30sx.h5"            # run B reuses run A's embeddings
done
$PY scripts/extract_embeddings.py faced --scale "$SCALE" --tag faced30v12 --montage 12ch --bs 256 --device "$DEVICE"   # 12-ch view
( $PY scripts/train.py --tag faced30 --search --device "$DEVICE" --subj-norm > logs/run_faced30.log 2>&1 && \
  $CPU scripts/evaluate.py --tag faced30 >> logs/run_faced30.log 2>&1 && $CPU scripts/hitl.py --tag faced30 >> logs/run_faced30.log 2>&1 ) &
( $PY scripts/train.py --tag faced30sx --search --device "$DEVICE" --extra spectral_full.h5:feats \
      --aug-tag faced30v12 --aug-extra spectral_full.h5:feats12 --loss-balance uncertainty --subj-norm --smooth 3 > logs/run_faced30sx.log 2>&1 && \
  $CPU scripts/evaluate.py --tag faced30sx >> logs/run_faced30sx.log 2>&1 && $CPU scripts/hitl.py --tag faced30sx >> logs/run_faced30sx.log 2>&1 ) &
wait
echo "done: reports/faced30 reports/faced30sx"
