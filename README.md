# H-CBM-EEG

Hierarchical Concept Bottleneck Models for EEG Emotion Recognition.

A research framework mapping EEGPT embeddings -> Neurophysiological concepts (L1) -> Affective dimensions (L2) -> Cognitive appraisal dimensions (L3) -> Emotion classification, with Qwen natural-language explanations.

## Protocol Highlights
- **Primary Dataset**: FACED (123 subjects, 28 clips, 9 emotion categories; 4 contaminated subjects excluded) for all training, cross-validation, and hyperparameter tuning. DENS is an optional secondary only.
- **Cross-Corpus Zero-Shot Testing**: DEAP and DREAMER evaluated frozen with zero fine-tuning.
- **Baseline**: Flat 2-layer MLP + GELU.
- **Shared Montage**: 12 channels (F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8) at 256 Hz.

## Running the pipeline
```bash
.venv/bin/pip install -e .            # once
.venv/bin/pytest -q                    # tests
scripts/run_all.sh                     # real EEGPT checkpoint at checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt -> reports/12ch/
make placeholder                       # random-weight encoder + template explainer (plumbing check) -> reports/12ch_random/
```
`run_all.sh` env vars: `SCALE` (EEGPT input multiplier), `BACKEND=qwen|template`, `DEVICE=cpu|cuda`, `N_EXPLAIN`.
For the Qwen stage on the GPU use `.venv-gpu` (CUDA torch):
`.venv-gpu/bin/python -E -P scripts/explain.py --tag 12ch --backend qwen --device cuda --n 120 --n-deap 30`.

| stage | script | output |
|---|---|---|
| concept targets (L1 band power/FAA, L2 ratings, L3 appraisal pseudo-labels) | `build_targets.py` | `data/processed/<ds>/targets.h5` |
| EEGPT embeddings / Gate-3 probe | `extract_embeddings.py`, `probe.py` | `data/embeddings/`, `reports/probe_*.json` |
| flat + CBM variants, 5 subject-wise folds | `train.py --search` | `runs/<tag>/` |
| in-corpus, Sim 1 cross-corpus, ablation | `evaluate.py` | `reports/<tag>/{in_corpus,sim1,sim_ablation}.md` |
| interventions + escalation | `hitl.py` | `reports/<tag>/hitl.md` |
| Qwen explanations + FaithScore | `explain.py`, `faithscore_agreement.py` | `reports/<tag>/faithscore.md` |
| human evaluation | `make_eval_set.py`, `analyse_ratings.py` | `reports/<tag>/human_eval/`, `human_eval.md` |

Protocol decisions and deviations: `docs/DEVIATIONS.md`. Rater instructions: `docs/EVAL_PROTOCOL.md`.
