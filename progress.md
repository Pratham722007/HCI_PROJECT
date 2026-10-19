# H-CBM-EEG: Progress Log

Last updated: 20 Oct 2026. Plan: [research-plan.md](research-plan.md) (rev 3). Decisions: [docs/DEVIATIONS.md](docs/DEVIATIONS.md).

## Status at a glance

| Stage | Status |
|---|---|
| Repo skeleton, configs, environment | Done (16 Oct) |
| Dataset inspection and audits (FACED, DEAP, DREAMER, DENS) | Done (16-20 Oct) |
| Preprocessing + unified HDF5 windows (DEAP, DREAMER, FACED, DENS) | Done |
| EEGPT encoder (real checkpoint, strict load, GPU) | Done (20 Oct) |
| Concept targets L1/L2/L3 | Done for DENS, DEAP, DREAMER (L3 = draft appraisal table) |
| Models (H-CBM, 5 variants + flat baseline), training, evaluation, HITL, explanations, human-eval tooling | Code done and tested; **not yet run on real DENS embeddings** |
| **Rev 3 Day 1** (DENS in, pipeline made dataset-agnostic, embeddings) | **Done (20 Oct)** |
| Rev 3 Day 2 (training, Sim 1-3, HITL, Qwen + FaithScore) | Next |
| Rev 3 Day 3 (human-eval materials, results summary, deck edits) | Pending |
| Human evaluation with real raters | Not started (raters/ethics not arranged) |

No research results exist yet. Everything run so far is either data QC, a Gate-3 probe, or a placeholder/smoke run that
only checks the plumbing.

---

## Timeline

| Date | Commit | What |
|---|---|---|
| 16 Oct | `dfd3e99` | Repository skeleton: `src/hcbm/` packages, configs, Makefile, smoke tests |
| 16 Oct | `c1b2423` | Phase 1: dataset inspection; FACED chosen as primary (rev 2); protocol decisions locked |
| 16 Oct | `817d162` | Phase 2: preprocessing pipeline, FACED/DEAP/DREAMER loaders, cross-dataset QC |
| 19 Oct | `51200ce` | All 32 DEAP subjects, EEGPT encoder wrapper, splits, embedding script |
| 19 Oct | — | Full pipeline built (concepts, models, training, evaluation, HITL, explanations, human eval) and run end to end on a random-weight encoder (plumbing check, tag `12ch_random`) |
| 20 Oct | `bf40e80` | Phase 3-4 code committed |
| 20 Oct | — | Real EEGPT checkpoint downloaded and verified; GPU environment (`.venv-gpu`) working |
| 20 Oct | — | Plan rev 3: **DENS trains, DEAP/DREAMER test, FACED deferred**, 3-day implementation target |
| 20 Oct | uncommitted | Rev 3 Day 1 implemented (below) |

---

## Datasets

| Dataset | Role (rev 3) | Processed file | Subjects | Windows | Labels |
|---|---|---|---|---|---|
| DENS | Train / val / in-corpus test | `data/processed/dens/windows_12ch.h5` | 40 | 1,900 (475 epochs x 4) | 6 classes, valence/arousal/dominance 1-9 |
| DEAP | Zero-shot test | `data/processed/deap/windows_12ch.h5` | 32 | 19,200 | V/A/D 1-9, quadrants |
| DREAMER | Zero-shot test | `data/processed/dreamer/windows_12ch.h5` | 23 | 6,210 | V/A/D 1-5, clip emotion (16/18 clips mapped) |
| FACED | Deferred | `data/processed/faced/windows_12ch.h5` | 119 | 23,324 | 9 categories, V/A |

All windows: 12 shared channels (F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8), 4 s, 256 Hz, band-passed, common average
reference over the 12 channels, per-subject z-score, artifact flags kept as a `reject` column.

### Audit findings that shaped the protocol
- **DEAP:** the local `.dat` copy differs from the official release on 439/1280 trials (every high-valence/high-arousal
  trial has V/A set to 9-x). EEG is unaltered. Labels are taken from `participant_ratings.xls`, which is genuine. Re-download
  DEAP from the official source before publication (D2).
- **DENS:** the official OpenNeuro release (ds003751 v1.0.6) has truncated EEG for 30/40 subjects. The older release's
  475 click-locked 7 s epochs cover all 40 subjects and carry a self-reported emotion; every epoch matches its trial's
  ratings. These are the training data (D17). Click numbers repeat for subject mit035 (5 pairs, different EEG), so epochs
  are numbered per subject instead (D20). The authors already cleaned the epochs: 0/1,900 windows hit the artifact rule (D22).
- **DREAMER:** labels verified against an independent copy; clip-to-emotion table transcribed from the paper (D12).
- **FACED:** 27/119 subjects' valence ratings don't follow the clips (r < 0.5 vs other raters); a masking rule was fixed
  before training (D11). Only relevant if FACED returns.

---

## Rev 3 Day 1 (20 Oct): what was done

| Task | Result |
|---|---|
| 1.1 Configs | `protocol.yaml`: `primary_dataset: dens`, DENS section (EGI channel map, epoch/window parameters), gamma band 30-40 Hz, DENS thresholds (5 on 1-9). `emotion_taxonomy.yaml`: 6 classes (happy, surprise, fear, disgust, anger, sadness), DENS group map, DREAMER map. `appraisal_profiles.yaml`: `dens` section (draft). `model.yaml`: 6 classes, dominance weight 1.0. `paths.yaml`: DENS epoch and behavioural folders |
| 1.2 DENS loader | `src/hcbm/io/dens.py`, `scripts/build_dens.py`, `cut_windows_strided` in `preprocess/pipeline.py`. Output: 1,900 windows x 12 ch x 1024 samples, 40 subjects. Class windows: happy 520, sadness 424, disgust 260, surprise 248, anger 248, fear 200 |
| 1.3 Splits | `data/processed/dens/splits.json`: 8 test subjects (384 windows) + 5 folds of 6-7 subjects; every class present in every partition |
| 1.4 Dataset-agnostic scripts | `train.py`, `evaluate.py`, `hitl.py`, `explain.py`, `probe.py`, `select_scale.py`, `make_splits.py`, `build_targets.py`, `qc_cross_dataset.py`, `extract_embeddings.py`, `run_all.sh` read the datasets from `protocol.yaml`. Prediction keys are now `val`/`test`. Explanations sample one window per DENS epoch (D23) |
| 1.5 Concept targets | `data/processed/{dens,deap,dreamer}/targets.h5`. DENS: real V/A/D, L3 from class. DREAMER: L3 pseudo-labels for its 16 mapped clips. DEAP: no class, no L3 |
| 1.6 QC | [reports/qc.md](reports/qc.md) regenerated with DENS. Dataset-of-origin classifier: 0.68 accuracy vs 0.33 chance, i.e. a large shift between datasets (limitation) |
| 1.7 Tests | `tests/test_dens.py` (parsing, rating join, windowing, epochs intact, splits, taxonomy). **38/38 tests pass** |
| 1.8 EEGPT | Input scale chosen on DENS train/val subjects only: 0.1 (6-class macro-F1 by candidate scale: 0.01 → 0.182, 0.1 → 0.184, 1 → 0.172, 10 → 0.158). Embeddings for all three datasets on the GPU. Smoke run (2 epochs, template explainer) of every downstream stage passed; smoke outputs deleted |

### Gate 1 probe (logistic regression on frozen DENS embeddings, 5 subject-wise folds)

| Target | Accuracy | Macro-F1 | Reference |
|---|---|---|---|
| 6-class emotion | 0.222 | 0.176 | chance F1 0.167, majority acc 0.282 |
| Valence (> 5) | 0.624 | 0.531 | chance F1 0.50, majority acc 0.682 |
| Arousal (> 5) | 0.697 | 0.432 | chance F1 0.50, majority acc 0.812 |

The embedding carries little emotion signal on 12 channels. This is reported, not hidden; the 3-day goal is a complete
pipeline, not strong results. Source: [reports/probe_dens_dens12.json](reports/probe_dens_dens12.json).

---

## Pipeline components (code)

| Component | Files |
|---|---|
| Preprocessing, HDF5 schema | `src/hcbm/preprocess/pipeline.py`, `src/hcbm/io/schema.py` |
| Loaders | `src/hcbm/io/{dens,deap,dreamer,faced}.py`; builders `scripts/build_{dens,cross_corpus,faced}.py` |
| EEGPT encoder | `src/hcbm/models/eegpt_encoder.py` (frozen, strict load, output 4 x 512 = 2048 per window) |
| Concepts | `src/hcbm/concepts/level1.py` (Welch theta/alpha/gamma x 12 ch + frontal alpha asymmetry = 37), `level2.py` (V/A/D rescaled 0-1), `level3.py` (appraisal pseudo-labels) |
| Models | `src/hcbm/models/hcbm.py` (L1 → L2 → L3 → emotion; variants level1_only, level12, level123 (main), level123_l2, bypass; concept interventions), `flat.py` (2-layer MLP + GELU baseline), `confidence.py` (temperature scaling, escalation threshold) |
| Training | `src/hcbm/train/{trainer,data,runs,splits}.py`, `scripts/train.py` (5 subject-wise folds, early stopping on val macro-F1, small lr x dropout grid) |
| Evaluation | `scripts/evaluate.py` → in-corpus, Sim 1 (zero-shot DEAP/DREAMER), Sim 3 (ablation); `src/hcbm/eval/metrics.py` (subject-bootstrap CIs, paired bootstrap, ECE, risk-coverage) |
| Human-in-the-loop | `src/hcbm/hitl/intervention.py`, `scripts/hitl.py` (intervention curves, escalation) |
| Explanations / Sim 2 | `src/hcbm/explain/{prompts,generator}.py` (Qwen2.5-1.5B-Instruct or template fallback), `scripts/explain.py` (FaithScore with valence flip + placebo, two judges), `scripts/faithscore_agreement.py` |
| Human evaluation | `scripts/make_eval_set.py` (blinded pairs, offline rater form, private key gitignored), `scripts/analyse_ratings.py` (Wilcoxon, FDR, Krippendorff's alpha, Fleiss' kappa), `docs/EVAL_PROTOCOL.md` |
| One command | `scripts/run_all.sh` |

---

## Where things are

| What | Location |
|---|---|
| Raw data (read-only) | `datasets/` (DENS epochs: `datasets/dens_old/Physiological Data/PreProcessed-Emotion (Group-wise)/EEG_Data/`) |
| Processed windows, targets, splits | `data/processed/<dataset>/{windows_12ch.h5, targets.h5, splits.json}` |
| Embeddings | `data/embeddings/<dataset>_dens12.h5`: `emb` (N, 4, 512), row-aligned ids, attrs `scale` = 0.1 and `ckpt_sha256` |
| EEGPT checkpoint | `checkpoints/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt` (sha256 `9d63ebc8…`) |
| Qwen + sentiment judge | Hugging Face cache (`~/.cache/huggingface`) |
| Trained models, predictions (Day 2) | `runs/dens12/` |
| Reports | `reports/` (Day 2 results go to `reports/dens12/`); `reports/12ch_random/` = superseded FACED placeholder run |
| Environments | `.venv` (CPU, tests), `.venv-gpu` (CUDA torch + transformers, no pytest) |

## How to run

Day 1 from scratch (from the project root):
```bash
.venv/bin/python -E -P scripts/build_dens.py
.venv/bin/python -E -P scripts/make_splits.py
.venv/bin/python -E -P scripts/build_targets.py
.venv/bin/python -E -P scripts/qc_cross_dataset.py
PY=".venv-gpu/bin/python -E -P" DEVICE=cuda STOP_AFTER=embed scripts/run_all.sh
.venv/bin/pytest -q
```
Day 2 (everything after embeddings):
```bash
PY=".venv-gpu/bin/python -E -P" DEVICE=cuda scripts/run_all.sh
```

---

## Open items

1. **Confirm the 6-class mapping** (DENS group 3 alarmed/startled/tense → surprise; group 7 incl. "taken aback" → sadness).
2. **Approve the Level-3 appraisal profiles** (`configs/appraisal_profiles.yaml`, `dens` section; currently a draft).
3. **Commit Day 1** (all rev 3 changes are uncommitted).
4. **Raters:** at least 3 for the human evaluation, and whether ethics approval is needed.
5. **EGI → 10-10 mapping:** T7/T8 are about 7° off; confirm against EGI's equivalence chart.
6. **DEAP:** re-download from the official source before publication.
7. Later: FACED as a second training corpus; DENS raw clips for the 10 complete subjects.

## Known limitations so far
- DENS is small (475 epochs) and its labels are click-localised self-reports; DEAP/DREAMER labels are whole-trial ratings.
- Large shift between datasets (QC domain classifier 0.68 vs 0.33 chance).
- Weak emotion signal in 12-channel EEGPT embeddings (Gate 1 probe).
- Level 3 is a category pseudo-label, so correcting it in the intervention experiment leaks the class.
- Qwen 1.5B was seen ignoring the supplied facts in a test (wrote "positive" for a negative valence); FaithScore measures this.
