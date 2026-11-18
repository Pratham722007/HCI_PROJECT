# H-CBM-EEG: Progress Log

Last updated: 24 Oct 2026. Plan: [research-plan.md](research-plan.md) (rev 3; the training corpus has since changed, see
rev 3b/3c below). Decisions: [docs/DEVIATIONS.md](docs/DEVIATIONS.md) (D1-D29).

## Status at a glance

| Stage | Status |
|---|---|
| Repo skeleton, configs, environment | Done (16 Oct) |
| Dataset inspection and audits (FACED, DEAP, DREAMER, DENS) | Done (16-20 Oct) |
| Preprocessing + unified HDF5 windows, 12 channels (all four datasets) | Done |
| Full 30-channel windows (FACED, DEAP, DENS) | Done (22 Oct) |
| EEGPT encoder (real checkpoint, strict load, GPU) | Done (20 Oct) |
| Concept targets L1/L2/L3 | Done for all four datasets (L3 = draft appraisal table, not yet approved) |
| Models (H-CBM, 5 variants + flat baseline), training, evaluation, HITL, explanations, human-eval tooling | Done; run on DENS (21 Oct) and FACED (22-23 Oct) |
| Rev 3 (DENS trains): Day 1 + Day 2 | Done (20-21 Oct), committed. Superseded as main result |
| **Rev 3b: FACED trains; DENS, DEAP, DREAMER zero-shot** | **Done (22 Oct)** |
| **Pipeline audit ("random beats us")** | **Done (22 Oct): no code/label/metric bug; two fixable weaknesses found** |
| **Rev 3c: 30 channels, per-subject normalisation, spectral features, smoothing** | **Done (23 Oct): best results so far** |
| Qwen explanations / FaithScore / human-eval pairs for the FACED model | **Not yet**: these exist only for the DENS model (`reports/dens12/`) |
| Results summary, deck updates | Not started |
| Human evaluation with real raters | Not started (raters/ethics not arranged) |
| Commit of 22-23 Oct work | **Not committed yet** (last commit `a5ebe07`, Day 2) |

## Current best result (run `faced30sx`, 23 Oct)

Trained on FACED, tested on 24 held-out FACED subjects, 7 classes (happy, tenderness, anger, fear, disgust, sadness,
neutral). Chance macro-F1 0.143; always predicting "happy" gives 32.2% accuracy but 0.070 macro-F1.

| Model | Accuracy [95% CI] | Macro-F1 [95% CI] | Balanced acc | Per-clip accuracy | Per-clip macro-F1 |
|---|---|---|---|---|---|
| **Main CBM (level123)** | **35.7%** [33.3, 38.3] | **0.295** [0.268, 0.324] | 0.303 | **43.5%** | 0.345 |
| Flat baseline | 37.9% [35.6, 40.1] | 0.331 [0.304, 0.358] | 0.324 | 42.7% | 0.352 |

"Per clip" = one prediction per (subject, 30-s clip), averaging the 7 window predictions of that clip; "per window" is
the stricter main metric. Ensemble of the 5 fold models; CIs resample subjects. Source:
[reports/faced30sx/in_corpus.md](reports/faced30sx/in_corpus.md).

How we got here (main CBM, FACED test subjects):

| Run | What changed | CBM acc / macro-F1 | Flat acc / macro-F1 |
|---|---|---|---|
| `faced12` (22 Oct) | FACED trains, 12 channels, frozen EEGPT | 21.6% / 0.197 | 25.2% / 0.227 |
| `faced30` (23 Oct) | + all 30 channels + per-subject embedding normalisation | 24.1% / 0.228 | 30.5% / 0.266 |
| **`faced30sx`** (23 Oct) | + spectral features + 12-ch-view augmentation + loss balancing + 3-window smoothing within a clip | **35.7% / 0.295** | **37.9% / 0.331** |

---

## Timeline

| Date | Commit | What |
|---|---|---|
| 16 Oct | `dfd3e99` | Repository skeleton: `src/hcbm/` packages, configs, Makefile, smoke tests |
| 16 Oct | `c1b2423` | Phase 1: dataset inspection; FACED chosen as primary (rev 2); protocol decisions locked |
| 16 Oct | `817d162` | Phase 2: preprocessing pipeline, FACED/DEAP/DREAMER loaders, cross-dataset QC |
| 19 Oct | `51200ce` | All 32 DEAP subjects, EEGPT encoder wrapper, splits, embedding script |
| 19 Oct | — | Full pipeline built and run end to end on a random-weight encoder (plumbing check, tag `12ch_random`) |
| 20 Oct | `bf40e80` | Phase 3-4 code committed |
| 20 Oct | — | Real EEGPT checkpoint verified; GPU environment (`.venv-gpu`) working |
| 20 Oct | `cc77072` | Plan rev 3: DENS trains, DEAP/DREAMER test, FACED deferred. Day 1 implemented |
| 21 Oct | `a5ebe07` | Day 2: training fix (D24), full DENS run, calibration fix (D25), demo script, leakage tests |
| 21 Oct | uncommitted | Fine-tuning experiment (D26), feature check, FACED-vs-DENS checks |
| 22 Oct | uncommitted | Rev 3b: FACED trains, DENS joins the zero-shot test sets (D27); run `faced12` |
| 22 Oct | uncommitted | Audit of the weak results; full-montage + feature code (D28) |
| 23 Oct | uncommitted | Rev 3c: per-subject normalisation + smoothing (D29); runs `faced30`, `faced30sx` |

---

## Datasets (current roles)

| Dataset | Role (rev 3b) | Processed files | Subjects | Windows | Labels |
|---|---|---|---|---|---|
| **FACED** | **Train / val / in-corpus test** | `data/processed/faced/windows_{12ch,full}.h5` | 119 | 23,324 (28 clips x 7) | 9 categories → 7 classes, V/A 0-7 |
| DENS | Zero-shot test | `data/processed/dens/windows_{12ch,full}.h5` | 40 | 1,900 | 5 of the 7 classes (surprise has no FACED class), V/A/D 1-9 |
| DEAP | Zero-shot test | `data/processed/deap/windows_{12ch,full}.h5` | 32 | 19,200 | V/A/D 1-9, quadrants |
| DREAMER | Zero-shot test | `data/processed/dreamer/windows_12ch.h5` | 23 | 6,210 | V/A/D 1-5, 16/18 clips mapped (calm → neutral; 2 surprise clips excluded) |

- **12-ch windows:** F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8, the channels shared by all four datasets.
- **Full windows:** FACED's 30 EEG channels. DEAP has all 30; DENS's 128 EGI electrodes are mapped to them by nearest angle
  (mean 6.1°, max 13.1°); DREAMER only has the 12.
- **All windows:** 4 s at 256 Hz, band-passed, common average reference, per-subject z-score, artifact flag kept.
- **Splits:** FACED uses 24 test subjects + 5 subject-wise folds, fixed on 16 Oct before any training
  (`data/processed/faced/splits.json`).

### Audit findings that shaped the protocol
- **DEAP:** the local `.dat` copy differs from the official release on 439/1280 trials (V/A set to 9-x on every
  high-valence/high-arousal trial). Labels are taken from `participant_ratings.xls`, which is genuine. Re-download DEAP before
  publication (D2).
- **DENS:** the official OpenNeuro release has truncated EEG for 30/40 subjects, so the older release's 475 click-locked
  epochs are used (D17). Click numbers repeat for mit035 (D20).
- **DREAMER:** labels verified; clip-to-emotion table from the paper (D12).
- **FACED:** 27/119 subjects' valence ratings don't follow the clips (r < 0.5 vs other raters). Their valence/arousal loss
  is masked in training (D11). Labels checked on 22 Oct: negative clips are rated valence 1.5-2.0, neutral ~3.1 with the
  lowest arousal, positive 4.0-5.2.

---

## Rev 3b (22 Oct): FACED as the training set

**Why:** the DENS run (rev 3, below) gave below-chance test scores with confidence intervals too wide to compare models: only
475 epochs and 8 test subjects. A feature check showed FACED has a clearer signal, and pooling FACED with DENS hurt DENS
(0.207 → 0.117 macro-F1 on 5 shared classes), so FACED replaces DENS rather than being added to it (D27).

**What changed:**
- `class_set: faced7`; target files are per class set (`targets_faced7.h5`).
- Dominance is a FACED pseudo-label again (loss weight 0.25).
- EEGPT input scale re-chosen on FACED train/val: 0.01.
- Training runs on the GPU.
- The number of classes is now read from the class set; it was hard-coded to 6, which crashed the first attempt.

**Result (`faced12`):**
- First run where every model beats chance on held-out subjects: CBM 21.6% / 0.197, flat 25.2% / 0.227.
- The CBM is 3 points of macro-F1 behind flat: H1 (non-inferiority) not shown.
- Escalation (H5) works on FACED test subjects.

## Audit (22 Oct): why do simple baselines look better?

Checks run on train/val subjects or recomputed from saved predictions; nothing was changed while auditing.

| Check | Result |
|---|---|
| Clip labels vs subjects' own ratings | Correct (see FACED bullet above) |
| Metrics recomputed independently from saved predictions | Match the reports exactly |
| EEGPT input units | Match pretraining (millivolts; z-scored x 0.01 ≈ 10 µV) |
| Keeping EEGPT's 16 time patches instead of averaging | No gain (probe 0.195-0.204 vs 0.204) |
| **Variance in EEGPT embeddings explained by subject vs emotion** | **12.2% vs 0.3%**: embeddings mostly encode *who*, not *what is felt* |
| **Per-subject z-scoring of embeddings** | **Probe 0.204 → 0.225 macro-F1** |
| **Published-style baseline on raw FACED (30 ch, DE features, smoothing, linear SVM)** | **7-class 32.0% acc / 0.291 F1** (9-class 26.9%, 32.6% per clip) |
| Same baseline on our 12-ch windows | 7-class 27.8% / 0.240: the 12-channel limit costs ~0.05 |

The "random beats us" impression came from accuracy alone. "Always happy" scores 32.2% accuracy but 0.070 macro-F1;
uniform random guessing scores 13.9% / 0.131. Our models train with class weights, so they spread their predictions
across classes. Reports now also give balanced accuracy and per-clip results.

**Published FACED numbers for context** (9 classes, all 32 channels, end-to-end training, so not directly comparable):
EEGNet 40.9% balanced acc, SDC-Net 42.4% (leave-one-subject-out), LaBraM 52.7%, CBraMod 55.1%, CSBrain 57.5%.

## Rev 3c (23 Oct): four improvements (D28, D29)

| Step | Change | Code |
|---|---|---|
| 1 | Per-subject z-scoring of EEGPT embeddings (unsupervised) | `subject_zscore` in `src/hcbm/train/data.py`, `--subj-norm` |
| 2 | All 30 channels for FACED/DEAP/DENS (DREAMER 12) | `scripts/build_full.py`, `src/hcbm/io/montage.py`, `--montage full`; scale re-chosen (0.01) |
| 3 | Spectral features (292: log/relative theta-alpha-beta-gamma power x 30 ch + 13 asymmetry pairs) appended to the embedding; each training window also enters as its 12-ch view; learned loss balancing | `src/hcbm/concepts/spectral.py`, `scripts/build_spectral.py`, `--extra`, `--aug-tag`, `--loss-balance uncertainty` |
| 4 | 3-window moving average within each clip; balanced-accuracy and per-clip reporting | `smooth_within_clip`, `--smooth 3`; `scripts/evaluate.py` |

Runs: `faced30` = steps 1-2; `faced30sx` = steps 1-4 (`scripts/run_full.sh`). Steps 1 and 4 use the unlabelled windows of the
subject/clip being predicted, as in the FACED paper's own normalisation and smoothing. This is stated in D29.

**Results of `faced30sx`** (beyond the table at the top):
- **H1 (CBM within 2 points of flat):** not met per window, CBM − flat macro-F1 = −0.036 [−0.055, −0.016]. Per clip the two
  are equal (0.345 vs 0.352).
- **H5 (escalation):** supported everywhere. Escalated cases have higher error and risk-coverage beats random on FACED
  validation, FACED test and DREAMER, for both models.
- **Sim 1, zero-shot (H2), mostly not supported:**

| Test set | CBM vs flat, paired bootstrap |
|---|---|
| DEAP | arousal +1.7 points (significant); valence via the emotion head −1.1 (significant, worse) |
| DENS | valence via the emotion head +5.2 (significant) |
| DREAMER | no significant difference |

  Raw levels stay near chance: valence/arousal accuracy 0.49-0.61, DEAP quadrants 26-27%, DREAMER 7-class 25%.
- **Ablation:**

| Variant | Test macro-F1 |
|---|---|
| flat | 0.331 |
| level1_only | 0.322 |
| bypass | 0.319 |
| level12 | 0.312 |
| level123_l2 | 0.303 |
| level123 (main) | 0.295 |

  Each extra concept level costs about 1 point, the price of interpretability.

### Bugs found and fixed on 22-23 Oct
- `HierarchicalCBM._apply` overrode PyTorch's `Module._apply`, which breaks `.to(device)`. Renamed `_intervene`. Only GPU use
  was affected.
- The number of classes was hard-coded to 6 (DENS). It is now read from the active class set.
- The loader read `fine`/`rater_r` after closing the targets file when extra features were used. Fixed before any affected
  run.
- `select_scale.py --montage full` looked for labels in the 30-ch file. Labels are now read from the 12-ch file (same rows).

---

## Earlier experiments (DENS as training set, 20-21 Oct)

Kept for the record; superseded by rev 3b/3c.

### Rev 3 Day 1 (20 Oct)
- Configs for DENS (6 classes, EGI channel map, gamma 30-40 Hz), DENS loader (1,900 windows, 40 subjects), splits
  (8 test subjects).
- Every script made dataset-agnostic; concept targets; QC (dataset-of-origin classifier 0.68 vs 0.33 chance, i.e. a large
  shift between datasets).
- 38 tests.
- EEGPT scale 0.1 chosen on DENS. Gate-1 probe: 6-class macro-F1 0.176 vs 0.167 chance.

### Rev 3 Day 2 (21 Oct)
- **Fixes:**
  - Training: batch 64, 30-epoch warm-up (D24). The CBM had been stopped while still predicting one class.
  - Calibration: temperature clipped to [0.1, 10] (D25). Fitted temperatures up to 3.7e6 had flattened every prediction to
    1/6.
  - Dominance no longer labelled "pseudo" for DENS; explanation limit raised to 256 tokens.
- **Results on DENS test subjects (6 classes):**
  - Below chance: CBM 15.1% / 0.135, flat 12.5% / 0.100.
  - Escalation failed on the test group.
- **FaithScore** (Qwen2.5-1.5B, 96 DENS + 30 DEAP windows):

| | CBM | Flat |
|---|---|---|
| Sentiment judge | 0.60 | 0.78 |
| Valence-word judge | 0.81 | 0.06 |
| Placebo false-positive rate | 0.30 / 0.47 | — |

  H3 not supported. These explanations predate the D25 prompt fixes.
- **Demo:** `scripts/demo.py` walks one window through every stage.
- **Tests:** leakage audit added; 42 tests.

### Side experiments (21 Oct)
- **Fine-tuning EEGPT on DENS** (last 2 blocks, D26, `runs/dens12_ft`): CBM 0.135 → 0.151, flat 0.100 → 0.116 with
  overlapping CIs. No real gain, because there is too little data.
- **Feature check** ([reports/feature_check.md](reports/feature_check.md), emotion macro-F1 on train/val subjects):

| Features | DENS | FACED |
|---|---|---|
| EEGPT embedding | 0.180 | 0.188 |
| Level-1 band powers | 0.193 | 0.229 |
| Spectral features | 0.197 | 0.229 |

---

## Pipeline components (code)

| Component | Files |
|---|---|
| Preprocessing, HDF5 schema | `src/hcbm/preprocess/pipeline.py`, `src/hcbm/io/schema.py` |
| Loaders, montages | `src/hcbm/io/{faced,dens,deap,dreamer,montage}.py`; builders `scripts/build_{faced,dens,cross_corpus,full}.py` |
| EEGPT encoder | `src/hcbm/models/eegpt_encoder.py` (frozen, strict load, 4 x 512 = 2048 per window); `scripts/extract_embeddings.py --montage {12ch,full}`, `scripts/select_scale.py` |
| Concepts and features | `src/hcbm/concepts/level{1,2,3}.py` (37 L1 concepts on 12 ch, V/A/D, appraisal pseudo-labels), `src/hcbm/concepts/spectral.py` (292 spectral features) |
| Models | `src/hcbm/models/hcbm.py` (L1 → L2 → L3 → emotion; variants; concept interventions), `flat.py`, `confidence.py` (temperature scaling, escalation) |
| Training | `src/hcbm/train/{trainer,data,runs,splits,finetune}.py`, `scripts/train.py` (5 subject-wise folds, warm-up + early stopping, lr x dropout grid, GPU, options `--subj-norm --smooth --extra --aug-tag --loss-balance`), `scripts/finetune.py`, `scripts/refit_confidence.py` |
| Evaluation | `scripts/evaluate.py` (in-corpus incl. balanced/per-clip, Sim 1 on DENS/DEAP/DREAMER, ablation), `src/hcbm/eval/metrics.py`, `scripts/probe.py`, `scripts/feature_check.py` |
| Human-in-the-loop | `src/hcbm/hitl/intervention.py`, `scripts/hitl.py` |
| Explanations / Sim 2 | `src/hcbm/explain/{prompts,generator}.py`, `scripts/explain.py`, `scripts/faithscore_agreement.py` |
| Human evaluation | `scripts/make_eval_set.py`, `scripts/analyse_ratings.py`, `docs/EVAL_PROTOCOL.md` |
| One command | `scripts/run_all.sh` (12-ch pipeline incl. Qwen), `scripts/run_full.sh` (rev 3c runs A and B) |
| Demo | `scripts/demo.py` (one window through every stage) |

## Where things are

| What | Location |
|---|---|
| Raw data (read-only) | `datasets/` |
| Processed windows, targets, features, splits | `data/processed/<dataset>/{windows_12ch.h5, windows_full.h5, targets_faced7.h5, spectral_full.h5, splits.json}` (DENS run: `targets.h5`) |
| Embeddings | `data/embeddings/<dataset>_<tag>.h5`. Tags: `faced12` (12 ch), `faced30` (30 ch), `faced30v12` (12-ch view for augmentation), `dens12` (DENS run) |
| Input-scale choices | `reports/scale_selection_{dens,faced,faced_full}.json` |
| EEGPT checkpoint | `checkpoints/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt` (sha256 `9d63ebc8…`) |
| Trained models, predictions | `runs/<tag>/` (`faced30sx` = current best) |
| Reports | `reports/<tag>/` (`faced30sx`, `faced30`, `faced12`, `dens12`, `dens12_ft`); `reports/feature_check.md`, `reports/qc.md` |
| Environments | `.venv` (CPU, tests), `.venv-gpu` (CUDA torch + transformers) |

## How to run

Current best pipeline (FACED trains, 30 channels; about 1-1.5 h on the GPU after `build_full.py`):
```bash
.venv/bin/python -E -P scripts/build_full.py                  # 30-ch windows (once)
PY=".venv-gpu/bin/python -E -P" DEVICE=cuda scripts/run_full.sh
.venv/bin/pytest -q                                            # 42 tests
```
Explanations + human-eval pairs for a run (not yet done for `faced30sx`):
```bash
P=".venv-gpu/bin/python -E -P"; $P scripts/explain.py --tag faced30sx --backend qwen --device cuda --n 120 --n-deap 30 && \
  $P scripts/make_eval_set.py --tag faced30sx && $P scripts/analyse_ratings.py --tag faced30sx --synthetic
```

---

## Open items

1. **Commit the 22-23 Oct work** (FACED switch, audit fixes, full montage, D26-D29, runs `faced12`/`faced30`/`faced30sx`).
2. **Explanations, FaithScore and human-eval pairs for `faced30sx`** (~40 min GPU); then label 20 FaithScore pairs.
3. **Approve the Level-3 appraisal profiles** (`configs/appraisal_profiles.yaml`; still a draft).
4. **Raters:** at least 3 for the human evaluation, and whether ethics approval is needed.
5. **Results summary and deck updates.** The deck still says SEED-VII, 7 classes incl. surprise, and Cohen's kappa.
6. **Next accuracy step:** fine-tune EEGPT on FACED with 30 channels. Published fine-tuned models reach ~0.53-0.58; code
   exists (`scripts/finetune.py`), ~2 h GPU.
7. DEAP: re-download from the official source before publication.

## Known limitations
- The main CBM is ~3.6 points of macro-F1 behind the flat baseline per window (equal per clip). Interpretability has a
  measurable cost.
- Zero-shot transfer to DEAP/DREAMER/DENS is close to chance; datasets differ strongly (QC domain classifier 0.68 vs 0.33).
- Frozen EEGPT embeddings are dominated by subject identity (12.2% vs 0.3% of variance); hand-made spectral features carry
  more emotion information on this task.
- Per-subject normalisation and within-clip smoothing use unlabelled data from the test subject/clip (transductive).
- The four negative FACED emotions (and the four positive ones) have almost identical valence/arousal ratings, so they are
  hard to separate.
- Level 3 is a category pseudo-label, so correcting it in the intervention experiment leaks the class.
- FaithScore so far (DENS model) had a high placebo rate; Qwen 1.5B sometimes contradicts the supplied facts.
