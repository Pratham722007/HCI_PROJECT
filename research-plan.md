# H-CBM-EEG: Research Plan

**Hierarchical Concept Bottleneck Model for Trustworthy EEG-to-Emotion LLMs**

> **Rev 2 changes (read first).** All data is now in `datasets/`. SEED-VII is unavailable, so **FACED** (123 subjects, 28 clips, 32 ch, 250 Hz, 9 emotion categories) replaces it as the primary train/validation/test corpus. DEAP and DREAMER stay as frozen zero-shot cross-corpus tests. DENS (only 3 of 40 raw subjects on disk) is demoted to an optional secondary in-corpus check. Consequences: the shared montage shrinks from 14 to **12 channels** (FACED has no AF3/AF4), FACED has **no dominance rating**, and the 7-class set loses **surprise**. §2, §4, and Phases 0-4 below are rewritten accordingly.

Sources: the deck `H-CBM-EEG_Presentation.pptx (1).pdf` (7 Oct 2026), the execution plan `H-CBM-EEG Execution Plan and Antigravity Prompts (DENS to DEAP DREAMER).docx`, and the current repo skeleton (`configs/`, `src/hcbm/`, `docs/DATASETS.md`). Where the deck and the execution plan disagree, **the execution plan wins** (DENS replaces SEED-VII). All such deviations are listed in §2.

Plan date: 16 Oct 2026. **Revised 16 Oct 2026 (rev 2): FACED is now the primary training dataset (replacing SEED-VII); DENS is demoted to optional/secondary.**

---

## 1. Research Problem and Hypotheses

### 1.1 Problem (deck slide 2)
EEG-to-LLM emotion systems (e.g. E²-LLM) predict and explain emotions, but:

| # | Challenge | Consequence |
|---|-----------|-------------|
| 1 | Hallucinated explanations | Fluent text not grounded in EEG evidence; faithfulness unverifiable |
| 2 | Black-box projection (opaque MLP) | Clinicians cannot inspect intermediate reasoning; no psychological concepts |
| 3 | No human validation | Only automated metrics; no expert trust or clinical-utility evaluation |

**Research gap:** no EEG-LLM framework offers all three of (a) interpretable reasoning, (b) faithfulness verification, (c) human-in-the-loop validation.

### 1.2 Proposed method
Frozen **EEGPT** encoder → **3-level hierarchical concept bottleneck** (inspired by Scherer's Component Process Model) → emotion head → **Qwen** explanation, with a confidence estimator routing low-confidence cases to a human reviewer.

```
EEG window (12ch shared montage, 4 s, 256 Hz)
  → EEGPT encoder (frozen) → embedding (512-d)
  → Level 1  Neurophysiological : theta / alpha / gamma power (x12 ch) + frontal alpha asymmetry (37 concepts)
  → Level 2  Affective          : valence, arousal (real ratings), dominance (pseudo-label, see §4.2)
  → Level 3  Cognitive appraisal: novelty, goal conduciveness, coping potential
  → Emotion classifier (7 classes)
  → Confidence estimator P(concept)
        ├─ high confidence → Qwen explanation
        └─ low confidence  → human reviewer (escalate / intervene)
```

### 1.3 Research questions (deck slide 5)

| RQ | Question | Expected result | Primary evidence |
|----|----------|-----------------|------------------|
| **RQ1** | Does the hierarchical CBM match or beat a flat projection in accuracy, and generalise better cross-corpus? | Comparable in-corpus, better cross-corpus | Sim 1 + Sim 3 (ablation) |
| **RQ2** | Are CBM explanations more faithful (do they change when a concept is flipped)? | Higher FaithScore than flat | Sim 2 (counterfactual perturbation) |
| **RQ3** | Do experts rate CBM explanations as more trustworthy and clinically useful, and does low confidence flag wrong predictions? | Higher human ratings; low confidence ↔ errors | Human eval + HITL experiments |

### 1.4 Hypotheses (falsifiable)
- **H1:** CBM macro-F1 on FACED held-out subjects is within ±2 pp of the flat baseline (non-inferiority).
- **H2:** CBM cross-corpus (DEAP/DREAMER, zero-shot) valence/arousal accuracy exceeds flat by a paired-bootstrap-significant margin.
- **H3:** FaithScore(CBM) > FaithScore(flat), and the placebo-perturbation false-positive rate is low (<15%).
- **H4:** Blind raters score CBM higher on faithfulness, plausibility, clarity, actionability (Wilcoxon, FDR-corrected).
- **H5:** Error rate among escalated (low-confidence) cases is higher than among retained cases; AURC beats random escalation.
- **H6:** Intervening on concepts (replacing predicted with ground truth) monotonically raises accuracy.

> **Report negative results honestly.** The deck states "expected results"; the paper must state which ones the data did not support.

### 1.5 Contributions to demonstrate (deck slide 6)
1. Hierarchical concept bottleneck for EEG-LLM.
2. **FaithScore**: faithful explanation changes ÷ total perturbations.
3. Dual human-in-the-loop: clinician intervention + confidence escalation.
4. Blind human-judge evaluation (faithfulness, plausibility, clarity, actionability).

---

## 2. Deviations from the Deck (Decisions Already Made)

| Deck says | Reality / decision | Action |
|-----------|-------------------|--------|
| Primary dataset SEED-VII (20 subj, 62 ch, 24,444 segments, 7 classes) | Not available. **FACED**: 123 subj, 28 clips/subj, 32 ch, 250 Hz, 30 s/clip, 9 categories (anger, disgust, fear, sadness, amusement, inspiration, joy, tenderness, neutral). 4 s windows give 7/clip, about 24,100 windows, close to SEED-VII's 24,444 | Update slide 3 |
| "Same dataset as E²-LLM, so baseline comparison is fair" | No longer true | **Delete.** Compare only against our own flat baseline on identical embeddings/splits. Never compare to E²-LLM's published numbers. |
| Cohen's kappa with 3+ raters | Cohen's kappa is for 2 raters only | Use **Fleiss' kappa or Krippendorff's alpha** |
| 7 emotion classes incl. surprise | FACED has no surprise | Coarse 7-class set redefined (§4.2). Surprise dropped from the deck |
| Level 2 = valence, arousal, dominance | FACED has valence/arousal self-ratings but **no dominance** (verify against its behavioural file) | V and A trained on real ratings; dominance is a category-prototype pseudo-label, flagged as such (or dropped; see §4.2) |
| EEGPT on full montage | 32 / 32 / 14 channels across FACED / DEAP / DREAMER | Primary setting = **shared 12-channel montage** (FACED ∩ DEAP ∩ DREAMER); full-montage FACED = secondary |
| DENS as primary (execution plan) | Only 3 raw subjects local | Optional secondary only |

---

## 3. Current State of the Repo (Audit, 16 Oct 2026)

**Done (Prompt 1 skeleton):**
- `configs/{protocol,paths,model}.yaml` populated (**14** shared channels, 4 s windows @ 256 Hz, band edges, 4-45 Hz bandpass, 50 Hz notch, thresholds, 5-fold, 20% held-out). Written for DENS; needs the rev 2 changes below.
- `src/hcbm/{io,preprocess,concepts,models,train,eval,explain,hitl,utils}/` stubs (only `utils/config.py`, `utils/seed.py` have code).
- `environment.yml`, `pyproject.toml`, `Makefile`, `tests/test_smoke.py`, `docs/DATASETS.md`.

**Data on disk (`datasets/`, untracked), checked 16 Oct 2026:**

| Dataset | What is present | Gap |
|---------|-----------------|-----|
| **FACED** (6.2 GB) | `sub000.pkl` … `sub122.pkl` (123 files, 53.7 MB each). Each is a NumPy array `(28, 32, 7500)` float64 = clips x channels x samples (30 s @ 250 Hz). | **No channel names, labels, or ratings in the files.** Need FACED's channel list, clip→category table and self-report file (valence/arousal etc.) from the dataset release. Confirm clip order. |
| **DEAP** (1.6 GB) | `data_preprocessed_python/s01..s32.dat`, Metadata (`participant_ratings.xls`, etc.) | Ready |
| **DREAMER** | `DREAMER.mat` (432 MB) + example notebooks | Ready |
| **DENS** (4.6 GB) | Raw: `mit003/004/014.zip` (3 of 40 subjects), group-wise preprocessed `EEG_Data.zip`, behavioural zip, papers, stimuli | Too few subjects for subject-wise CV. Optional secondary only. |

**Inconsistencies to fix in the repo:**
- `configs/protocol.yaml` and `README.md` still say DENS is primary, 14 shared channels, 256 Hz. Needs: FACED as primary, 12 shared channels (drop AF3, AF4), a `faced` thresholds entry (scale from its rating file).
- `configs/model.yaml` `embedding_dim: 512` is an assumption → verify against EEGPT's actual output (Phase 3).
- `README.md`/`pyproject.toml` do not mention the deck's SEED-VII; fine, but the deck itself must be updated (Phase 11).
- `configs/emotion_taxonomy.yaml` is referenced but **does not exist yet**. `configs/model.yaml` lists 7 classes including surprise; replace per §4.2. L1 `output_dim: 43` becomes 12 ch x 3 bands + 1 FAA = **37**.
- `.gitignore` is modified and `datasets/` is untracked; make sure raw data and the PDF are ignored or intentionally tracked before the first data commit.

---

## 4. Experimental Protocol (Locked Before Any Training)

Write all of these into `configs/protocol.yaml` and commit **before** running experiments. Changing a locked item after seeing results must be logged as a deviation.

### 4.1 Hard rules
1. `data/raw/` is read-only. Datasets under `datasets/` are copied/linked in, never modified.
2. No absolute paths in code; everything from `configs/*.yaml`.
3. Seed 42 default; log all seeds.
4. **DEAP and DREAMER are never used for tuning**: hyperparameters, loss weights, temperature scaling, escalation threshold are chosen on FACED validation subjects only.
5. **Splits are by subject**, asserted in code (no subject in both train and test).
6. One unified HDF5 schema: `X [n_windows, channels, samples] float32`, `ch_names`, `fs=256`, `subject_id`, `trial_id`, `window_idx`, `labels/` (raw + processed), `meta` JSON (provenance).
7. **No bypass** around the concept bottleneck in the main model (bypass is an extra ablation only).
8. If real data contradicts an assumption, stop and record it; do not silently adapt.

### 4.2 Locked design decisions

| Item | Decision |
|------|----------|
| Emotion taxonomy | `configs/emotion_taxonomy.yaml`. FACED 9 → coarse 7: **happy** (amusement, joy, inspiration), **tenderness**, anger, fear, disgust, sadness, neutral (**surprise removed**; the happy/tenderness grouping is a proposal for you to approve). Fine 9-class label kept. DREAMER 9 → coarse 7 where possible (amusement, excitement → happy; calmness → neutral; **secondary**, loose); DEAP = quadrants only |
| Rating binarisation | FACED at the midpoint of its own scale (confirm from rating file); DEAP 5 (1-9); DREAMER 3 (1-5). Also report Spearman (scale-free). Fixed before seeing results |
| Windowing | 4 s non-overlapping @ 256 Hz = 1024 samples (confirm against EEGPT). FACED: 30 s clip @ 250 Hz → resample to 256 Hz → 7 windows/clip (2 s discarded), about 123 x 28 x 7 = 24,100 windows. All windows of a clip share its label, so **never split windows of one clip/subject across train and test** |
| Preprocessing | Bandpass 4-45 Hz (zero-phase Butterworth), 50 Hz notch, common average reference, resample to 256 Hz, per-subject z-score, amplitude-threshold artifact rejection (report rejection rate per subject) |
| Shared montage | **12 ch**: F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 (FACED lacks AF3/AF4, so the old 14-ch set cannot be used); CAR computed **over those 12**. FAA = ln(α F4) − ln(α F3). Verify FACED channel names first (not stored in the pickles) |
| Splits (FACED) | 123 subjects: subject-wise 5-fold CV + one fixed held-out test group (20% ≈ 25 subjects). Validation only from FACED training subjects. Note: all subjects watch the same 28 clips, so clip identity is shared between train and test; report this as a limitation and add a clip-held-out check (hold out e.g. 1 of 3-4 clips per category) as a secondary result |
| Level 1 | Welch band power (theta 4-8, alpha 8-13, gamma 30-45) per channel + FAA; z-score per subject; tertile bins per subject. **Identical code for all datasets**; computed from the 12-ch windows only, never from model outputs. Note gamma (30-45 Hz) is muscle-contaminated in scalp EEG; flag in limitations |
| Level 2 | Valence and arousal from real self-ratings (FACED, DEAP, DREAMER; rescaled to 0-1 per dataset, plus per-subject z-scores). **Dominance: FACED has none**, so train it on category-prototype pseudo-labels (flagged) with a masked loss, evaluate it only against real ratings on DEAP/DREAMER as an exploratory result. Alternative: drop dominance entirely (decision needed) |
| Level 3 | Category-level pseudo-labels from Scherer CPM appraisal predictions, in `configs/appraisal_profiles.yaml` (novelty, goal conduciveness, coping in [-1, 1]; each row has citation + confidence high/med/low). **Reviewed and approved by you before freezing. Always labelled pseudo-labels.** |
| Baseline | Flat 2-layer MLP + GELU on the same embeddings, same splits, similar parameter order |
| Explainer | Qwen2.5 1.5B-3B Instruct, deterministic decoding, prompting only (no LoRA), must use only supplied concepts |
| Statistics | Bootstrap CIs resampling **subjects** (not windows); paired bootstrap/permutation for CBM vs flat; Wilcoxon + effect size for ratings; FDR correction across criteria |

### 4.3 Metrics

| Experiment | Metrics |
|-----------|---------|
| In-corpus FACED | Accuracy, macro-F1 (coarse 7-class and fine 9-class), concept accuracies (L1 R²/corr, L2 binary acc, L3 MAE), mean ± std across folds |
| Cross-corpus (Sim 1) | Valence/arousal binary acc & macro-F1 (DEAP, DREAMER); DEAP 4-quadrant acc; DREAMER 7-class (secondary); Spearman/Pearson of predicted vs. real L2 with subject-bootstrap CIs |
| Ablation (Sim 3) | Same metrics for flat / L1 only / L1+L2 / L1+L2+L3 (/ bypass extra) |
| FaithScore (Sim 2) | Faithful changes ÷ total perturbations; placebo false-positive rate; judge reliability vs. your 20 manual labels |
| HITL | Intervention curves (0-100% concepts corrected, random vs. uncertainty-ordered); risk-coverage curve, AURC, ECE before/after temperature scaling |
| Human eval | Means ± bootstrap CI per criterion, Wilcoxon + effect size, Fleiss' kappa / Krippendorff's alpha, split by correct vs. wrong prediction |

---

## 5. Implementation Plan by Phase

Timeline is indicative (12 weeks, from 16 Oct 2026 ≈ done by early Jan 2027). **Build order: finish Phases 1-3 on FACED first, then add DEAP/DREAMER via the same scripts.** Every phase has a **gate**: do not proceed until it passes.

### Phase 0: Access, setup, risk removal (Week 1)

| Step | Task | Output |
|------|------|--------|
| 0.1 | Get EEGPT code + checkpoint from the official repo; record path in `configs/paths.yaml` (currently `checkpoints/eegpt_pretrained.pt`) | Checkpoint on disk, sha256 noted |
| 0.2 | Create Hugging Face account / Qwen access; Kaggle or Colab account for the GPU stage | Accounts ready |
| 0.3 | Build env: `conda env create -f environment.yml` (Python 3.11, CPU torch); `make setup && make test` | Green smoke test |
| 0.4 | **Recruit ≥3 evaluators now** (longest lead time; ideally clinicians/psychologists, plus consent) | Names + availability |
| 0.5 | **Obtain FACED metadata** (not in the pickles): channel list/order, clip→category table, per-subject self-report ratings (valence, arousal, discrete emotion intensities), clip order in each pickle. Source: the FACED release (Synapse). Decide dominance handling (§4.2). DENS needs no further download unless you want the secondary check | `docs/FACED_META.md` + raw metadata files |
| 0.6 | Confirm DEAP/DREAMER licence/EULA status is documented (files are already local; record terms in `docs/DATASETS.md`) | Updated registry |
| 0.7 | Fix `.gitignore` so `datasets/`, `data/raw`, `data/processed`, `data/embeddings`, `checkpoints/` are ignored | Clean `git status` |

**Gate 0:** env works, EEGPT checkpoint present, evaluator recruitment started, FACED metadata in hand.

### Phase 1: Data acquisition and inspection (Weeks 1-2)

| Step | Task | Output |
|------|------|--------|
| 1.1 | Link FACED pickles read-only into `data/raw/faced/` (do not unpickle untrusted files outside an isolated env; use `python -I`). Keep DENS archives untouched unless the secondary check is pursued | Linked tree |
| 1.2 | Write `scripts/inspect_faced.py` → `reports/faced_inspection.md` answering **all** items below with evidence (filenames, example values) | Inspection report |
| 1.3 | Same for DEAP (`scripts/inspect_deap.py`) and DREAMER (use `datasets/dreamer-eeg.ipynb` as a starting point) | Short reports |
| 1.4 | Update `docs/DATASETS.md` statuses (`not downloaded` → `downloaded` → `verified`) | Registry |

**Items the FACED inspection (`scripts/inspect_faced.py` → `reports/faced_inspection.md`) must answer:**
- (a) pickle contents: array shape per subject (confirmed `(28, 32, 7500)` float64 on sub000), identical across all 123?; NaN/Inf/flat channels; amplitude range; evidence of prior filtering/ICA/referencing; fs (expected 250 Hz)
- (b) channel names and order; does it include A1/A2 or others; mapping to the 12 shared channels
- (c) clip order and clip→emotion-category table; how many clips per category
- (d) rating file: columns, scale, range, missing values; is there any dominance column
- (e) subject IDs vs. file indices (sub000..sub122); any excluded/low-quality subjects in the release
- (f) per-subject spectra and amplitude sanity plots (4-45 Hz power, 50 Hz line noise present or removed)
- (g) the same inspection for DEAP and DREAMER (use `datasets/dreamer-eeg.ipynb` and `deap/EDA_DEAP.ipynb` as starting points)
- (h) optional: DENS archive structure, if you want the secondary check

**Gate 1 — FACED part PASSED 16 Oct 2026** (see `reports/faced_inspection.md`, `reports/deap_dreamer_inspection.md`, `docs/DEVIATIONS.md`); DEAP/DREAMER caveats recorded there. Original criterion: report answers a-h; wrong assumptions listed at the top; decisions in §4.2 updated (rating scale, dominance, channel set).

### Phase 2: Loaders and preprocessing (Weeks 2-3)

| Step | Module / file | Notes |
|------|---------------|-------|
| 2.1 | `src/hcbm/preprocess/pipeline.py`: pure functions: `bandpass`, `notch`, `car`, `resample`, `zscore_subject`, `cut_windows`, `reject_artifacts` | Identical for all datasets; params from `protocol.yaml` |
| 2.2 | `src/hcbm/io/faced.py` + `configs/channel_map_faced.yaml` | Load each `subNNN.pkl` (28 clips x 32 ch x 7500), select the 12 shared channels by name, resample 250→256 Hz, shared pipeline (data may already be filtered; apply only what is missing), cut 7 x 4 s windows per clip; labels: fine category, coarse 7-class, valence/arousal ratings raw + per-subject normalised, subject_id, trial_id (= clip), window_idx. DEAP also has 32 ch here but keep the 12 for the primary setting |
| 2.3 | `src/hcbm/io/deap.py` | `pickle.load(f, encoding='latin1')`; keep first 32 EEG ch; drop first 3 s baseline (384 samples @128 Hz); data already 4-45 Hz, CAR, 128 Hz → resample to 256 Hz, z-score; ~15 windows/trial, ~19,200 total; labels V/A/D/liking + binary @5 + quadrants |
| 2.4 | `src/hcbm/io/dreamer.py` | Last 60 s of each stimulus clip; remove DC offset, shared bandpass/notch, amplitude rejection (no ICA), CAR over the 12 shared channels, resample, z-score; ~6,200 windows; V/A/D (1-5) + binary @3 + 9-emotion → 7-class map |
| 2.5 | `src/hcbm/io/schema.py`: HDF5 writer + validator | Used by every loader |
| 2.6 | Outputs per dataset: `windows_12ch.h5` (primary) and `windows_full.h5` (FACED/DEAP 32 ch where EEGPT's vocabulary allows; secondary) | |
| 2.7 | `docs/EEGPT_REQUIREMENTS.md`: EEGPT's expected window length, fs, channel vocabulary, amplitude scaling | **If it conflicts with 4 s / 256 Hz, stop and decide** |
| 2.8 | `scripts/qc_cross_dataset.py` → `reports/qc.md` | See below |

**QC report contents:** mean spectra per dataset; band-power distributions pre/post z-score; amplitude distributions; rejection rates; windows per subject; label distributions; **domain-shift check** (logistic regression on band-power predicting dataset of origin; chance = 1/3; high accuracy = strong dataset shift that the paper must discuss).

**Tests:** schema validator; no NaN/Inf; shape `(n, C, 1024)`; synthetic 10 Hz sine survives bandpass and 50 Hz sine is removed; same 12-channel order in all three `windows_12ch.h5`; FACED about 24,100 windows, DEAP/DREAMER window counts within 5% of expected.

**Gate 2 — PASSED 16 Oct 2026 for 12-ch pipeline** (`reports/qc.md`, domain-shift acc 0.64 vs 0.33 chance → substantial shift, to be discussed; windows_full.h5 and EEGPT requirements doc (2.6/2.7) still open). Original criterion: QC report shows comparable spectra and counts; all tests pass.

### Phase 3: EEGPT embeddings (Weeks 3-4)

| Step | Task |
|------|------|
| 3.1 | `src/hcbm/models/eegpt_encoder.py`: load checkpoint, `eval()`, no grad, replicate EEGPT's channel vocabulary and amplitude scaling exactly |
| 3.2 | `scripts/extract_embeddings.py <dataset> <full\|12ch>`: batched, resumable, `--device` flag, progress bar + ETA; writes `data/embeddings/<dataset>_<variant>.h5` row-aligned with windows (same subject_id, trial_id, window_idx) + hash of source file and checkpoint. CPU is fine (EEGPT ≈10M params), run once and cache |
| 3.3 | Primary: `12ch` for all three datasets. `full` for FACED only (secondary) |
| 3.4 | `scripts/probe.py`: logistic regression on frozen FACED embeddings (subject-wise split) for binary valence, binary arousal and coarse emotion vs. majority baseline |

**Tests:** row counts match windows files; no NaN; determinism (re-run 50 windows, identical output).

**Gate 3:** probe beats chance on FACED. **If it does not, report that before continuing**; the entire project depends on the embedding carrying signal (fallbacks: check scaling/channel vocab, try the full-montage variant, review window length).

### Phase 4: Concept targets and labels (Week 4)

| Step | File | Notes |
|------|------|-------|
| 4.1 | `src/hcbm/concepts/level1.py` | Welch band power per channel, FAA, per-subject z, tertile bins; same code on all datasets; unit test with synthetic 10 Hz signal |
| 4.2 | `src/hcbm/concepts/level2.py` | Real V/A/D ratings; binarise with locked thresholds; rescale to 0-1 per dataset + keep raw and per-subject z |
| 4.3 | `configs/appraisal_profiles.yaml` + `src/hcbm/concepts/level3.py` | Map each of the **9 FACED categories** (and DREAMER's 9) → novelty/goal/coping with Scherer CPM justification and confidence. Small table, so review it carefully. **Print and get your approval before freezing** |
| 4.4 | `configs/emotion_taxonomy.yaml` | FACED 9→coarse 7, DREAMER 9→7, DEAP quadrants; fine labels retained |
| 4.5 | FACED has per-category intensity ratings (verify). Optional train-only low-intensity filter (off by default; assert it can never touch val/test) | Only if the rating file has them |
| 4.6 | `data/processed/<dataset>/targets.h5` aligned row-by-row with windows | |

**Tests:** row alignment; label-distribution tables; sanity correlations (e.g. frontal asymmetry vs. valence on FACED, reported without claiming significance).

**Gate 4:** Level 1 code identical across datasets; Level 3 table approved.

### Phase 5: Models and training on FACED (Weeks 5-6)

| Step | Task |
|------|------|
| 5.1 | `models/hcbm.py` – `HierarchicalCBM`: embedding → L1 head (hidden [256,128], out **37**) → L2 head (input = **only** L1 outputs; out 3) → L3 head (input = **only** L2 outputs; out 3) → emotion classifier (input = **only** L3 outputs; 7 classes). No residual path from embedding to later stages. A 3-d L3 bottleneck separating 7 classes is tight, so expect a hit and handle as in 5.3 |
| 5.2 | `models/flat.py` – `FlatBaseline`: 2-layer MLP + GELU (hidden 128, dropout 0.1) → 7 classes. Report parameter counts of both |
| 5.3 | Variant flags: `level1_only`, `level12`, `level123`, `bypass` (extra ablation); note that the pure L3→emotion chain is restrictive. If it hurts accuracy badly, document it and (only then) test the variant where emotion head also sees L2, as a labelled variant, not the main model |
| 5.4 | `models/confidence.py`: sigmoid/softmax concept probabilities, temperature scaling and escalation threshold fit **on FACED validation only** |
| 5.5 | `train/`: subject-wise 5-fold + fixed held-out test group; joint loss `w1·L1 + w2·L2 + w3·L3 + w4·L_emotion`; small documented hyperparameter search in `configs/model.yaml`; early stopping on validation; CSV logging |
| 5.6 | **Concept leakage check**: the bottleneck must be the only path; also check L1 predictions are not trivially decoded without information (compare against a "shuffled targets" run) |

**Tests:** subject disjointness across folds; **gradient-path test** proving the emotion loss reaches the embedding only through concept layers; overfit a tiny batch; results table (acc, macro-F1, concept accuracies; CBM vs. flat; mean ± std across folds).

**Gate 5:** both models trained on identical splits; in-corpus table produced. (No DEAP/DREAMER touched.)

### Phase 6: Simulation 1, cross-corpus generalisation (Week 7)

Freeze all models, temperatures, thresholds. Evaluate **zero-shot, no fine-tuning** on DEAP and DREAMER 12ch embeddings.
- Valence/arousal binary acc & macro-F1 on both; DEAP 4-quadrant acc.
- Spearman/Pearson of predicted L2 vs. real ratings, bootstrap CI resampling subjects.
- DREAMER 7-class accuracy labelled **secondary**.
- CBM vs. flat, same layout, with paired bootstrap/permutation test.
- Any few-shot calibration goes in a **separate** section, never mixed with zero-shot.

**Output:** `reports/sim1.md` + CSV. A statement at the top lists every metric where the CBM is **not** better. Audit: no code path where DEAP/DREAMER labels reach training or threshold selection.

### Phase 7: Ablation (Weeks 7-8)

Evaluate `flat`, `level1_only`, `level12`, `level123` (+ `bypass` extra) on FACED in-corpus and DEAP/DREAMER cross-corpus with identical splits and seeds → one grouped bar chart + one table (`reports/sim_ablation.md`). Answers: which level contributes what.

### Phase 8: Qwen explanation + FaithScore (Weeks 8-9, GPU stage)

No local GPU, so heavy work is a notebook: `notebooks/kaggle_explain.ipynb` (run on Kaggle/Colab) plus a CPU fallback for 20 examples.

| Step | Task |
|------|------|
| 8.1 | `explain/prompts.py`: serialise predicted L1 bins, L2 values, L3 values, emotion + confidence as structured text; instruct Qwen to use **only** supplied concepts and not invent EEG facts. Qwen2.5-1.5B/3B-Instruct, 4-bit if needed, temperature 0 / fixed seed |
| 8.2 | Same explainer on the flat baseline's output (emotion + its continuous projection in same format without concept names), so human eval compares like with like |
| 8.3 | **FaithScore**: on FACED test windows, flip predicted valence (pos↔neg), propagate through L3 and the emotion head, regenerate; auto-judge whether explanation changed correctly using (a) sentiment/NLI model on before/after text and (b) check that the explanation mentions the changed concept; store both signals separately; FaithScore = faithful changes ÷ total perturbations |
| 8.4 | **Placebo control**: flip a concept that should not change valence wording (e.g. a Level 1 band bin) and report how often polarity changes anyway = metric false-positive rate |
| 8.5 | Repeat on a DEAP subset and on a sample of the flat baseline |
| 8.6 | Judge reliability: 20 hand-readable before/after pairs; you label manually; report judge agreement |

**Output:** FaithScore table with CIs, placebo rate, 20 example pairs, judge reliability note. Note: the flat baseline has no concepts to flip, so define its perturbation (e.g. flip the emotion/projection summary) explicitly and describe the asymmetry as a limitation.

### Phase 9: Human-in-the-loop experiments (Weeks 9-10)

| Experiment | Spec |
|-----------|------|
| **Intervention curve** (`hitl/intervention.py`) | Replace 0%, 10%, … 100% of predicted concepts with ground truth; run per level (L1, L2, L3), random vs. uncertainty-ordered selection. DEAP/DREAMER use real V/A/D as the "expert" correcting L2. **L3 curve labelled "pseudo-expert"**. Plot mean + 95% CI across subjects |
| **Escalation** (`hitl/escalation.py`) | Risk-coverage curve, AURC, accuracy at the pre-chosen threshold (from FACED validation), vs. random-escalation baseline; ECE before/after temperature scaling on FACED val, FACED test, DEAP, DREAMER |

**Output:** figures + short markdown stating whether low-confidence cases really are the wrong ones, **including negative findings**.

### Phase 10: Human evaluation (Weeks 10-12; start scheduling in Week 1)

| Step | Task |
|------|------|
| 10.1 | `scripts/make_eval_set.py`: ~150 CBM-vs-flat explanation pairs, stratified by emotion class and correct/wrong prediction, mostly FACED test + a smaller DEAP share; randomise A/B with saved seed and **private key file**; scrub system-revealing text |
| 10.2 | Rater interface (local HTML/Streamlit or Google Form export): 1-5 scales for faithfulness, plausibility, clarity, actionability + free text; ground-truth label hidden unless protocol says otherwise |
| 10.3 | `docs/EVAL_PROTOCOL.md`: evaluator instructions, criterion definitions, ethics note, consent text (get institutional ethics sign-off if required **before** recruiting) |
| 10.4 | Pilot with 2-3 pairs; fix wording |
| 10.5 | Collect ratings from ≥3 evaluators |
| 10.6 | `scripts/analyse_ratings.py`: unblind, per-criterion means + bootstrap CI, Wilcoxon signed-rank + effect size, Fleiss' kappa / Krippendorff's alpha, split by correct vs. wrong predictions, FDR correction |

**Tests:** dry run with synthetic ratings from 3 fake raters produces the full report; unblinding confirms A/B balance.

### Phase 11: Writing and deck update (Week 12)

- `docs/PAPER_DRAFT.md`: abstract, method, datasets + taxonomy, protocol, all results with CIs, limitations, and a statement of which expected results the data did not support. **Only numbers from `reports/` files.**
- Deck edits: slide 3 (SEED-VII → FACED stats (123 subjects, 28 clips, 32 ch, about 24,100 windows, 9 categories → coarse 7, surprise removed); DEAP/DREAMER as cross-corpus test sets; note L2 = real self-ratings, L3 = pseudo-labels); slide 3 footer (delete "Same dataset as E²-LLM…", replace with "flat baseline re-implemented on identical embeddings and splits"); slide 5 (evaluation-data row; Cohen's kappa → Fleiss' kappa/Krippendorff's alpha; remove E²-LLM numerical comparison claims); update slide 6 contributions with measured numbers.

**Gate 11:** every slide claim backed by a saved result.

---

## 6. Proposed Repository Additions

```
configs/        emotion_taxonomy.yaml, appraisal_profiles.yaml, channel_map_faced.yaml
docs/           EEGPT_REQUIREMENTS.md, EVAL_PROTOCOL.md, PAPER_DRAFT.md
scripts/        inspect_{faced,deap,dreamer}.py, qc_cross_dataset.py, extract_embeddings.py,
                probe.py, make_eval_set.py, analyse_ratings.py, run_all.sh
src/hcbm/       io/{faced,deap,dreamer,schema}.py, preprocess/pipeline.py,
                concepts/level{1,2,3}.py, models/{eegpt_encoder,hcbm,flat,confidence}.py,
                train/, eval/, explain/, hitl/{intervention,escalation}.py
notebooks/      kaggle_explain.ipynb
reports/        faced_inspection.md, qc.md, sim1.md, sim_ablation.md, faithscore.md, hitl.md, human_eval.md
```

`make` targets already exist (`setup test preprocess embed train eval`); add `qc`, `explain`, `hitl`, `humaneval`.

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **FACED metadata missing** (channel names, clip labels, ratings are not in the pickles) | Confirmed | High: cannot build any label or channel map | Phase 0.5: fetch from the FACED release before anything else |
| EEGPT embeddings from 12 channels carry little signal | Medium | Critical | Gate 3 probe; check scaling/vocab; try full montage; consider an alternative encoder only as documented deviation |
| Strict L1→L2→L3→emotion chain lowers accuracy | Medium | Medium | Report honestly; test emotion-head-sees-L2 as labelled variant; H1 is non-inferiority, not superiority |
| Cross-corpus shift (FACED: Neuroscan-style 32-ch lab recording of Chinese-subject film clips vs. DEAP music videos vs. DREAMER Emotiv; 32/32/14 ch → 12) | High | Medium | Domain-shift check in QC; per-subject z-score; Spearman reporting; discuss as limitation |
| Clip identity shared across subjects (same 28 clips), windows within a clip are highly correlated | High | Medium | Subject-wise splits plus clip-held-out secondary check; never split a clip's windows across folds |
| Level 3 has no real labels | Certain | Medium | Pseudo-labels labelled as such everywhere; "pseudo-expert" intervention curve; limitation in paper |
| FaithScore judge itself unreliable / flat baseline has no concepts to perturb | Medium | Medium | Two separate signals, placebo control, 20 manual labels, document asymmetry |
| Qwen hallucination despite prompt | Medium | Medium | Constrained prompt, deterministic decoding, post-hoc check that only supplied concepts are mentioned |
| Evaluator recruitment/ethics delay | High | High (blocks Phase 10) | Start Week 1; pilot early; need ≥3 raters |
| No local GPU | Certain | Low | Embeddings on CPU (once, cached); Qwen stage in Kaggle/Colab notebook with 20-example CPU fallback |
| Test leakage (tuning on DEAP/DREAMER) | Low if rules held | Critical | Hard rule 4; code audit; assert in tests |
| DREAMER's 7-class mapping is loose | Certain | Low | Secondary result only |

---

## 8. Milestones and Deliverables

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Access + setup | Env, EEGPT weights, FACED metadata, evaluators recruited |
| 2 | Inspection | `reports/faced_inspection.md` |
| 3 | Preprocessing | 3× `windows_12ch.h5`, `qc.md` |
| 4 | Embeddings + targets | `data/embeddings/*`, `targets.h5`, probe result |
| 6 | Models | In-corpus CBM vs. flat table |
| 7 | Sim 1 | `reports/sim1.md` |
| 8 | Ablation | `reports/sim_ablation.md` |
| 9 | FaithScore | `reports/faithscore.md`, 20 examples |
| 10 | HITL | `reports/hitl.md` |
| 12 | Human eval + paper | `human_eval.md`, `PAPER_DRAFT.md`, updated deck |

## 9. Reproducibility Checklist

- [ ] Single command regenerates every table from saved prediction files
- [ ] Seeds, config hashes, EEGPT checkpoint hash, git commit stored with each result
- [ ] Subject-disjointness asserted in tests; no DEAP/DREAMER label reaches training/threshold code
- [ ] Every number in the paper/deck traceable to `reports/`
- [ ] Negative results reported
- [ ] Raw data untouched; derived data regenerable via `make preprocess embed`

## 10. Immediate Next Actions (this week)

1. Fetch FACED metadata (channel list, clip→category table, ratings) and decide the dominance handling.
2. Obtain the EEGPT checkpoint and put its path in `configs/paths.yaml`.
3. Message ≥3 prospective evaluators; check whether ethics approval is needed.
4. Fix `.gitignore` for `datasets/` and large artifacts.
5. Start Phase 1: write `scripts/inspect_faced.py` and answer items a-h.
