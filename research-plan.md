# H-CBM-EEG: Research Plan (rev 3, 20 Oct 2026)

**Hierarchical Concept Bottleneck Model for Trustworthy EEG-to-Emotion LLMs**

> **Rev 3 changes (read first).** Training data is now **DENS** (40 subjects); **DEAP** and **DREAMER** remain frozen zero-shot
> test sets; **FACED is deferred** (its processed data and code stay, so it can be added back as a second training corpus).
> Goal for this revision: a **complete, correct pipeline in 3 days** (21-23 Oct 2026). Result quality is secondary;
> every stage must run end to end and produce its report. Consequences: 6 emotion classes instead of 7 (no neutral in the
> DENS epochs), real dominance ratings (the FACED dominance pseudo-label goes away), gamma band 30-40 Hz (DENS is low-passed
> at 40 Hz), and the 12-channel shared montage is kept so FACED stays compatible.

Sources: the deck `H-CBM-EEG_Presentation.pptx (1).pdf` (7 Oct 2026), the dataset audits of 18-20 Oct 2026, and the code
already in `src/hcbm/` and `scripts/` (rev 2 built and tested the full pipeline on FACED; rev 3 re-points it at DENS).

---

## 1. Research problem, method, questions (from the deck, unchanged)

**Problem (slide 2).** EEG-to-LLM emotion systems (e.g. E²-LLM) have (1) hallucinated, unverifiable explanations, (2) an
opaque MLP projection with no intermediate psychological concepts, (3) no human validation. No framework offers
interpretable reasoning + faithfulness verification + human-in-the-loop validation together.

**Method (slide 4).**
```
EEG window (12 ch, 4 s, 256 Hz) -> EEGPT encoder (frozen) -> embedding (4 x 512 = 2048-d)
  -> Level 1 Neurophysiological : theta / alpha / gamma power x 12 ch + frontal alpha asymmetry (37 concepts)
  -> Level 2 Affective          : valence, arousal, dominance (real self-ratings on DENS)
  -> Level 3 Cognitive appraisal: novelty, goal conduciveness, coping (category pseudo-labels, Scherer CPM)
  -> Emotion classifier (6 classes)
  -> Confidence estimator P(concept / class)
       |- high confidence -> Qwen explanation
       '- low confidence  -> human reviewer (escalate / intervene)
```

| RQ | Question | Evidence |
|---|---|---|
| RQ1 | Does the hierarchical CBM match the flat projection in-corpus and generalise better cross-corpus? | Sim 1 + Sim 3 |
| RQ2 | Are CBM explanations more faithful (change when a concept is flipped)? | Sim 2 (FaithScore) |
| RQ3 | Do experts rate CBM explanations as more trustworthy/useful, and does low confidence flag errors? | Human eval + HITL |

Hypotheses H1-H6 are kept from rev 2: H1 CBM macro-F1 within +-2 pp of flat on DENS test subjects; H2 CBM beats flat
zero-shot on DEAP/DREAMER valence/arousal (paired subject bootstrap); H3 FaithScore(CBM) > FaithScore(flat) with placebo
false-positive rate < 15%; H4 raters prefer CBM (Wilcoxon, FDR); H5 escalated cases have higher error, AURC < random;
H6 concept intervention raises accuracy. **Negative results are reported as such.**

Contributions (slide 6): hierarchical CBM for EEG-LLM; FaithScore; dual human-in-the-loop (intervention + escalation);
blind human-judge evaluation.

---

## 2. Datasets (checked on disk 20 Oct 2026)

| Dataset | Role | What we use | Subjects | Labels |
|---|---|---|---|---|
| **DENS** (Mishra et al. 2023, OpenNeuro ds003751) | **Training / validation / in-corpus test** | 475 click-locked 7 s epochs, 128-ch EGI, 250 Hz, already 1-40 Hz filtered (`datasets/dens_old/Physiological Data/PreProcessed-Emotion (Group-wise)/EEG_Data/Group-01..08/*.mat`, each `data` (128, 1751) float32 µV) | 40 (4-32 epochs each) | self-reported emotion word per click (8 groups) + that trial's valence/arousal/dominance/liking (1-9) |
| **DEAP** | Zero-shot test | `data/processed/deap/windows_12ch.h5` (already built) | 32 | V/A/D 1-9 from `participant_ratings.xls` (local `.dat` copy is modified, D2); quadrants |
| **DREAMER** | Zero-shot test | `data/processed/dreamer/windows_12ch.h5` (already built) | 23 | V/A/D 1-5; clip -> target emotion table (D12) |
| FACED | Deferred (optional 2nd training corpus) | `data/processed/faced/` kept | 119 | 9 categories, V/A |

### 2.1 DENS: why the click-locked epochs are the primary training data
- **Raw continuous EEG is incomplete in the official release.** OpenNeuro v1.0.6 has truncated `.fdt` files for 30/40
  subjects (audit 20 Oct); only mit003, 004, 061, 076, 108, 117, 121, 123, b2017007 are complete (+ mit014 from the old
  release's intact `.mff`). Ten subjects are too few for subject-wise CV.
- **The group-wise epochs cover all 40 subjects** and carry a categorical label: each 7 s epoch is locked to a mouse click
  where the participant marked feeling a named emotion during a 60 s clip (the dataset's "temporally localised emotional
  events"). This matches how the DENS authors themselves classify emotions.
- **Ratings join exactly.** Filename `<Emotion>_<subject>Trial-<t>Click-<k>-slor.mat`: `t` is the 1-based row in the
  subject's behavioural CSV (`dens_old/.../Raw Data/Behavioural Data/<subject>.csv`, rows with a valence response;
  neutral_1 is trial 1), `k` the subject's cumulative click count. Verified: all 475 epochs find their trial row.
- **Optional extension (not in the 3-day scope):** raw 60 s clip windows for the 10 complete subjects (ratings only, plus
  neutral clips) via `mne.io.read_raw_eeglab`; events are in samples at 250 Hz.

### 2.2 DENS caveats to state in the paper
- Preprocessing of the group-wise epochs is the authors' (1-40 Hz; the `-slor` suffix and reference are undocumented).
  We re-apply our own CAR over the 12 channels and per-subject z-score; we do not undo theirs.
- Labels are self-reported, click-localised, and unevenly distributed (4-32 epochs per subject; mit076/081 have 4).
- 475 epochs is small (SEED-VII on the deck had 24,444 segments). Expect high variance; report CIs.
- Only 7 s per label, so windows overlap (section 4.2); epochs are never split across folds.

---

## 3. Decisions (locked before training; logged in `docs/DEVIATIONS.md`)

| Item | Decision |
|---|---|
| Emotion classes | **6 coarse classes: happy, surprise, fear, disgust, anger, sadness** (deck's 7 minus neutral; DENS click epochs contain no neutral). Mapping in 3.1 |
| Shared montage | Keep **12 ch**: F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 (FACED-compatible; DEAP/DREAMER files reused as-is). EGI mapping in 3.2 |
| Level 1 | Welch band power theta 4-8, alpha 8-13, **gamma 30-40 Hz** (was 30-45; DENS is low-passed at 40) per channel + FAA = ln(alpha F4) - ln(alpha F3); per-subject z; tertile bins. Same code on all datasets |
| Level 2 | DENS: real valence, arousal, **dominance** (1-9, rescaled to 0-1, binarise > 0.5 = rating > 5, same rule as DEAP). DEAP 1-9 @5, DREAMER 1-5 @3; ties dropped. D3 dominance pseudo-label retired for DENS |
| Level 3 | Category pseudo-labels from `configs/appraisal_profiles.yaml`, a `dens` section with the 6 classes (Scherer CPM; **draft, needs your approval**). Always labelled pseudo-labels |
| Windows | 7 s epoch @ 250 Hz -> resample to 256 Hz (1792 samples) -> **4 x 4 s windows, stride 1 s** (overlapping); ~1,900 windows. All windows of an epoch share its label and fold |
| Preprocessing (DENS) | Select 12 channels -> band-pass 4-45 Hz (effective 4-40) -> CAR over the 12 -> resample 250->256 -> artifact flag (D8 rule) -> per-subject z-score. No notch needed (already low-passed at 40 Hz) |
| Splits | Subject-wise: fixed test group of 8 subjects (20%) + 5 folds over the other 32; seed 42; asserted in code |
| Baseline | Flat 2-layer MLP + GELU on identical embeddings and splits (never compared to E²-LLM's published numbers) |
| Explainer | Qwen2.5-1.5B-Instruct (3B if GPU time allows), greedy decoding, prompting only, must use only the supplied concepts |
| Statistics | Subject-bootstrap CIs; paired bootstrap CBM vs flat; Wilcoxon + rank-biserial + BH-FDR for ratings; Fleiss' kappa / Krippendorff's alpha (not Cohen's kappa, which is for 2 raters) |
| Hard rules | DEAP/DREAMER never used for any choice (hyperparameters, temperature, thresholds, EEGPT input scale); `datasets/` read-only; no bypass around the bottleneck in the main model |

### 3.1 Emotion taxonomy (6 classes)

| Coarse class | DENS group (self-reported words) | DENS epochs | DREAMER clips |
|---|---|---|---|
| happy | G1 Amused, Delighted, Happy, Joyous + G2 Adventurous, Aroused, Excited, Lust, Passionate | 62 + 68 = 130 | amusement 3, 12; excitement 5, 16; happiness 7, 13 |
| surprise | G3 Alarmed, Startled, Tense | 62 | surprise 2, 18 |
| disgust | G4 Disgust, Hate | 65 | 6, 10 |
| fear | G5 Afraid | 50 | 4, 15 |
| anger | G6 Angry, Distress | 62 | 8, 14 |
| sadness | G7 Depressed, Despondent, Dissatisfied, Melancholic, Miserable, Taken Aback + G8 Sad | 61 + 45 = 106 | 9, 17 |
| *(excluded)* | none | none | calmness 1, 11 |

G3 is closer to "alarm" than pure surprise and G7 contains "Taken Aback"; we follow the DENS authors' grouping and say so.
DREAMER now has 16/18 clips with a class (was 14 under rev 2). DEAP: quadrants only.

### 3.2 EGI HydroCel-128 -> 10-10 channels
Nearest electrode by angle after centring both MNE montages (`GSN-HydroCel-129` vs `standard_1005`):

| 10-10 | F7 | F3 | FC5 | T7 | P7 | O1 | O2 | P8 | T8 | FC6 | F4 | F8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EGI | E33 | E24 | E34 | E39 | E58 | E70 | E83 | E96 | E115 | E116 | E124 | E122 |
| angle | 6.4° | 2.9° | 0.5° | 7.1° | 5.3° | 4.9° | 4.9° | 5.3° | 6.5° | 0.9° | 4.0° | 6.7° |

T7/T8 are the loosest (EGI's own 10-20 tables often give E45/E108); confirm against EGI's equivalence chart on Day 1 and
freeze in `configs/protocol.yaml` before any embedding. Epoch rows are E1..E128 (Cz reference not included).

---

## 4. What already exists (reused unchanged or with small edits)

Built and tested in rev 2 (32 tests pass; full run verified on placeholder embeddings 19 Oct; real EEGPT checkpoint
strict-loads, sha256 `9d63ebc8...`, at `checkpoints/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`):

| Component | File(s) | Rev 3 change |
|---|---|---|
| Preprocessing functions, HDF5 schema | `preprocess/pipeline.py`, `io/schema.py` | none |
| DEAP / DREAMER windows | `data/processed/{deap,dreamer}/windows_12ch.h5` | none |
| EEGPT encoder + extraction (resumable, GPU) | `models/eegpt_encoder.py`, `scripts/extract_embeddings.py`, `scripts/select_scale.py` | primary dataset from config |
| Concepts L1/L2/L3 + targets | `concepts/level{1,2,3}.py`, `scripts/build_targets.py` | DENS branch; gamma 30-40 |
| CBM (5 variants), flat, temperature/escalation | `models/{hcbm,flat,confidence}.py` | `n_classes` 6 |
| Training, 5-fold CV, hyperparameter grid | `train/{trainer,data,runs,splits}.py`, `scripts/train.py` | primary dataset from config |
| In-corpus / Sim 1 / ablation reports | `scripts/evaluate.py` | dataset names from config |
| HITL (interventions, escalation) | `hitl/intervention.py`, `scripts/hitl.py` | dataset names from config |
| Qwen explainer, FaithScore, placebo | `explain/{prompts,generator}.py`, `scripts/explain.py` | dataset names; class list |
| Human eval set, offline rater form, analysis | `scripts/{make_eval_set,analyse_ratings,faithscore_agreement}.py` | none |
| One-command run | `scripts/run_all.sh` (`PY=.venv-gpu/bin/python`, `DEVICE=cuda`) | none |

---

## 5. Three-day implementation plan

Every day ends with `pytest` green and the day's reports regenerated. Run tag: `dens12` (real EEGPT).

### Day 1 (21 Oct): DENS data in, pipeline re-pointed
| # | Task | Output |
|---|---|---|
| 1.1 | Freeze configs: `protocol.yaml` (`primary_dataset: dens`, `dens:` paths + EGI channel map + epoch params, gamma 30-40, DENS thresholds), `emotion_taxonomy.yaml` (6 classes, DENS groups, DREAMER map), `appraisal_profiles.yaml` (`dens` section), `model.yaml` (`n_classes: 6`) | configs committed **before** any embedding |
| 1.2 | `src/hcbm/io/dens.py` + `scripts/build_dens.py`: parse filenames, join ratings, select 12 ch, preprocess (3), cut 4 windows/epoch; labels: group, coarse, valence/arousal/dominance/liking raw, epoch_id | `data/processed/dens/windows_12ch.h5` (~1,900 x 12 x 1024) |
| 1.3 | `scripts/make_splits.py dens` | `data/processed/dens/splits.json` (8 test subjects + 5 folds) |
| 1.4 | Remove FACED hard-coding: scripts read `primary_dataset` / `cross_corpus_datasets` from config; FACED-only rules (D11 rater mask) apply only when present | same scripts, dataset-agnostic |
| 1.5 | `build_targets.py` for dens/deap/dreamer (gamma 30-40; DREAMER 6-class) | `targets.h5` x 3 |
| 1.6 | QC: extend `qc_cross_dataset.py` to DENS (spectra, rejection, label counts, domain-shift check) | `reports/qc.md` |
| 1.7 | Tests: DENS schema/counts, epoch-not-split-across-folds, rating join, 6-class taxonomy | `tests/test_dens.py` |
| 1.8 | EEGPT: `select_scale.py` on DENS train/val subjects, then embeddings for dens/deap/dreamer on GPU (~minutes) + Gate-3 probe | `data/embeddings/*_dens12.h5`, `reports/probe_*.json` |

**Gate 1:** DENS file validates; splits disjoint; probe runs (beating chance is reported, not required, in rev 3).

### Day 2 (22 Oct): models, simulations, explanations
| # | Task | Output |
|---|---|---|
| 2.1 | `train.py --tag dens12 --search`: flat + level1_only / level12 / level123 / level123_l2 / bypass, 5 folds | `runs/dens12/` |
| 2.2 | `evaluate.py`: in-corpus DENS (acc, macro-F1, concept accuracy), **Sim 1** zero-shot DEAP/DREAMER (V/A binary, Spearman, DEAP quadrants, DREAMER 6-class), **Sim 3** ablation | `reports/dens12/{in_corpus,sim1,sim_ablation}.md` |
| 2.3 | `hitl.py`: intervention curves (L1, L2 real, L3 pseudo-expert), escalation (risk-coverage, AURC, ECE) | `reports/dens12/hitl.md` |
| 2.4 | `explain.py --backend qwen --device cuda --n 120 --n-deap 30`: **Sim 2** FaithScore + placebo, both judges | `reports/dens12/faithscore.md`, `explanations.jsonl` |
| 2.5 | Label 20 FaithScore pairs by hand (you), `faithscore_agreement.py` | judge reliability |

**Gate 2:** all reports generated; leakage audit (no DEAP/DREAMER label in any training/threshold path) passes.

### Day 3 (23 Oct): human evaluation materials, write-up, buffer
| # | Task | Output |
|---|---|---|
| 3.1 | `make_eval_set.py` (~150 blinded pairs), pilot the rater form on 2-3 items, fix wording | `reports/dens12/human_eval/` |
| 3.2 | `analyse_ratings.py --synthetic` dry run (real ratings come when raters return them) | `human_eval_DRYRUN.md` |
| 3.3 | Results summary: every number from `reports/`, list of hypotheses not supported | `docs/RESULTS_SUMMARY.md` |
| 3.4 | Deck updates (section 8) | updated slides |
| 3.5 | Buffer: fix whatever broke; optional FACED as second training corpus or DENS raw clips | |

---

## 6. Experiments and metrics

| Experiment | Data | Metrics |
|---|---|---|
| In-corpus | DENS test subjects (8) | accuracy, macro-F1 (6 classes), L1 Pearson r, L2 binary acc (V/A/D), L3 MAE; fold mean +- std and 5-fold ensemble with subject-bootstrap CI |
| **Sim 1** cross-corpus | DEAP, DREAMER (zero-shot, frozen) | V/A binary acc + macro-F1 (L2 head and via emotion head for both models), Spearman vs ratings, DEAP 4-quadrant acc, DREAMER 6-class acc (secondary), dominance Spearman (now trained on real DENS dominance), paired bootstrap CBM vs flat |
| **Sim 2** FaithScore | DENS test + DEAP subset | valence flip propagated through L3 and emotion head; judge (a) sentiment classifier, (b) valence-term check; placebo L1 text flip; flat baseline flips the class summary (asymmetry stated) |
| **Sim 3** ablation | all | flat vs level1_only vs level12 vs level123 (+ labelled variants level123_l2, bypass) |
| HITL | DENS test (+ DREAMER) | intervention curves random vs uncertainty-ordered; risk-coverage, AURC vs random, error escalated vs retained, ECE before/after temperature scaling |
| Human eval | ~150 pairs, >= 3 raters | 1-5 faithfulness, plausibility, clarity, actionability; Wilcoxon, effect size, FDR; Fleiss' kappa, Krippendorff's alpha; split by correct/wrong |

---

## 7. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| 475 epochs / 40 subjects is small, classes uneven (50-130 epochs) | wide CIs, unstable CBM | macro-F1, class weights, CIs, honest reporting; results are not the 3-day goal |
| DENS epochs are pre-filtered by the authors (1-40 Hz, undocumented `-slor`) | preprocessing mismatch vs DEAP/DREAMER | gamma narrowed to 30-40 everywhere; domain-shift check in QC; stated as limitation |
| Click-localised 7 s labels (DENS) vs whole-trial ratings (DEAP/DREAMER) | label-granularity shift in Sim 1 | report as a limitation; Spearman alongside binary metrics |
| EGI -> 10-10 mapping approximate (T7/T8 ~7°) | channel mismatch | confirm on Day 1; list mapping in paper |
| 3-d L3 bottleneck for 6 classes is tight | accuracy loss | expected; labelled `level123_l2` variant |
| Qwen 1.5B ignores the facts (seen 19 Oct: wrote "positive" for a negative valence) | low FaithScore | this is what Sim 2 measures; try 3B on GPU if time |
| Raters / ethics not arranged | human eval cannot run in 3 days | deliver form + protocol + dry-run analysis; collect later |
| Overlapping windows inflate apparent sample size | optimistic CIs | bootstrap resamples subjects, never windows |

---

## 8. Deck updates needed

- **Slide 3:** SEED-VII -> **DENS**: 40 subjects, 128-ch EGI (12 shared channels used), 475 click-localised emotion events,
  ~1,900 4 s windows, 6 emotion classes; DEAP = cross-corpus test, DREAMER = cross-corpus test (additional validation).
  Delete "Same dataset as E²-LLM, so the baseline comparison is fair" -> "Flat baseline re-implemented on identical
  embeddings and splits."
- **Slide 3/5:** "7 emotion classes" -> 6 (no neutral); Simulation 1 "Train on DENS, test on DEAP / DREAMER".
- **Slide 5:** Cohen's kappa -> Fleiss' kappa / Krippendorff's alpha; evaluation data DENS -> DEAP / DREAMER; remove
  numerical E²-LLM comparisons.
- **Slide 4:** Level 3 values are category pseudo-labels (say so).

---

## 9. Reproducibility checklist
- [ ] `scripts/run_all.sh` regenerates every report from processed data
- [ ] Seeds, config hashes, EEGPT checkpoint sha256, git commit stored with each run (`runs/<tag>/manifest.json`)
- [ ] Subject disjointness and epoch-not-split asserted in tests; no DEAP/DREAMER label reaches training/threshold code
- [ ] Every number in the paper/deck traceable to `reports/`
- [ ] Negative results reported

## 10. Open items for you
1. Approve the Level-3 appraisal profiles for the 6 classes (`configs/appraisal_profiles.yaml`, `dens` section, Day 1).
2. Confirm the 6-class mapping in 3.1 (especially G3 -> surprise, G7 -> sadness).
3. Raters (>= 3) and whether ethics approval is needed.
4. Whether FACED returns as a second training corpus after Day 3.
