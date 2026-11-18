# H-CBM-EEG: Full Project Context

Last updated: 4 Nov 2026. This file is the single place to understand the whole project: what it is, what was built,
every important decision and why, the results, and what is still open. Detailed logs: [progress.md](progress.md)
(chronological progress), [docs/DEVIATIONS.md](docs/DEVIATIONS.md) (decisions D1–D29 with evidence),
[research-plan.md](research-plan.md) (plan, rev 3; partly superseded, see section 3).

---

## 1. The project in one paragraph

**H-CBM-EEG: Hierarchical Concept Bottleneck Model for Trustworthy EEG-to-Emotion LLMs.** We recognise emotions from EEG
with a model whose reasoning is visible:

```
EEG (4-s window) → frozen EEGPT encoder → Level 1 brain-signal concepts → Level 2 affect → Level 3 appraisal → emotion → Qwen explanation
```

- **Level 1, neurophysiological:** theta/alpha/gamma power on 12 channels plus frontal alpha asymmetry (37 concepts).
- **Level 2, affective:** valence, arousal, dominance.
- **Level 3, cognitive appraisal:** novelty, goal conduciveness, coping.

Concepts can be inspected and corrected. An LLM (Qwen) explains predictions using only the concepts. A confidence
estimator would route uncertain cases to a human. The model is compared against an opaque flat MLP baseline.

The original proposal is the deck [H-CBM-EEG_Presentation.pptx (1).pdf](H-CBM-EEG_Presentation.pptx%20(1).pdf) (7 Oct 2026).
Its research questions:
- **RQ1:** accuracy and cross-dataset generalisation vs the flat model.
- **RQ2:** faithfulness of explanations (FaithScore).
- **RQ3:** expert trust, human-in-the-loop, and whether low confidence flags errors.

Team of 5; the professor received a progress report ([h-cbm-eeg.pdf](h-cbm-eeg.pdf)).

---

## 2. Current state (headline)

- **Pipeline:** runs end to end. Data → EEGPT → concepts → CBM / flat models → evaluation → zero-shot tests → Qwen
  explanations + FaithScore. Human-evaluation tooling exists but has not been used.
- **Best run `faced30sx`** (FACED trains; 24 held-out subjects; 7 classes; chance macro-F1 0.143):

| Model | Accuracy | Macro-F1 | Balanced accuracy | Per-clip accuracy |
|---|---|---|---|---|
| **Main CBM (level123)** | **35.7%** | **0.295** | 0.303 | **43.5%** |
| Flat baseline | 37.9% | 0.331 | 0.324 | 42.7% |
| Always guess "happy" | 32.2% | 0.070 | 0.143 | — |
| Uniform random | 13.9% | 0.131 | 0.143 | — |

- **Zero-shot** on DEAP, DENS and DREAMER (never seen in training): valence/arousal accuracy 49–61%, close to chance.
- **Human-in-the-loop and human validation are NOT done yet**, by the user's explicit statement, 4 Nov. Scripts exist
  (`scripts/hitl.py`, `scripts/make_eval_set.py`) and simulated HITL numbers were computed, but the user removed this part
  from the professor's report. Treat it as the next phase, not a result.
- **Git:** last commit `a5ebe07` (21 Oct, Day 2). Everything since (FACED switch, audit, 30 channels, improvements, reports,
  `progress.md`, `context.md`, the PDF) is **uncommitted** (~47 changed files).

---

## 3. How the research direction changed vs the deck, and why

| Deck (7 Oct) | Now | Why |
|---|---|---|
| Train on SEED-VII | **Train on FACED** (123 subjects, 28 film clips, 9 emotions → 7 classes) | SEED-VII not available. DENS tried first (20–21 Oct) but too small (475 usable epochs; results below chance). FACED is closest in size and design. |
| Test on DEAP / DREAMER | Test zero-shot on **DEAP, DREAMER and DENS** | DENS kept as an extra unseen dataset |
| 7 classes incl. surprise | 7 FACED classes: happy, tenderness, anger, fear, disgust, sadness, neutral | FACED has no surprise |
| "Same dataset as E²-LLM, fair comparison" | Compare only with our own flat baseline on identical inputs and splits | E²-LLM numbers are on SEED-VII, so not comparable |
| 62 channels | 30 channels (FACED's EEG set; DREAMER only 12) | Only these channels exist across the datasets |
| Dominance as real concept | Dominance is a down-weighted pseudo-label for FACED | FACED has no dominance rating |
| Cohen's kappa for 3+ raters | Fleiss' kappa / Krippendorff's alpha | Cohen's kappa is for 2 raters |
| Frozen EEGPT embedding only | EEGPT + 292 spectral features, per-subject normalised, smoothed within clip | Audit showed the embedding mostly encodes subject identity and band powers carry more emotion signal |

Unchanged: the architecture, the flat baseline, the three simulations (cross-dataset, FaithScore valence flip, ablation),
and the RQs.

---

## 4. Datasets and data audit

| Dataset | Role | Subjects | 4-s windows | Labels |
|---|---|---|---|---|
| **FACED** | Train / val / in-dataset test | 119 (4 excluded for artifacts, D1) | 23,324 (28 clips × 7) | 9 categories → 7 classes; valence, arousal 0–7 |
| DENS | Zero-shot test | 40 | 1,900 (475 epochs × 4) | 5 of the 7 classes; V/A/D 1–9 |
| DEAP | Zero-shot test | 32 | 19,200 | V/A/D 1–9, quadrants |
| DREAMER | Zero-shot test | 23 | 6,210 | V/A/D 1–5; 16/18 clips mapped (calm → neutral; surprise excluded) |

**Splits:** FACED uses 24 test subjects + 5 subject-wise folds, fixed 16 Oct before training
(`data/processed/faced/splits.json`).

**Audit findings:**
- **DEAP:** the local `.dat` copy is *modified* on 439/1,280 trials (valence/arousal set to 9−x). Labels are taken from
  the genuine `participant_ratings.xls`. Re-download before publication. Frame this as a local-copy issue, not a DEAP
  flaw (D2).
- **DENS:** the official OpenNeuro release has truncated EEG for 30/40 subjects, so the old release's 475 click-locked
  epochs are used (D17). Click numbers repeat for mit035, so epochs are numbered per subject (D20).
- **FACED:**
  - 27/119 subjects rate valence unlike the others (r < 0.5). Their valence/arousal loss is masked in training (D11).
  - Clip labels verified against ratings: negative clips 1.5–2.0, neutral ~3.1, positive 4.0–5.2.
- **DREAMER:** labels verified; clip → emotion table transcribed from the paper (D12).

**Preprocessing (identical for all datasets):**
- 4–45 Hz band-pass, 50 Hz notch, common average reference, 256 Hz.
- 4-s windows, per-subject z-score, subject-relative artifact flag (D8).
- Two montages:
  - the 12 shared channels (F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8);
  - FACED's 30 EEG channels: DEAP has all 30; DENS's 128 EGI electrodes are mapped by nearest angle (mean 6.1°);
    DREAMER has 12.
- Gamma band is 30–40 Hz (D16).

**Concept targets:**
- **L1** is measured from the EEG (12 channels).
- **L2** comes from participants' ratings.
- **L3** comes from Scherer appraisal-theory profiles per category (`configs/appraisal_profiles.yaml`). **DRAFT, never
  approved.**

---

## 5. Pipeline and methods

- **Encoder:** EEGPT `eegpt_mcae_58chs_4s_large4E.ckpt` (25.3M params, frozen, sha256 `9d63ebc8…`; downloaded manually from
  figshare because scripted download was blocked). Output: 4 summary tokens × 512, mean-pooled over 16 patches = 2,048.
  Input scale chosen on training subjects only (0.01 for FACED).
- **Models:**
  - Main CBM `level123`: L1 (37) → L2 (3) → L3 (3) → emotion. The emotion head sees only L3.
  - Variants: `level1_only`, `level12`, `level123_l2`, `bypass`.
  - Flat baseline: 2-layer MLP + GELU.
- **Training:**
  - Loss: sum of level losses (MSE for concepts, class-weighted cross-entropy for emotion), with learned uncertainty
    weighting in the best run.
  - AdamW; lr × dropout grid search on validation.
  - Batch 64, 30-epoch warm-up, early stopping on validation macro-F1.
  - 5 fold models averaged for test; CIs by bootstrapping subjects.
- **Best-run input:** EEGPT (30 ch, per-subject z-scored) + 292 spectral features, plus a 12-ch view of each training window
  as augmentation, and a 3-window moving average within each clip.
- **Safeguards:**
  - Splits by subject.
  - Test subjects and zero-shot datasets never used for any choice; an automated leakage test checks this.
  - 42 pytest tests, including a gradient-path test proving no bypass of the bottleneck.
  - All decisions logged before results.
- **Explanations:** Qwen2.5-1.5B-Instruct, prompted only with predicted concepts; FaithScore = explanation changes
  correctly after a valence flip, plus a placebo control and two automatic judges.

---

## 6. Experiment history and results

| Run tag | Setup | CBM acc / macro-F1 | Flat acc / macro-F1 | Notes |
|---|---|---|---|---|
| `12ch_random` (19 Oct) | Random-weight encoder, FACED | — | — | Plumbing check only |
| `dens12` (21 Oct) | DENS trains, 6 classes, 12 ch | 15.1% / 0.135 | 12.5% / 0.100 | Below chance; CIs too wide |
| `dens12_ft` (21 Oct) | EEGPT fine-tuned on DENS (last 2 blocks) | 0.151 F1 | 0.116 F1 | No real gain: too little data (D26) |
| `faced12` (22 Oct) | FACED trains, 12 ch | 21.6% / 0.197 | 25.2% / 0.227 | First above-chance run |
| `faced30` (23 Oct) | + 30 ch + per-subject normalisation | 24.1% / 0.228 | 30.5% / 0.266 | |
| **`faced30sx`** (23 Oct) | + spectral features + 12-ch augmentation + loss balancing + smoothing | **35.7% / 0.295** | **37.9% / 0.331** | **Current best** |

**Hypotheses:**

| | Result (best run) |
|---|---|
| H1, CBM within 2 points of flat | Not met per window (−3.6, CI −5.5 to −1.6); equal per clip |
| H2, better zero-shot transfer | Mostly not supported. DEAP arousal +1.7 and DENS valence +5.2 (significant); DEAP valence −1.1 (worse); DREAMER no difference |
| H3, faithful explanations | Measured only on the DENS model. CBM 0.60–0.81 vs placebo 0.30–0.47, so not supported; rerun needed for FACED |
| H4–H6, human evaluation and HITL | Not done (see section 2) |

**Ablation** (macro-F1): flat 0.331 > level1_only 0.322 > bypass 0.319 > level12 0.312 > level123_l2 0.303 >
level123 0.295. Each concept level costs about 1 point.

**Context:** published FACED models reach 41–58% balanced accuracy on 9 classes, but use all 32 channels and end-to-end
training (EEGNet 40.9%, SDC-Net 42.4% LOSO, LaBraM 52.7%, CBraMod 55.1%, CSBrain 57.5%), so they are not directly
comparable.

---

## 7. The audit of weak results (22 Oct)

The user suspected a flaw ("random models perform better"). Checked without changing anything:

| Check | Result |
|---|---|
| Labels, metrics recomputed independently, EEGPT input units | All correct, no bug |
| Keeping temporal patches instead of averaging | No gain |
| "Random beats us" | Only "always happy" beats us on *accuracy*, because of class imbalance; on macro-F1 we are ~3× better. Reports now also show balanced and per-clip accuracy. |
| Subject vs emotion variance in EEGPT embeddings | **Subject 12.2% vs emotion 0.3%.** Per-subject z-scoring: probe 0.204 → 0.225 |
| Published-style baseline (30 ch, DE features, SVM) on raw FACED | 0.291 macro-F1 vs 0.240 on our 12 ch. **The 12-channel limit cost ~0.05.** |

These led to the four improvements in `faced30sx` (D28, D29). Per-subject normalisation and smoothing use the test
subject's *unlabelled* data (transductive, standard in FACED work) and are stated as such.

---

## 8. Key decisions made in this chat (chronological)

1. **Deadline-driven scope (19 Oct):** the user wanted the whole pipeline implemented in 3–4 days, with result quality
   secondary. Plumbing first, using a random-weight placeholder encoder until the real checkpoint arrived.
2. **EEGPT checkpoint (20 Oct):** downloaded manually by the user, then strict-load verified.
3. **DENS as training set (20 Oct, plan rev 3):** user's choice, with DEAP/DREAMER zero-shot and FACED deferred. 6 classes,
   12 channels, gamma 30–40 Hz.
4. **Training fix D24:** the CBM was collapsing to one class on DENS, fixed with batch 64 and a 30-epoch warm-up. Found on
   validation only.
5. **Calibration fix D25:** temperature capped at 10, after fitted values up to 3.7e6 flattened every probability. Also
   fixed the dominance wording in Qwen prompts and raised the explanation limit to 256 tokens.
6. **Fine-tuning experiment D26:** no gain on DENS.
7. **Feature check (21 Oct):** band powers beat EEGPT; FACED → DENS transfer hurt; pooling FACED+DENS hurt (0.207 → 0.117).
8. **Switch to FACED for training (22 Oct, D27):** user approved after the reasoning. DENS becomes a zero-shot test set;
   class set `faced7`; per-class-set target files.
9. **Audit (22 Oct):** found no bug but two weaknesses (see section 7).
10. **Full montage and improvements (22–23 Oct, D28/D29):** 30 channels, per-subject normalisation, spectral features,
    12-ch augmentation, uncertainty loss weighting, within-clip smoothing, balanced and per-clip reporting.
11. **Professor report (24–25 Oct):** `h-cbm-eeg.pdf`, 5 parts for 5 presenters, plus a presentation script
    `scripts/show_results.sh` (no training, ~40 s) and a demo `scripts/demo.py`.
12. **HITL / human validation removed from the report (4 Nov):** the user stated it has not been done.

**Bugs fixed along the way:**
- `HierarchicalCBM._apply` shadowed PyTorch's method, breaking `.to(device)`.
- `n_classes` was hard-coded to 6.
- The loader read fields after closing the file.
- `select_scale --montage full` read labels from the wrong file.

---

## 9. Where things are

| What | Location |
|---|---|
| Raw data | `datasets/` (read-only) |
| Processed data | `data/processed/<ds>/{windows_12ch.h5, windows_full.h5, targets_faced7.h5, spectral_full.h5, splits.json}`. DENS run used `targets.h5`. |
| Embeddings | `data/embeddings/<ds>_<tag>.h5`. Tags: `faced12`, `faced30`, `faced30v12`, `dens12` |
| Trained models | `runs/<tag>/` |
| Reports | `reports/<tag>/{in_corpus,sim1,sim_ablation,hitl}.md`; `reports/feature_check.md`, `reports/qc.md` |
| Configs | `configs/{protocol,model,paths,emotion_taxonomy,appraisal_profiles}.yaml`. Current: `primary_dataset: faced`, `class_set: faced7` |
| Key scripts | `run_full.sh` (best pipeline), `run_all.sh` (12-ch pipeline incl. Qwen), `train.py`, `evaluate.py`, `explain.py`, `finetune.py`, `demo.py`, `show_results.sh` |
| Environments | `.venv` (CPU, tests), `.venv-gpu` (CUDA, RTX 4060) |
| Professor report | `h-cbm-eeg.pdf`. Generator: scratchpad `make_report.py`, not in the repo; regenerate from this file if lost. |

**Show the working pipeline (no training):** `scripts/show_results.sh`

**Full re-run of the best setup** (~1–1.5 h GPU): `PY=".venv-gpu/bin/python -E -P" DEVICE=cuda scripts/run_full.sh`

---

## 10. Open items

1. **Commit** everything since `a5ebe07`.
2. **Human-in-the-loop experiments and human evaluation:** the next phase. Needs raters (≥3) and an ethics decision.
3. **Rerun Qwen explanations / FaithScore for `faced30sx`** (~40 min GPU), then label 20 FaithScore pairs by hand.
4. **Approve the Level-3 appraisal table** (still a draft).
5. **Improve accuracy:** fine-tune EEGPT on FACED with 30 channels (the most promising route to published-level numbers).
6. **Results write-up and deck update:** datasets, classes, kappa; remove the E²-LLM comparison claim.
7. **DEAP:** re-download from the official source before publication.
8. **Known issue: concept leakage.** In the best run, correcting L1 with measured values *lowers* accuracy
   (36.5% → 23.4%), because the model packs extra information into its L1 outputs.

---

## 11. Working preferences learned in this chat

- The user prefers simple language in explanations and speeches.
- The user wants honest reporting, including negative results.
- The user said results quality is secondary to a complete, working pipeline (deadline-driven).
- The user does not want long training runs started without asking: they stopped a full re-run once.
- The user wants presentable outputs for the professor (PDF report, presentation script, speeches) without claiming
  unfinished work, such as HITL.
