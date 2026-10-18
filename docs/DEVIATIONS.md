# Protocol decisions and deviations log

All entries dated 16 Oct 2026 and made **before any model training or embedding extraction**.

| ID | Decision | Reason / evidence |
|----|----------|-------------------|
| D1 | Exclude FACED subjects sub004, sub023, sub030, sub056 | Amplitude artifacts surviving ICA (std 527 / 26,010 / 5,271 / 526 µV vs median 9 µV; up to ±330 mV). `reports/faced_inspection.md`. sub003 (368 µV) and sub064 (169 µV) are kept but rely on window-level rejection; revisit only if the rejection rate is extreme (>30% windows). |
| D2 | DEAP: all 32 subjects (originally 16 on disk; remaining 16 added 16 Oct 2026). Labels from `participant_ratings.xls` matched on Experiment_id, never from the `.dat` files | `.dat` labels corrupted on 439/1280 trials (all 32 subjects): V/A reflected to 9-x, giving impossible 0 ratings and zero HVHA trials. |
| D3 | Dominance: category-prototype pseudo-label, masked loss, weight <= 0.25, reported as exploratory | FACED has no dominance item. Validated only on DEAP/DREAMER real ratings. Not derived from valence/arousal (circular). |
| D4 | Binarisation: "high" = rating > midpoint (FACED 3.5 on 0-7; DEAP 5 on 1-9; DREAMER 3 on 1-5); exact ties dropped from binary metrics; Spearman reported alongside | Fixed before results; avoids threshold shopping. |
| D5 | Coarse 7-class set: happy (amusement, inspiration, joy), tenderness, anger, fear, disgust, sadness, neutral | Surprise absent in FACED. Class sizes in clips: happy 9, neutral 4, others 3 -> macro-F1 primary metric. |
| D6 | FACED clip-held-out check holds out 1 of 3 clips per category (not 1 of 4) | Stimuli_info.xlsx gives 3 clips/category (neutral 4). |
| D7 | FACED pipeline additions: 50 Hz notch + 4-45 Hz band-pass + re-CAR over the 12 shared channels + amplitude rejection | Processed data has no notch, 0.05-47 Hz filter, 32-ch CAR (`Code/Preprocessing/main.py`). |

Open items: Level-3 appraisal table (Phase 4, needs sign-off); DREAMER clip->emotion table not on disk; EEGPT checkpoint and evaluator recruitment (Phase 0).

| D8 | Artifact rule is subject-relative: reject window if peak > max(5 x subject median window peak, 100 uV), identical for all datasets; raw peak stored per window (`labels/peak_uv`) | A fixed 150 uV rule rejected 42% of DEAP windows (s09 99.5%, s11 100%) because typical raw window peak varies 20-860 uV across DEAP subjects. Decided after seeing preprocessing QC only, before any embedding/training. Rejected windows are flagged (`reject`), not dropped, so rows stay aligned. |
| D9 | No post-hoc subject exclusion on DEAP/DREAMER; noisy subjects (see `reports/qc.md`) are flagged in `reports/qc.md` for sensitivity analysis | Zero-shot corpora must not be filtered using outcomes; amplitude-only flags are reported instead. |
| D10 | FACED 30 s -> 7 windows; the first 2 s of each clip are discarded (filter edge effects, mood still building) | Remainder must go somewhere; keeps the end of the clip where the emotion is strongest. |
