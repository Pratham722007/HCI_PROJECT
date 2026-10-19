# Ablation (tag `12ch_random`)

> **PLACEHOLDER RUN: EEGPT weights were NOT loaded (random-init encoder).** Numbers below only show the pipeline runs end to end; they are not research results. Re-run `scripts/run_all.sh` with the real checkpoint.


Same splits/seed/hyperparameters for every variant. FACED = test macro-F1 (ensemble). Cross-corpus = valence/arousal accuracy via the emotion head (comparable across all variants, incl. flat).

| variant | FACED test macro-F1 | deap V acc | deap A acc | dreamer V acc | dreamer A acc |
|---|---|---|---|---|---|
| flat | 0.198 | 0.441 | 0.539 | 0.493 | 0.539 |
| level1_only | 0.188 | 0.443 | 0.535 | 0.498 | 0.539 |
| level12 | 0.191 | 0.444 | 0.554 | 0.493 | 0.545 |
| level123 | 0.180 | 0.444 | 0.535 | 0.490 | 0.510 |
| level123_l2 | 0.185 | 0.441 | 0.562 | 0.494 | 0.562 |
| bypass | 0.181 | 0.441 | 0.574 | 0.496 | 0.561 |

`level123` is the main model; `level123_l2` (emotion head also sees L2) and `bypass` (emotion head also sees the embedding trunk) are labelled variants, not the main model.
