# Ablation (tag `dens12`)


Same splits/seed/hyperparameters for every variant. DENS = test macro-F1 (ensemble). Cross-corpus = valence/arousal accuracy via the emotion head (comparable across all variants, incl. flat).

| variant | DENS test macro-F1 | deap V acc | deap A acc | dreamer V acc | dreamer A acc |
|---|---|---|---|---|---|
| flat | 0.100 | 0.470 | 0.584 | 0.503 | 0.625 |
| level1_only | 0.127 | 0.472 | 0.584 | 0.517 | 0.623 |
| level12 | 0.117 | 0.470 | 0.584 | 0.510 | 0.623 |
| level123 | 0.135 | 0.457 | 0.584 | 0.507 | 0.623 |
| level123_l2 | 0.135 | 0.459 | 0.584 | 0.509 | 0.623 |
| bypass | 0.104 | 0.484 | 0.584 | 0.511 | 0.625 |

`level123` is the main model; `level123_l2` (emotion head also sees L2) and `bypass` (emotion head also sees the embedding trunk) are labelled variants, not the main model.
