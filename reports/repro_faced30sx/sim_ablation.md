# Ablation (tag `repro_faced30sx`)


Same splits/seed/hyperparameters for every variant. FACED = test macro-F1 (ensemble). Cross-corpus = valence/arousal accuracy via the emotion head (comparable across all variants, incl. flat).

| variant | FACED test macro-F1 | deap V acc | deap A acc | dens V acc | dens A acc | dreamer V acc | dreamer A acc |
|---|---|---|---|---|---|---|---|
| flat | 0.331 | 0.505 | 0.532 | 0.499 | 0.597 | 0.491 | 0.612 |
| level123 | 0.295 | 0.494 | 0.542 | 0.551 | 0.586 | 0.500 | 0.592 |

`level123` is the main model; `level123_l2` (emotion head also sees L2) and `bypass` (emotion head also sees the embedding trunk) are labelled variants, not the main model.
