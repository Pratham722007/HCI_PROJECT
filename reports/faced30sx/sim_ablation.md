# Ablation (tag `faced30sx`)


Same splits/seed/hyperparameters for every variant. FACED = test macro-F1 (ensemble). Cross-corpus = valence/arousal accuracy via the emotion head (comparable across all variants, incl. flat).

| variant | FACED test macro-F1 | deap V acc | deap A acc | dens V acc | dens A acc | dreamer V acc | dreamer A acc |
|---|---|---|---|---|---|---|---|
| flat | 0.331 | 0.505 | 0.532 | 0.499 | 0.597 | 0.491 | 0.612 |
| level1_only | 0.322 | 0.504 | 0.527 | 0.538 | 0.582 | 0.492 | 0.591 |
| level12 | 0.312 | 0.496 | 0.542 | 0.569 | 0.592 | 0.489 | 0.591 |
| level123 | 0.295 | 0.494 | 0.542 | 0.551 | 0.586 | 0.500 | 0.592 |
| level123_l2 | 0.303 | 0.511 | 0.544 | 0.548 | 0.585 | 0.493 | 0.600 |
| bypass | 0.319 | 0.502 | 0.530 | 0.566 | 0.570 | 0.493 | 0.587 |

`level123` is the main model; `level123_l2` (emotion head also sees L2) and `bypass` (emotion head also sees the embedding trunk) are labelled variants, not the main model.
