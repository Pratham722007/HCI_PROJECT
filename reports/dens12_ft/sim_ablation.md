# Ablation (tag `dens12_ft`)


Same splits/seed/hyperparameters for every variant. DENS = test macro-F1 (ensemble). Cross-corpus = valence/arousal accuracy via the emotion head (comparable across all variants, incl. flat).

| variant | DENS test macro-F1 | deap V acc | deap A acc | dreamer V acc | dreamer A acc |
|---|---|---|---|---|---|
| flat | 0.116 | 0.479 | 0.584 | 0.500 | 0.625 |
| level123 | 0.151 | 0.455 | 0.584 | 0.507 | 0.623 |

`level123` is the main model; `level123_l2` (emotion head also sees L2) and `bypass` (emotion head also sees the embedding trunk) are labelled variants, not the main model.
