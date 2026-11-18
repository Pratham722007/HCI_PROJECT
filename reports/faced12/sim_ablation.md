# Ablation (tag `faced12`)


Same splits/seed/hyperparameters for every variant. FACED = test macro-F1 (ensemble). Cross-corpus = valence/arousal accuracy via the emotion head (comparable across all variants, incl. flat).

| variant | FACED test macro-F1 | deap V acc | deap A acc | dens V acc | dens A acc | dreamer V acc | dreamer A acc |
|---|---|---|---|---|---|---|---|
| flat | 0.227 | 0.451 | 0.513 | 0.601 | 0.518 | 0.500 | 0.590 |
| level1_only | 0.222 | 0.459 | 0.527 | 0.567 | 0.569 | 0.493 | 0.568 |
| level12 | 0.218 | 0.449 | 0.529 | 0.616 | 0.672 | 0.496 | 0.584 |
| level123 | 0.197 | 0.449 | 0.539 | 0.631 | 0.678 | 0.495 | 0.586 |
| level123_l2 | 0.208 | 0.443 | 0.559 | 0.644 | 0.675 | 0.496 | 0.595 |
| bypass | 0.224 | 0.461 | 0.492 | 0.537 | 0.561 | 0.504 | 0.564 |

`level123` is the main model; `level123_l2` (emotion head also sees L2) and `bypass` (emotion head also sees the embedding trunk) are labelled variants, not the main model.
