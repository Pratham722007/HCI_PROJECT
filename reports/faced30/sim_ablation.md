# Ablation (tag `faced30`)


Same splits/seed/hyperparameters for every variant. FACED = test macro-F1 (ensemble). Cross-corpus = valence/arousal accuracy via the emotion head (comparable across all variants, incl. flat).

| variant | FACED test macro-F1 | deap V acc | deap A acc | dens V acc | dens A acc | dreamer V acc | dreamer A acc |
|---|---|---|---|---|---|---|---|
| flat | 0.266 | 0.459 | 0.529 | 0.639 | 0.569 | 0.515 | 0.590 |
| level1_only | 0.261 | 0.483 | 0.523 | 0.596 | 0.568 | 0.523 | 0.587 |
| level12 | 0.241 | 0.466 | 0.544 | 0.644 | 0.656 | 0.524 | 0.597 |
| level123 | 0.228 | 0.453 | 0.537 | 0.680 | 0.650 | 0.503 | 0.587 |
| level123_l2 | 0.235 | 0.454 | 0.548 | 0.663 | 0.666 | 0.502 | 0.594 |
| bypass | 0.271 | 0.478 | 0.524 | 0.594 | 0.574 | 0.530 | 0.588 |

`level123` is the main model; `level123_l2` (emotion head also sees L2) and `bypass` (emotion head also sees the embedding trunk) are labelled variants, not the main model.
