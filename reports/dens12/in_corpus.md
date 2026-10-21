# In-corpus DENS results (tag `dens12`)


Held-out test group: 8 subjects, 384 non-rejected windows. 6-class coarse emotion (happy, surprise, fear, disgust, anger, sadness). Ensemble = mean of the 5 fold models' probabilities; CI = subject bootstrap.

| model | test acc (ensemble, 95% CI) | test macro-F1 (ensemble, 95% CI) | test macro-F1 (fold mean +- std) | CV val macro-F1 | params |
|---|---|---|---|---|---|
| flat | 0.125 [0.094, 0.168] | 0.100 [0.071, 0.120] | 0.120 +- 0.013 | 0.207 | 263,046 |
| level1_only | 0.164 [0.129, 0.208] | 0.127 [0.096, 0.146] | 0.125 +- 0.015 | 0.207 | 568,945 |
| level12 | 0.133 [0.086, 0.180] | 0.117 [0.072, 0.147] | 0.132 +- 0.031 | 0.213 | 567,857 |
| level123 | 0.151 [0.096, 0.222] | 0.135 [0.069, 0.180] | 0.122 +- 0.015 | 0.211 | 567,857 |
| level123_l2 | 0.164 [0.127, 0.200] | 0.135 [0.096, 0.158] | 0.128 +- 0.015 | 0.191 | 567,953 |
| bypass | 0.135 [0.098, 0.185] | 0.104 [0.070, 0.131] | 0.124 +- 0.026 | 0.212 | 571,953 |

Chance macro-F1 ~ 0.167; majority-class accuracy 0.281.

**H1 (non-inferiority, +-2 pp):** CBM(level123) - flat macro-F1 = +0.034 [-0.023, +0.090], p = 0.332. Non-inferiority NOT shown (CI lower bound <= -0.02).

## Concept accuracy on DENS test (CBM variants, ensemble)

| model | L1 mean Pearson r (37 concepts) | L2 valence bin acc | L2 arousal bin acc | L2 dominance bin acc | L2 valence acc, reliable raters only | L3 MAE (pseudo) |
|---|---|---|---|---|---|---|
| level1_only | 0.472 | 0.674 | 0.639 | 0.424 | 0.674 | 0.472 |
| level12 | 0.479 | 0.637 | 0.495 | 0.397 | 0.637 | 0.464 |
| level123 | 0.474 | 0.653 | 0.584 | 0.424 | 0.653 | 0.490 |
| level123_l2 | 0.476 | 0.661 | 0.511 | 0.471 | 0.661 | 0.484 |
| bypass | 0.480 | 0.674 | 0.645 | 0.413 | 0.674 | 0.456 |

Unreliable raters in the test group (D11, FACED only; V/A loss masked in training; included above unless stated): none. Dominance: real self-rating.
