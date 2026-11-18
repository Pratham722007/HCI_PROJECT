# In-corpus DENS results (tag `dens12_ft`)


Held-out test group: 8 subjects, 384 non-rejected windows. 6-class coarse emotion (happy, surprise, fear, disgust, anger, sadness). Ensemble = mean of the 5 fold models' probabilities; CI = subject bootstrap.

| model | test acc (ensemble, 95% CI) | test macro-F1 (ensemble, 95% CI) | test macro-F1 (fold mean +- std) | CV val macro-F1 | params |
|---|---|---|---|---|---|
| flat | 0.151 [0.114, 0.205] | 0.116 [0.084, 0.135] | 0.123 +- 0.012 | 0.198 | 6,568,838 |
| level123 | 0.177 [0.111, 0.238] | 0.151 [0.071, 0.199] | 0.129 +- 0.013 | 0.192 | 6,873,649 |

Chance macro-F1 ~ 0.167; majority-class accuracy 0.281.

**H1 (non-inferiority, +-2 pp):** CBM(level123) - flat macro-F1 = +0.035 [-0.038, +0.086], p = 0.399. Non-inferiority NOT shown (CI lower bound <= -0.02).

## Concept accuracy on DENS test (CBM variants, ensemble)

| model | L1 mean Pearson r (37 concepts) | L2 valence bin acc | L2 arousal bin acc | L2 dominance bin acc | L2 valence acc, reliable raters only | L3 MAE (pseudo) |
|---|---|---|---|---|---|---|
| level123 | 0.488 | 0.655 | 0.587 | 0.397 | 0.655 | 0.486 |

Unreliable raters in the test group (D11, FACED only; V/A loss masked in training; included above unless stated): none. Dominance: real self-rating.
