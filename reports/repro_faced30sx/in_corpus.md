# In-corpus FACED results (tag `repro_faced30sx`)


Held-out test group: 24 subjects, 4683 non-rejected windows. 7-class coarse emotion (happy, tenderness, anger, fear, disgust, sadness, neutral). Ensemble = mean of the 5 fold models' probabilities; CI = subject bootstrap.

| model | test acc (ensemble, 95% CI) | test macro-F1 (ensemble, 95% CI) | test macro-F1 (fold mean +- std) | CV val macro-F1 | params |
|---|---|---|---|---|---|
| flat | 0.379 [0.356, 0.401] | 0.331 [0.304, 0.358] | 0.308 +- 0.011 | 0.320 | 300,551 |
| level123 | 0.357 [0.333, 0.383] | 0.295 [0.268, 0.324] | 0.264 +- 0.010 | 0.282 | 642,642 |

Chance macro-F1 ~ 0.143; majority-class accuracy 0.322.

**H1 (non-inferiority, +-2 pp):** CBM(level123) - flat macro-F1 = -0.036 [-0.055, -0.016], p = 0.000. Non-inferiority NOT shown (CI lower bound <= -0.02).

## Balanced accuracy and clip-level results (FACED test)

Clip level = mean of window probabilities within each (subject, clip): 672 clips. Balanced accuracy = mean per-class recall (chance = 1/classes); always predicting the majority class scores accuracy 0.322 but balanced accuracy 0.143.

| model | window balanced acc | clip acc [95% CI] | clip macro-F1 [95% CI] | clip balanced acc |
|---|---|---|---|---|
| flat | 0.324 | 0.427 [0.391, 0.460] | 0.352 [0.311, 0.390] | 0.346 |
| level123 | 0.303 | 0.435 [0.400, 0.470] | 0.345 [0.303, 0.390] | 0.355 |

## Concept accuracy on FACED test (CBM variants, ensemble)

| model | L1 mean Pearson r (37 concepts) | L2 valence bin acc | L2 arousal bin acc | L2 dominance bin acc | L2 valence acc, reliable raters only | L3 MAE (pseudo) |
|---|---|---|---|---|---|---|
| level123 | 0.648 | 0.548 | 0.584 | 0.662 | 0.565 | 0.442 |

Unreliable raters in the test group (D11, FACED only; V/A loss masked in training; included above unless stated): ['sub077', 'sub095', 'sub096', 'sub108']. Dominance: FACED pseudo-label (D3).
