# In-corpus FACED results (tag `faced12`)


Held-out test group: 24 subjects, 4683 non-rejected windows. 7-class coarse emotion (happy, tenderness, anger, fear, disgust, sadness, neutral). Ensemble = mean of the 5 fold models' probabilities; CI = subject bootstrap.

| model | test acc (ensemble, 95% CI) | test macro-F1 (ensemble, 95% CI) | test macro-F1 (fold mean +- std) | CV val macro-F1 | params |
|---|---|---|---|---|---|
| flat | 0.252 [0.230, 0.269] | 0.227 [0.208, 0.243] | 0.211 +- 0.005 | 0.210 | 263,175 |
| level1_only | 0.270 [0.245, 0.295] | 0.222 [0.199, 0.245] | 0.197 +- 0.003 | 0.200 | 568,978 |
| level12 | 0.244 [0.220, 0.265] | 0.218 [0.195, 0.238] | 0.190 +- 0.003 | 0.193 | 567,890 |
| level123 | 0.216 [0.194, 0.238] | 0.197 [0.178, 0.216] | 0.178 +- 0.004 | 0.184 | 567,890 |
| level123_l2 | 0.220 [0.200, 0.243] | 0.208 [0.186, 0.230] | 0.182 +- 0.004 | 0.186 | 567,986 |
| bypass | 0.266 [0.240, 0.287] | 0.224 [0.202, 0.244] | 0.205 +- 0.006 | 0.203 | 571,986 |

Chance macro-F1 ~ 0.143; majority-class accuracy 0.322.

**H1 (non-inferiority, +-2 pp):** CBM(level123) - flat macro-F1 = -0.030 [-0.042, -0.018], p = 0.000. Non-inferiority NOT shown (CI lower bound <= -0.02).

## Balanced accuracy and clip-level results (FACED test)

Clip level = mean of window probabilities within each (subject, clip): 672 clips. Balanced accuracy = mean per-class recall (chance = 1/classes); always predicting the majority class scores accuracy 0.322 but balanced accuracy 0.143.

| model | window balanced acc | clip acc [95% CI] | clip macro-F1 [95% CI] | clip balanced acc |
|---|---|---|---|---|
| flat | 0.236 | 0.353 [0.301, 0.394] | 0.295 [0.252, 0.336] | 0.306 |
| level1_only | 0.224 | 0.336 [0.300, 0.369] | 0.252 [0.218, 0.281] | 0.254 |
| level12 | 0.222 | 0.295 [0.256, 0.333] | 0.248 [0.209, 0.287] | 0.250 |
| level123 | 0.202 | 0.265 [0.222, 0.308] | 0.233 [0.191, 0.275] | 0.240 |
| level123_l2 | 0.215 | 0.262 [0.231, 0.295] | 0.237 [0.202, 0.266] | 0.250 |
| bypass | 0.227 | 0.332 [0.284, 0.376] | 0.247 [0.204, 0.289] | 0.254 |

## Concept accuracy on FACED test (CBM variants, ensemble)

| model | L1 mean Pearson r (37 concepts) | L2 valence bin acc | L2 arousal bin acc | L2 dominance bin acc | L2 valence acc, reliable raters only | L3 MAE (pseudo) |
|---|---|---|---|---|---|---|
| level1_only | 0.617 | 0.523 | 0.536 | 0.591 | 0.530 | 0.483 |
| level12 | 0.639 | 0.523 | 0.514 | 0.534 | 0.533 | 0.478 |
| level123 | 0.642 | 0.525 | 0.524 | 0.605 | 0.540 | 0.485 |
| level123_l2 | 0.639 | 0.524 | 0.536 | 0.655 | 0.533 | 0.481 |
| bypass | 0.662 | 0.533 | 0.538 | 0.609 | 0.538 | 0.480 |

Unreliable raters in the test group (D11, FACED only; V/A loss masked in training; included above unless stated): ['sub077', 'sub095', 'sub096', 'sub108']. Dominance: FACED pseudo-label (D3).
