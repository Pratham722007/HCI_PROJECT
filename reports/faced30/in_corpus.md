# In-corpus FACED results (tag `faced30`)


Held-out test group: 24 subjects, 4683 non-rejected windows. 7-class coarse emotion (happy, tenderness, anger, fear, disgust, sadness, neutral). Ensemble = mean of the 5 fold models' probabilities; CI = subject bootstrap.

| model | test acc (ensemble, 95% CI) | test macro-F1 (ensemble, 95% CI) | test macro-F1 (fold mean +- std) | CV val macro-F1 | params |
|---|---|---|---|---|---|
| flat | 0.305 [0.288, 0.321] | 0.266 [0.245, 0.286] | 0.244 +- 0.004 | 0.251 | 263,175 |
| level1_only | 0.330 [0.311, 0.346] | 0.261 [0.237, 0.282] | 0.233 +- 0.005 | 0.238 | 568,978 |
| level12 | 0.286 [0.266, 0.307] | 0.241 [0.219, 0.263] | 0.220 +- 0.008 | 0.226 | 567,890 |
| level123 | 0.241 [0.223, 0.260] | 0.228 [0.209, 0.248] | 0.204 +- 0.003 | 0.213 | 567,890 |
| level123_l2 | 0.257 [0.243, 0.273] | 0.235 [0.218, 0.252] | 0.209 +- 0.007 | 0.216 | 567,986 |
| bypass | 0.323 [0.302, 0.343] | 0.271 [0.246, 0.294] | 0.239 +- 0.005 | 0.245 | 571,986 |

Chance macro-F1 ~ 0.143; majority-class accuracy 0.322.

**H1 (non-inferiority, +-2 pp):** CBM(level123) - flat macro-F1 = -0.038 [-0.052, -0.022], p = 0.000. Non-inferiority NOT shown (CI lower bound <= -0.02).

## Balanced accuracy and clip-level results (FACED test)

Clip level = mean of window probabilities within each (subject, clip): 672 clips. Balanced accuracy = mean per-class recall (chance = 1/classes); always predicting the majority class scores accuracy 0.322 but balanced accuracy 0.143.

| model | window balanced acc | clip acc [95% CI] | clip macro-F1 [95% CI] | clip balanced acc |
|---|---|---|---|---|
| flat | 0.264 | 0.430 [0.399, 0.463] | 0.339 [0.303, 0.376] | 0.334 |
| level1_only | 0.256 | 0.409 [0.379, 0.438] | 0.283 [0.243, 0.321] | 0.279 |
| level12 | 0.239 | 0.393 [0.359, 0.427] | 0.306 [0.264, 0.345] | 0.305 |
| level123 | 0.230 | 0.295 [0.259, 0.338] | 0.271 [0.232, 0.313] | 0.267 |
| level123_l2 | 0.234 | 0.339 [0.311, 0.368] | 0.287 [0.253, 0.316] | 0.279 |
| bypass | 0.267 | 0.426 [0.391, 0.464] | 0.321 [0.271, 0.369] | 0.311 |

## Concept accuracy on FACED test (CBM variants, ensemble)

| model | L1 mean Pearson r (37 concepts) | L2 valence bin acc | L2 arousal bin acc | L2 dominance bin acc | L2 valence acc, reliable raters only | L3 MAE (pseudo) |
|---|---|---|---|---|---|---|
| level1_only | 0.668 | 0.535 | 0.559 | 0.648 | 0.540 | 0.464 |
| level12 | 0.681 | 0.533 | 0.545 | 0.558 | 0.557 | 0.465 |
| level123 | 0.688 | 0.533 | 0.543 | 0.631 | 0.554 | 0.468 |
| level123_l2 | 0.687 | 0.541 | 0.563 | 0.669 | 0.556 | 0.465 |
| bypass | 0.716 | 0.526 | 0.563 | 0.654 | 0.529 | 0.468 |

Unreliable raters in the test group (D11, FACED only; V/A loss masked in training; included above unless stated): ['sub077', 'sub095', 'sub096', 'sub108']. Dominance: FACED pseudo-label (D3).
