# In-corpus FACED results (tag `12ch_random`)

> **PLACEHOLDER RUN: EEGPT weights were NOT loaded (random-init encoder).** Numbers below only show the pipeline runs end to end; they are not research results. Re-run `scripts/run_all.sh` with the real checkpoint.


Held-out test group: 24 subjects, 4683 non-rejected windows. 7-class coarse emotion. Ensemble = mean of the 5 fold models' probabilities; CI = subject bootstrap.

| model | test acc (ensemble, 95% CI) | test macro-F1 (ensemble, 95% CI) | test macro-F1 (fold mean +- std) | CV val macro-F1 | params |
|---|---|---|---|---|---|
| flat | 0.205 [0.189, 0.223] | 0.198 [0.180, 0.216] | 0.185 +- 0.005 | 0.187 | 263,175 |
| level1_only | 0.195 [0.180, 0.210] | 0.188 [0.173, 0.202] | 0.173 +- 0.007 | 0.180 | 568,978 |
| level12 | 0.212 [0.192, 0.232] | 0.191 [0.173, 0.208] | 0.177 +- 0.008 | 0.177 | 567,890 |
| level123 | 0.193 [0.172, 0.214] | 0.180 [0.160, 0.198] | 0.156 +- 0.010 | 0.160 | 567,890 |
| level123_l2 | 0.201 [0.180, 0.222] | 0.185 [0.166, 0.201] | 0.172 +- 0.004 | 0.180 | 567,986 |
| bypass | 0.186 [0.169, 0.202] | 0.181 [0.162, 0.197] | 0.174 +- 0.009 | 0.186 | 571,986 |

Chance macro-F1 ~ 0.143; majority-class accuracy 0.322.

**H1 (non-inferiority, +-2 pp):** CBM(level123) - flat macro-F1 = -0.018 [-0.039, +0.002], p = 0.078. Non-inferiority NOT shown (CI lower bound <= -0.02).

## Concept accuracy on FACED test (CBM variants, ensemble)

| model | L1 mean Pearson r (37 concepts) | L2 valence bin acc | L2 arousal bin acc | L2 valence acc, reliable raters only | L3 MAE (pseudo) |
|---|---|---|---|---|---|
| level1_only | 0.615 | 0.513 | 0.519 | 0.529 | 0.488 |
| level12 | 0.630 | 0.521 | 0.529 | 0.531 | 0.490 |
| level123 | 0.634 | 0.517 | 0.530 | 0.537 | 0.499 |
| level123_l2 | 0.631 | 0.505 | 0.528 | 0.519 | 0.493 |
| bypass | 0.626 | 0.505 | 0.553 | 0.522 | 0.488 |

Unreliable raters in the test group (D11, V/A loss masked in training; included above unless stated): ['sub077', 'sub095', 'sub096', 'sub108'].
