# Human evaluation (tag `12ch_random`) - DRY RUN, SYNTHETIC RATERS

3 raters, 80 items. A/B balance: 0.59 of items had CBM as A.

| criterion | CBM mean [95% CI] | flat mean [95% CI] | Wilcoxon p | FDR q | rank-biserial r | Krippendorff alpha | Fleiss kappa |
|---|---|---|---|---|---|---|---|
| faithfulness | 3.36 [3.24, 3.47] | 3.05 [2.92, 3.16] | 0.0028 | 0.0029 | +0.41 | 0.05 | -0.02 |
| plausibility | 3.27 [3.17, 3.39] | 2.98 [2.86, 3.10] | 0.0014 | 0.0028 | +0.45 | 0.04 | 0.04 |
| clarity | 3.36 [3.25, 3.47] | 3.00 [2.87, 3.12] | 0.0003 | 0.0013 | +0.51 | 0.04 | 0.00 |
| actionability | 3.36 [3.24, 3.48] | 3.10 [2.97, 3.22] | 0.0029 | 0.0029 | +0.42 | 0.04 | 0.02 |

## By CBM prediction correctness (mean over raters and items)

| criterion | CBM correct: CBM / flat | CBM wrong: CBM / flat |
|---|---|---|
| faithfulness | 3.24 / 3.18 | 3.51 / 2.87 |
| plausibility | 3.32 / 2.97 | 3.39 / 3.00 |
| clarity | 3.33 / 2.89 | 3.40 / 3.04 |
| actionability | 3.46 / 3.09 | 3.23 / 3.12 |