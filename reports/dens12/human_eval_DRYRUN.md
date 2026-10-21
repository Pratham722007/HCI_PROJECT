# Human evaluation (tag `dens12`) - DRY RUN, SYNTHETIC RATERS

3 raters, 126 items. A/B balance: 0.60 of items had CBM as A.

| criterion | CBM mean [95% CI] | flat mean [95% CI] | Wilcoxon p | FDR q | rank-biserial r | Krippendorff alpha | Fleiss kappa |
|---|---|---|---|---|---|---|---|
| faithfulness | 3.41 [3.32, 3.51] | 3.03 [2.94, 3.13] | 0.0000 | 0.0000 | +0.50 | 0.08 | 0.07 |
| plausibility | 3.32 [3.24, 3.39] | 2.96 [2.87, 3.05] | 0.0000 | 0.0000 | +0.56 | 0.01 | -0.00 |
| clarity | 3.33 [3.24, 3.41] | 2.97 [2.87, 3.06] | 0.0000 | 0.0000 | +0.57 | -0.01 | 0.00 |
| actionability | 3.43 [3.33, 3.51] | 3.01 [2.92, 3.09] | 0.0000 | 0.0000 | +0.59 | 0.05 | -0.01 |

## By CBM prediction correctness (mean over raters and items)

| criterion | CBM correct: CBM / flat | CBM wrong: CBM / flat |
|---|---|---|
| faithfulness | 3.58 / 2.79 | 3.43 / 3.06 |
| plausibility | 3.29 / 3.21 | 3.36 / 2.94 |
| clarity | 3.42 / 2.58 | 3.34 / 3.03 |
| actionability | 3.21 / 2.88 | 3.44 / 3.03 |