# Feature check: is there emotion signal in the 12-channel EEG?

Same subject-wise 5-fold logistic probe on every feature set; train/val subjects only (test group untouched). Macro-F1 mean +- std over folds; class-balanced weights.

## DENS (32 train/val subjects, 1516 windows)

| features | dims | emotion (chance 0.167) | valence (chance 0.500) | arousal (chance 0.500) |
|---|---|---|---|---|
| eegpt | 2048 | 0.180 +- 0.024 | 0.524 +- 0.051 | 0.476 +- 0.024 |
| band37 | 37 | 0.193 +- 0.019 | 0.562 +- 0.066 | 0.454 +- 0.033 |
| spectral | 116 | 0.197 +- 0.023 | 0.542 +- 0.037 | 0.475 +- 0.022 |
| eegpt+spectral | 2164 | 0.181 +- 0.010 | 0.518 +- 0.046 | 0.467 +- 0.030 |

(44s)

## FACED (95 train/val subjects, 18511 windows)

| features | dims | emotion (chance 0.143) | valence (chance 0.500) | arousal (chance 0.500) |
|---|---|---|---|---|
| eegpt | 2048 | 0.188 +- 0.011 | 0.537 +- 0.012 | 0.528 +- 0.027 |
| band37 | 37 | 0.229 +- 0.014 | 0.535 +- 0.012 | 0.532 +- 0.012 |
| spectral | 116 | 0.229 +- 0.014 | 0.534 +- 0.010 | 0.532 +- 0.012 |
| eegpt+spectral | 2164 | 0.225 +- 0.012 | 0.540 +- 0.014 | 0.541 +- 0.026 |

(331s)

If hand-made spectral features beat EEGPT, the signal exists in the EEG and the frozen 12-channel EEGPT embedding is the bottleneck; if nothing beats chance, the labels/windows themselves carry little decodable signal.
