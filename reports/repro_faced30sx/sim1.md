# Simulation 1: zero-shot cross-corpus generalisation (tag `repro_faced30sx`)


Models trained on FACED only; temperature/thresholds frozen; no DEAP/DREAMER data used for any choice. Binary threshold = scale midpoint (0.5 after rescaling), ties dropped. CIs resample subjects.

**Metrics where the CBM is NOT significantly better than the flat baseline (H2):** deap valence accuracy (CBM L2 concept: +0.003, CI [-0.008, +0.014]); deap valence accuracy (CBM via emotion: -0.011, CI [-0.019, -0.003]); dens valence accuracy (CBM L2 concept: +0.017, CI [-0.020, +0.056]); dens arousal accuracy (CBM L2 concept: +0.000, CI [-0.037, +0.038]); dens arousal accuracy (CBM via emotion: -0.011, CI [-0.044, +0.019]); dreamer valence accuracy (CBM L2 concept: +0.013, CI [-0.015, +0.041]); dreamer arousal accuracy (CBM L2 concept: -0.014, CI [-0.037, +0.010]); dreamer valence accuracy (CBM via emotion: +0.008, CI [-0.015, +0.035]); dreamer arousal accuracy (CBM via emotion: -0.020, CI [-0.055, +0.007])


## DEAP (zero-shot, 32 subjects, 19133 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.505 [0.484, 0.526] | 0.498 | 0.532 [0.504, 0.557] | 0.497 | 0.060 [0.020, 0.100] | 0.010 [-0.032, 0.052] |
| level123 | L2 concept | 0.508 [0.491, 0.526] | 0.507 | 0.549 [0.519, 0.579] | 0.499 | 0.054 [0.016, 0.096] | 0.037 [0.005, 0.069] |
| level123 | via emotion | 0.494 [0.475, 0.513] | 0.487 | 0.542 [0.514, 0.568] | 0.500 | 0.048 [0.010, 0.089] | 0.027 [-0.008, 0.064] |

DEAP 4-quadrant accuracy (ties at 5 excluded; chance ~ 0.25):

| model | readout | quadrant acc [95% CI] |
|---|---|---|
| flat | via emotion | 0.266 [0.247, 0.286] |
| level123 | L2 concept | 0.273 [0.251, 0.296] |
| level123 | via emotion | 0.259 [0.239, 0.280] |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.003 [-0.008, +0.014] | 0.582 |
| CBM L2 concept vs flat | arousal | +0.017 [+0.005, +0.030] | 0.013 |
| CBM via emotion vs flat | valence | -0.011 [-0.019, -0.003] | 0.009 |
| CBM via emotion vs flat | arousal | +0.009 [+0.000, +0.018] | 0.044 |

Predicted dominance (trained on FACED pseudo-labels, exploratory (D3)) vs real DEAP dominance, Spearman 0.028 [-0.008, 0.067].


## DENS (zero-shot, 40 subjects, 1900 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.499 [0.472, 0.526] | 0.467 | 0.597 [0.561, 0.632] | 0.507 | -0.056 [-0.125, 0.004] | 0.083 [0.016, 0.158] |
| level123 | L2 concept | 0.516 [0.478, 0.557] | 0.477 | 0.597 [0.553, 0.637] | 0.487 | -0.053 [-0.111, 0.006] | 0.047 [-0.023, 0.114] |
| level123 | via emotion | 0.551 [0.514, 0.592] | 0.491 | 0.586 [0.549, 0.618] | 0.483 | -0.053 [-0.115, 0.013] | 0.031 [-0.031, 0.088] |

DENS coarse-emotion accuracy (**secondary**: labels mapped by class name onto the 7 training classes, unmapped excluded; chance ~ 0.200):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.211 [0.176, 0.246] | 0.147 |
| level123 | 0.192 [0.164, 0.218] | 0.154 |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.017 [-0.020, +0.056] | 0.411 |
| CBM L2 concept vs flat | arousal | +0.000 [-0.037, +0.038] | 1.000 |
| CBM via emotion vs flat | valence | +0.052 [+0.016, +0.088] | 0.004 |
| CBM via emotion vs flat | arousal | -0.011 [-0.044, +0.019] | 0.459 |

Predicted dominance (trained on FACED pseudo-labels, exploratory (D3)) vs real DENS dominance, Spearman -0.013 [-0.066, 0.042].


## DREAMER (zero-shot, 23 subjects, 5817 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.491 [0.442, 0.536] | 0.484 | 0.612 [0.566, 0.661] | 0.576 | -0.032 [-0.108, 0.035] | 0.114 [0.040, 0.195] |
| level123 | L2 concept | 0.504 [0.460, 0.543] | 0.497 | 0.598 [0.562, 0.637] | 0.541 | -0.012 [-0.083, 0.049] | 0.081 [0.007, 0.155] |
| level123 | via emotion | 0.500 [0.454, 0.542] | 0.482 | 0.592 [0.558, 0.632] | 0.539 | -0.017 [-0.084, 0.042] | 0.072 [-0.002, 0.152] |

DREAMER coarse-emotion accuracy (**secondary**: labels mapped by class name onto the 7 training classes, unmapped excluded; chance ~ 0.167):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.249 [0.229, 0.269] | 0.184 |
| level123 | 0.242 [0.220, 0.263] | 0.177 |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.013 [-0.015, +0.041] | 0.375 |
| CBM L2 concept vs flat | arousal | -0.014 [-0.037, +0.010] | 0.238 |
| CBM via emotion vs flat | valence | +0.008 [-0.015, +0.035] | 0.564 |
| CBM via emotion vs flat | arousal | -0.020 [-0.055, +0.007] | 0.191 |

Predicted dominance (trained on FACED pseudo-labels, exploratory (D3)) vs real DREAMER dominance, Spearman 0.013 [-0.068, 0.100].
