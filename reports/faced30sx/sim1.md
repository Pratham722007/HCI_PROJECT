# Simulation 1: zero-shot cross-corpus generalisation (tag `faced30sx`)


Models trained on FACED only; temperature/thresholds frozen; no DEAP/DREAMER data used for any choice. Binary threshold = scale midpoint (0.5 after rescaling), ties dropped. CIs resample subjects.

**Metrics where the CBM is NOT significantly better than the flat baseline (H2):** deap valence accuracy (CBM L2 concept: +0.003, CI [-0.008, +0.014]); deap valence accuracy (CBM via emotion: -0.011, CI [-0.019, -0.003]); dens valence accuracy (CBM L2 concept: +0.017, CI [-0.020, +0.056]); dens arousal accuracy (CBM L2 concept: +0.000, CI [-0.037, +0.038]); dens arousal accuracy (CBM via emotion: -0.011, CI [-0.044, +0.019]); dreamer valence accuracy (CBM L2 concept: +0.013, CI [-0.015, +0.041]); dreamer arousal accuracy (CBM L2 concept: -0.014, CI [-0.037, +0.010]); dreamer valence accuracy (CBM via emotion: +0.008, CI [-0.015, +0.035]); dreamer arousal accuracy (CBM via emotion: -0.020, CI [-0.055, +0.007])


## DEAP (zero-shot, 32 subjects, 19133 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.505 [0.484, 0.526] | 0.498 | 0.532 [0.504, 0.557] | 0.497 | 0.060 [0.020, 0.100] | 0.010 [-0.032, 0.052] |
| level1_only | L2 concept | 0.524 [0.510, 0.538] | 0.522 | 0.551 [0.515, 0.586] | 0.487 | 0.058 [0.025, 0.097] | 0.025 [-0.014, 0.065] |
| level1_only | via emotion | 0.504 [0.489, 0.519] | 0.501 | 0.527 [0.499, 0.554] | 0.498 | 0.056 [0.023, 0.095] | 0.001 [-0.042, 0.045] |
| level12 | L2 concept | 0.498 [0.478, 0.517] | 0.488 | 0.521 [0.499, 0.543] | 0.498 | 0.055 [0.012, 0.102] | -0.003 [-0.037, 0.034] |
| level12 | via emotion | 0.496 [0.476, 0.515] | 0.485 | 0.542 [0.513, 0.568] | 0.500 | 0.052 [0.013, 0.093] | 0.007 [-0.030, 0.046] |
| level123 | L2 concept | 0.508 [0.491, 0.526] | 0.507 | 0.549 [0.519, 0.579] | 0.499 | 0.054 [0.016, 0.096] | 0.037 [0.005, 0.069] |
| level123 | via emotion | 0.494 [0.475, 0.513] | 0.487 | 0.542 [0.514, 0.568] | 0.500 | 0.048 [0.010, 0.089] | 0.027 [-0.008, 0.064] |
| level123_l2 | L2 concept | 0.513 [0.494, 0.534] | 0.512 | 0.536 [0.511, 0.561] | 0.503 | 0.060 [0.019, 0.108] | 0.021 [-0.011, 0.056] |
| level123_l2 | via emotion | 0.511 [0.493, 0.531] | 0.508 | 0.544 [0.515, 0.572] | 0.498 | 0.059 [0.018, 0.103] | 0.022 [-0.014, 0.060] |
| bypass | L2 concept | 0.520 [0.507, 0.534] | 0.520 | 0.541 [0.514, 0.567] | 0.504 | 0.060 [0.027, 0.096] | 0.034 [-0.004, 0.071] |
| bypass | via emotion | 0.502 [0.484, 0.519] | 0.495 | 0.530 [0.509, 0.550] | 0.505 | 0.057 [0.024, 0.093] | 0.010 [-0.028, 0.046] |

DEAP 4-quadrant accuracy (ties at 5 excluded; chance ~ 0.25):

| model | readout | quadrant acc [95% CI] |
|---|---|---|
| flat | via emotion | 0.266 [0.247, 0.286] |
| level1_only | L2 concept | 0.292 [0.268, 0.314] |
| level1_only | via emotion | 0.265 [0.248, 0.283] |
| level12 | L2 concept | 0.260 [0.242, 0.278] |
| level12 | via emotion | 0.263 [0.243, 0.284] |
| level123 | L2 concept | 0.273 [0.251, 0.296] |
| level123 | via emotion | 0.259 [0.239, 0.280] |
| level123_l2 | L2 concept | 0.275 [0.258, 0.294] |
| level123_l2 | via emotion | 0.276 [0.256, 0.297] |
| bypass | L2 concept | 0.281 [0.262, 0.298] |
| bypass | via emotion | 0.261 [0.245, 0.279] |

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
| level1_only | L2 concept | 0.479 [0.450, 0.508] | 0.463 | 0.647 [0.612, 0.681] | 0.532 | -0.027 [-0.088, 0.026] | 0.053 [-0.022, 0.130] |
| level1_only | via emotion | 0.538 [0.499, 0.576] | 0.482 | 0.582 [0.552, 0.612] | 0.510 | -0.034 [-0.096, 0.022] | 0.063 [-0.006, 0.139] |
| level12 | L2 concept | 0.571 [0.531, 0.608] | 0.507 | 0.549 [0.518, 0.582] | 0.492 | -0.035 [-0.098, 0.029] | 0.069 [-0.007, 0.142] |
| level12 | via emotion | 0.569 [0.530, 0.606] | 0.502 | 0.592 [0.556, 0.629] | 0.505 | -0.035 [-0.096, 0.029] | 0.084 [0.018, 0.152] |
| level123 | L2 concept | 0.516 [0.478, 0.557] | 0.477 | 0.597 [0.553, 0.637] | 0.487 | -0.053 [-0.111, 0.006] | 0.047 [-0.023, 0.114] |
| level123 | via emotion | 0.551 [0.514, 0.592] | 0.491 | 0.586 [0.549, 0.618] | 0.483 | -0.053 [-0.115, 0.013] | 0.031 [-0.031, 0.088] |
| level123_l2 | L2 concept | 0.533 [0.502, 0.564] | 0.491 | 0.574 [0.532, 0.615] | 0.504 | 0.020 [-0.040, 0.075] | 0.051 [-0.015, 0.119] |
| level123_l2 | via emotion | 0.548 [0.515, 0.579] | 0.498 | 0.585 [0.544, 0.626] | 0.497 | 0.018 [-0.038, 0.070] | 0.050 [-0.018, 0.117] |
| bypass | L2 concept | 0.513 [0.490, 0.537] | 0.493 | 0.611 [0.568, 0.650] | 0.517 | -0.019 [-0.073, 0.039] | 0.084 [0.007, 0.162] |
| bypass | via emotion | 0.566 [0.532, 0.600] | 0.516 | 0.570 [0.532, 0.605] | 0.490 | -0.006 [-0.060, 0.048] | 0.047 [-0.020, 0.122] |

DENS coarse-emotion accuracy (**secondary**: labels mapped by class name onto the 7 training classes, unmapped excluded; chance ~ 0.200):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.211 [0.176, 0.246] | 0.147 |
| level1_only | 0.202 [0.177, 0.228] | 0.155 |
| level12 | 0.189 [0.158, 0.220] | 0.136 |
| level123 | 0.192 [0.164, 0.218] | 0.154 |
| level123_l2 | 0.215 [0.183, 0.248] | 0.152 |
| bypass | 0.207 [0.181, 0.234] | 0.158 |

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
| level1_only | L2 concept | 0.486 [0.445, 0.523] | 0.485 | 0.604 [0.558, 0.648] | 0.537 | -0.024 [-0.095, 0.039] | 0.089 [0.021, 0.156] |
| level1_only | via emotion | 0.492 [0.445, 0.535] | 0.477 | 0.591 [0.553, 0.632] | 0.556 | -0.022 [-0.091, 0.040] | 0.080 [0.007, 0.151] |
| level12 | L2 concept | 0.490 [0.446, 0.528] | 0.471 | 0.583 [0.549, 0.622] | 0.555 | -0.041 [-0.116, 0.032] | 0.083 [0.004, 0.166] |
| level12 | via emotion | 0.489 [0.445, 0.531] | 0.469 | 0.591 [0.555, 0.632] | 0.546 | -0.043 [-0.108, 0.022] | 0.073 [-0.002, 0.153] |
| level123 | L2 concept | 0.504 [0.460, 0.543] | 0.497 | 0.598 [0.562, 0.637] | 0.541 | -0.012 [-0.083, 0.049] | 0.081 [0.007, 0.155] |
| level123 | via emotion | 0.500 [0.454, 0.542] | 0.482 | 0.592 [0.558, 0.632] | 0.539 | -0.017 [-0.084, 0.042] | 0.072 [-0.002, 0.152] |
| level123_l2 | L2 concept | 0.498 [0.456, 0.534] | 0.490 | 0.590 [0.552, 0.632] | 0.551 | -0.021 [-0.083, 0.041] | 0.092 [0.020, 0.171] |
| level123_l2 | via emotion | 0.493 [0.450, 0.531] | 0.481 | 0.600 [0.565, 0.637] | 0.545 | -0.021 [-0.082, 0.037] | 0.106 [0.038, 0.181] |
| bypass | L2 concept | 0.484 [0.448, 0.517] | 0.483 | 0.596 [0.558, 0.637] | 0.550 | -0.029 [-0.089, 0.027] | 0.088 [0.008, 0.165] |
| bypass | via emotion | 0.493 [0.456, 0.527] | 0.481 | 0.587 [0.545, 0.630] | 0.552 | -0.039 [-0.097, 0.011] | 0.081 [0.005, 0.157] |

DREAMER coarse-emotion accuracy (**secondary**: labels mapped by class name onto the 7 training classes, unmapped excluded; chance ~ 0.167):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.249 [0.229, 0.269] | 0.184 |
| level1_only | 0.247 [0.221, 0.273] | 0.183 |
| level12 | 0.230 [0.207, 0.254] | 0.180 |
| level123 | 0.242 [0.220, 0.263] | 0.177 |
| level123_l2 | 0.248 [0.228, 0.268] | 0.187 |
| bypass | 0.240 [0.216, 0.263] | 0.186 |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.013 [-0.015, +0.041] | 0.375 |
| CBM L2 concept vs flat | arousal | -0.014 [-0.037, +0.010] | 0.238 |
| CBM via emotion vs flat | valence | +0.008 [-0.015, +0.035] | 0.564 |
| CBM via emotion vs flat | arousal | -0.020 [-0.055, +0.007] | 0.191 |

Predicted dominance (trained on FACED pseudo-labels, exploratory (D3)) vs real DREAMER dominance, Spearman 0.013 [-0.068, 0.100].
