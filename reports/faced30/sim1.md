# Simulation 1: zero-shot cross-corpus generalisation (tag `faced30`)


Models trained on FACED only; temperature/thresholds frozen; no DEAP/DREAMER data used for any choice. Binary threshold = scale midpoint (0.5 after rescaling), ties dropped. CIs resample subjects.

**Metrics where the CBM is NOT significantly better than the flat baseline (H2):** deap arousal accuracy (CBM L2 concept: +0.008, CI [-0.002, +0.018]); deap valence accuracy (CBM via emotion: -0.006, CI [-0.015, +0.002]); deap arousal accuracy (CBM via emotion: +0.008, CI [-0.002, +0.019]); dens valence accuracy (CBM L2 concept: -0.066, CI [-0.103, -0.029]); dreamer valence accuracy (CBM L2 concept: -0.017, CI [-0.049, +0.015]); dreamer arousal accuracy (CBM L2 concept: -0.019, CI [-0.047, +0.012]); dreamer valence accuracy (CBM via emotion: -0.012, CI [-0.030, +0.007]); dreamer arousal accuracy (CBM via emotion: -0.004, CI [-0.030, +0.027])


## DEAP (zero-shot, 32 subjects, 19133 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.459 [0.434, 0.482] | 0.411 | 0.529 [0.511, 0.547] | 0.506 | 0.006 [-0.017, 0.032] | 0.023 [-0.003, 0.048] |
| level1_only | L2 concept | 0.521 [0.512, 0.532] | 0.512 | 0.548 [0.520, 0.576] | 0.497 | 0.031 [0.010, 0.052] | 0.034 [0.007, 0.062] |
| level1_only | via emotion | 0.483 [0.465, 0.502] | 0.471 | 0.523 [0.509, 0.537] | 0.510 | 0.023 [-0.000, 0.046] | 0.029 [0.000, 0.057] |
| level12 | L2 concept | 0.491 [0.474, 0.507] | 0.484 | 0.541 [0.515, 0.563] | 0.505 | 0.038 [0.016, 0.057] | 0.026 [0.000, 0.048] |
| level12 | via emotion | 0.466 [0.440, 0.492] | 0.420 | 0.544 [0.518, 0.570] | 0.495 | 0.027 [0.003, 0.048] | 0.022 [-0.002, 0.045] |
| level123 | L2 concept | 0.486 [0.471, 0.501] | 0.482 | 0.537 [0.518, 0.554] | 0.514 | 0.009 [-0.015, 0.032] | 0.042 [0.018, 0.066] |
| level123 | via emotion | 0.453 [0.424, 0.481] | 0.380 | 0.537 [0.516, 0.559] | 0.505 | -0.003 [-0.028, 0.021] | 0.032 [0.006, 0.057] |
| level123_l2 | L2 concept | 0.498 [0.485, 0.511] | 0.498 | 0.540 [0.519, 0.561] | 0.514 | 0.023 [-0.003, 0.046] | 0.051 [0.025, 0.075] |
| level123_l2 | via emotion | 0.454 [0.425, 0.480] | 0.379 | 0.548 [0.520, 0.576] | 0.503 | 0.010 [-0.015, 0.032] | 0.036 [0.011, 0.060] |
| bypass | L2 concept | 0.511 [0.500, 0.523] | 0.508 | 0.540 [0.519, 0.562] | 0.505 | 0.027 [0.001, 0.054] | 0.033 [0.011, 0.056] |
| bypass | via emotion | 0.478 [0.459, 0.497] | 0.461 | 0.524 [0.507, 0.541] | 0.500 | 0.025 [-0.002, 0.052] | 0.011 [-0.012, 0.037] |

DEAP 4-quadrant accuracy (ties at 5 excluded; chance ~ 0.25):

| model | readout | quadrant acc [95% CI] |
|---|---|---|
| flat | via emotion | 0.236 [0.218, 0.254] |
| level1_only | L2 concept | 0.287 [0.269, 0.305] |
| level1_only | via emotion | 0.251 [0.237, 0.266] |
| level12 | L2 concept | 0.266 [0.248, 0.283] |
| level12 | via emotion | 0.246 [0.223, 0.270] |
| level123 | L2 concept | 0.260 [0.248, 0.271] |
| level123 | via emotion | 0.237 [0.215, 0.258] |
| level123_l2 | L2 concept | 0.267 [0.254, 0.280] |
| level123_l2 | via emotion | 0.241 [0.216, 0.264] |
| bypass | L2 concept | 0.273 [0.258, 0.289] |
| bypass | via emotion | 0.245 [0.230, 0.260] |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.027 [+0.013, +0.042] | 0.001 |
| CBM L2 concept vs flat | arousal | +0.008 [-0.002, +0.018] | 0.130 |
| CBM via emotion vs flat | valence | -0.006 [-0.015, +0.002] | 0.137 |
| CBM via emotion vs flat | arousal | +0.008 [-0.002, +0.019] | 0.118 |

Predicted dominance (trained on FACED pseudo-labels, exploratory (D3)) vs real DEAP dominance, Spearman 0.005 [-0.017, 0.029].


## DENS (zero-shot, 40 subjects, 1900 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.639 [0.600, 0.674] | 0.512 | 0.569 [0.542, 0.594] | 0.483 | 0.034 [-0.033, 0.106] | 0.029 [-0.023, 0.075] |
| level1_only | L2 concept | 0.467 [0.443, 0.489] | 0.466 | 0.658 [0.604, 0.703] | 0.495 | 0.042 [-0.011, 0.092] | 0.021 [-0.042, 0.082] |
| level1_only | via emotion | 0.596 [0.570, 0.622] | 0.532 | 0.568 [0.532, 0.601] | 0.483 | 0.041 [-0.012, 0.093] | 0.008 [-0.052, 0.064] |
| level12 | L2 concept | 0.578 [0.543, 0.612] | 0.524 | 0.630 [0.582, 0.674] | 0.486 | 0.049 [-0.015, 0.113] | 0.013 [-0.043, 0.069] |
| level12 | via emotion | 0.644 [0.608, 0.679] | 0.501 | 0.656 [0.603, 0.703] | 0.502 | 0.037 [-0.032, 0.105] | 0.025 [-0.028, 0.075] |
| level123 | L2 concept | 0.573 [0.541, 0.605] | 0.524 | 0.619 [0.568, 0.659] | 0.495 | 0.054 [-0.021, 0.134] | 0.034 [-0.031, 0.092] |
| level123 | via emotion | 0.680 [0.638, 0.719] | 0.489 | 0.650 [0.596, 0.691] | 0.509 | 0.032 [-0.043, 0.110] | 0.032 [-0.032, 0.088] |
| level123_l2 | L2 concept | 0.527 [0.496, 0.560] | 0.504 | 0.611 [0.568, 0.646] | 0.494 | 0.031 [-0.039, 0.106] | 0.043 [-0.001, 0.087] |
| level123_l2 | via emotion | 0.663 [0.619, 0.703] | 0.481 | 0.666 [0.618, 0.707] | 0.508 | 0.010 [-0.056, 0.078] | 0.047 [0.003, 0.086] |
| bypass | L2 concept | 0.471 [0.440, 0.500] | 0.469 | 0.620 [0.575, 0.659] | 0.500 | 0.071 [0.002, 0.136] | 0.037 [-0.026, 0.091] |
| bypass | via emotion | 0.594 [0.572, 0.616] | 0.536 | 0.574 [0.544, 0.605] | 0.476 | 0.058 [-0.012, 0.126] | 0.005 [-0.052, 0.057] |

DENS coarse-emotion accuracy (**secondary**: labels mapped by class name onto the 7 training classes, unmapped excluded; chance ~ 0.200):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.205 [0.175, 0.235] | 0.171 |
| level1_only | 0.239 [0.201, 0.277] | 0.148 |
| level12 | 0.223 [0.191, 0.259] | 0.174 |
| level123 | 0.199 [0.172, 0.226] | 0.172 |
| level123_l2 | 0.199 [0.168, 0.229] | 0.162 |
| bypass | 0.223 [0.187, 0.260] | 0.163 |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | -0.066 [-0.103, -0.029] | 0.000 |
| CBM L2 concept vs flat | arousal | +0.050 [+0.017, +0.080] | 0.003 |
| CBM via emotion vs flat | valence | +0.042 [+0.024, +0.060] | 0.000 |
| CBM via emotion vs flat | arousal | +0.081 [+0.043, +0.114] | 0.000 |

Predicted dominance (trained on FACED pseudo-labels, exploratory (D3)) vs real DENS dominance, Spearman 0.032 [-0.039, 0.104].


## DREAMER (zero-shot, 23 subjects, 5817 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.515 [0.468, 0.559] | 0.465 | 0.590 [0.560, 0.622] | 0.553 | 0.031 [-0.014, 0.070] | 0.090 [0.025, 0.155] |
| level1_only | L2 concept | 0.523 [0.497, 0.550] | 0.523 | 0.607 [0.565, 0.648] | 0.560 | 0.047 [0.001, 0.094] | 0.120 [0.062, 0.184] |
| level1_only | via emotion | 0.523 [0.479, 0.564] | 0.497 | 0.587 [0.555, 0.618] | 0.562 | 0.026 [-0.022, 0.073] | 0.095 [0.027, 0.157] |
| level12 | L2 concept | 0.521 [0.485, 0.556] | 0.510 | 0.568 [0.530, 0.600] | 0.549 | 0.035 [-0.019, 0.090] | 0.093 [0.031, 0.161] |
| level12 | via emotion | 0.524 [0.472, 0.570] | 0.474 | 0.597 [0.556, 0.640] | 0.561 | 0.028 [-0.021, 0.079] | 0.118 [0.050, 0.196] |
| level123 | L2 concept | 0.498 [0.457, 0.533] | 0.484 | 0.572 [0.535, 0.609] | 0.545 | -0.012 [-0.074, 0.045] | 0.091 [0.027, 0.154] |
| level123 | via emotion | 0.503 [0.450, 0.553] | 0.411 | 0.587 [0.548, 0.627] | 0.543 | -0.015 [-0.077, 0.043] | 0.095 [0.028, 0.163] |
| level123_l2 | L2 concept | 0.513 [0.483, 0.542] | 0.510 | 0.590 [0.550, 0.633] | 0.560 | 0.003 [-0.050, 0.052] | 0.116 [0.051, 0.182] |
| level123_l2 | via emotion | 0.502 [0.451, 0.549] | 0.415 | 0.594 [0.556, 0.636] | 0.541 | -0.010 [-0.066, 0.039] | 0.103 [0.043, 0.163] |
| bypass | L2 concept | 0.532 [0.502, 0.562] | 0.532 | 0.593 [0.558, 0.628] | 0.551 | 0.072 [0.019, 0.123] | 0.102 [0.044, 0.166] |
| bypass | via emotion | 0.530 [0.486, 0.571] | 0.505 | 0.588 [0.555, 0.622] | 0.548 | 0.054 [0.000, 0.104] | 0.080 [0.017, 0.139] |

DREAMER coarse-emotion accuracy (**secondary**: labels mapped by class name onto the 7 training classes, unmapped excluded; chance ~ 0.167):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.216 [0.199, 0.232] | 0.178 |
| level1_only | 0.234 [0.213, 0.253] | 0.171 |
| level12 | 0.232 [0.213, 0.250] | 0.181 |
| level123 | 0.186 [0.170, 0.206] | 0.167 |
| level123_l2 | 0.204 [0.187, 0.218] | 0.167 |
| bypass | 0.239 [0.222, 0.255] | 0.183 |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | -0.017 [-0.049, +0.015] | 0.275 |
| CBM L2 concept vs flat | arousal | -0.019 [-0.047, +0.012] | 0.225 |
| CBM via emotion vs flat | valence | -0.012 [-0.030, +0.007] | 0.216 |
| CBM via emotion vs flat | arousal | -0.004 [-0.030, +0.027] | 0.745 |

Predicted dominance (trained on FACED pseudo-labels, exploratory (D3)) vs real DREAMER dominance, Spearman 0.021 [-0.032, 0.072].
