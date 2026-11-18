# Simulation 1: zero-shot cross-corpus generalisation (tag `faced12`)


Models trained on FACED only; temperature/thresholds frozen; no DEAP/DREAMER data used for any choice. Binary threshold = scale midpoint (0.5 after rescaling), ties dropped. CIs resample subjects.

**Metrics where the CBM is NOT significantly better than the flat baseline (H2):** deap valence accuracy (CBM L2 concept: +0.014, CI [-0.007, +0.035]); deap valence accuracy (CBM via emotion: -0.002, CI [-0.008, +0.004]); deap arousal accuracy (CBM via emotion: +0.026, CI [-0.009, +0.062]); dens valence accuracy (CBM L2 concept: -0.066, CI [-0.104, -0.034]); dreamer valence accuracy (CBM L2 concept: -0.024, CI [-0.079, +0.029]); dreamer arousal accuracy (CBM L2 concept: -0.018, CI [-0.065, +0.023]); dreamer valence accuracy (CBM via emotion: -0.005, CI [-0.021, +0.010]); dreamer arousal accuracy (CBM via emotion: -0.004, CI [-0.040, +0.032])


## DEAP (zero-shot, 32 subjects, 19133 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.451 [0.418, 0.482] | 0.362 | 0.513 [0.482, 0.545] | 0.493 | -0.006 [-0.054, 0.042] | 0.014 [-0.044, 0.068] |
| level1_only | L2 concept | 0.480 [0.452, 0.505] | 0.476 | 0.554 [0.506, 0.592] | 0.530 | 0.009 [-0.041, 0.064] | 0.095 [0.020, 0.168] |
| level1_only | via emotion | 0.459 [0.422, 0.486] | 0.404 | 0.527 [0.485, 0.562] | 0.523 | -0.005 [-0.056, 0.048] | 0.088 [0.006, 0.158] |
| level12 | L2 concept | 0.468 [0.438, 0.492] | 0.457 | 0.508 [0.468, 0.548] | 0.508 | -0.031 [-0.079, 0.021] | 0.077 [0.001, 0.150] |
| level12 | via emotion | 0.449 [0.418, 0.478] | 0.369 | 0.529 [0.484, 0.564] | 0.515 | -0.050 [-0.097, -0.007] | 0.071 [-0.002, 0.150] |
| level123 | L2 concept | 0.465 [0.436, 0.492] | 0.460 | 0.555 [0.513, 0.592] | 0.536 | -0.043 [-0.096, 0.014] | 0.097 [0.026, 0.164] |
| level123 | via emotion | 0.449 [0.418, 0.478] | 0.364 | 0.539 [0.493, 0.574] | 0.521 | -0.051 [-0.104, 0.002] | 0.071 [-0.004, 0.139] |
| level123_l2 | L2 concept | 0.468 [0.436, 0.494] | 0.431 | 0.545 [0.510, 0.576] | 0.526 | 0.009 [-0.054, 0.071] | 0.079 [0.018, 0.139] |
| level123_l2 | via emotion | 0.443 [0.409, 0.475] | 0.320 | 0.559 [0.523, 0.590] | 0.521 | -0.001 [-0.061, 0.054] | 0.078 [0.014, 0.137] |
| bypass | L2 concept | 0.478 [0.447, 0.503] | 0.474 | 0.538 [0.497, 0.574] | 0.507 | -0.014 [-0.059, 0.037] | 0.043 [-0.019, 0.104] |
| bypass | via emotion | 0.461 [0.426, 0.488] | 0.410 | 0.492 [0.455, 0.529] | 0.492 | -0.017 [-0.063, 0.030] | 0.032 [-0.031, 0.099] |

DEAP 4-quadrant accuracy (ties at 5 excluded; chance ~ 0.25):

| model | readout | quadrant acc [95% CI] |
|---|---|---|
| flat | via emotion | 0.212 [0.186, 0.236] |
| level1_only | L2 concept | 0.268 [0.233, 0.300] |
| level1_only | via emotion | 0.233 [0.200, 0.265] |
| level12 | L2 concept | 0.244 [0.214, 0.274] |
| level12 | via emotion | 0.229 [0.201, 0.262] |
| level123 | L2 concept | 0.263 [0.232, 0.293] |
| level123 | via emotion | 0.238 [0.207, 0.268] |
| level123_l2 | L2 concept | 0.259 [0.228, 0.290] |
| level123_l2 | via emotion | 0.240 [0.211, 0.273] |
| bypass | L2 concept | 0.256 [0.228, 0.286] |
| bypass | via emotion | 0.224 [0.192, 0.255] |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.014 [-0.007, +0.035] | 0.214 |
| CBM L2 concept vs flat | arousal | +0.042 [+0.004, +0.081] | 0.030 |
| CBM via emotion vs flat | valence | -0.002 [-0.008, +0.004] | 0.662 |
| CBM via emotion vs flat | arousal | +0.026 [-0.009, +0.062] | 0.156 |

Predicted dominance (trained on FACED pseudo-labels, exploratory (D3)) vs real DEAP dominance, Spearman 0.101 [0.020, 0.172].


## DENS (zero-shot, 40 subjects, 1900 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.601 [0.570, 0.644] | 0.478 | 0.518 [0.460, 0.579] | 0.476 | -0.022 [-0.094, 0.066] | 0.044 [-0.051, 0.123] |
| level1_only | L2 concept | 0.416 [0.381, 0.450] | 0.415 | 0.684 [0.629, 0.730] | 0.509 | -0.027 [-0.100, 0.067] | 0.025 [-0.091, 0.115] |
| level1_only | via emotion | 0.567 [0.530, 0.612] | 0.493 | 0.569 [0.526, 0.624] | 0.495 | -0.033 [-0.103, 0.064] | 0.038 [-0.064, 0.109] |
| level12 | L2 concept | 0.550 [0.513, 0.588] | 0.503 | 0.601 [0.546, 0.654] | 0.507 | 0.000 [-0.076, 0.077] | 0.069 [-0.069, 0.173] |
| level12 | via emotion | 0.616 [0.577, 0.662] | 0.477 | 0.672 [0.620, 0.729] | 0.523 | -0.005 [-0.079, 0.076] | 0.076 [-0.035, 0.169] |
| level123 | L2 concept | 0.535 [0.497, 0.571] | 0.484 | 0.641 [0.584, 0.695] | 0.521 | -0.033 [-0.126, 0.057] | 0.053 [-0.131, 0.207] |
| level123 | via emotion | 0.631 [0.590, 0.671] | 0.471 | 0.678 [0.628, 0.733] | 0.539 | -0.040 [-0.135, 0.047] | 0.080 [-0.077, 0.207] |
| level123_l2 | L2 concept | 0.504 [0.463, 0.545] | 0.466 | 0.607 [0.557, 0.659] | 0.521 | -0.050 [-0.128, 0.039] | 0.053 [-0.091, 0.159] |
| level123_l2 | via emotion | 0.644 [0.604, 0.683] | 0.459 | 0.675 [0.626, 0.722] | 0.540 | -0.041 [-0.126, 0.049] | 0.075 [-0.047, 0.180] |
| bypass | L2 concept | 0.439 [0.402, 0.478] | 0.437 | 0.652 [0.611, 0.699] | 0.524 | -0.037 [-0.116, 0.056] | 0.017 [-0.101, 0.132] |
| bypass | via emotion | 0.537 [0.495, 0.590] | 0.486 | 0.561 [0.518, 0.611] | 0.487 | -0.045 [-0.132, 0.058] | 0.004 [-0.090, 0.091] |

DENS coarse-emotion accuracy (**secondary**: labels mapped by class name onto the 7 training classes, unmapped excluded; chance ~ 0.200):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.189 [0.161, 0.222] | 0.143 |
| level1_only | 0.257 [0.223, 0.296] | 0.157 |
| level12 | 0.214 [0.192, 0.239] | 0.182 |
| level123 | 0.180 [0.144, 0.215] | 0.160 |
| level123_l2 | 0.180 [0.160, 0.207] | 0.162 |
| bypass | 0.230 [0.200, 0.264] | 0.166 |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | -0.066 [-0.104, -0.034] | 0.001 |
| CBM L2 concept vs flat | arousal | +0.124 [+0.055, +0.195] | 0.000 |
| CBM via emotion vs flat | valence | +0.030 [+0.003, +0.060] | 0.023 |
| CBM via emotion vs flat | arousal | +0.161 [+0.098, +0.225] | 0.000 |

Predicted dominance (trained on FACED pseudo-labels, exploratory (D3)) vs real DENS dominance, Spearman 0.026 [-0.054, 0.113].


## DREAMER (zero-shot, 23 subjects, 5817 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.500 [0.445, 0.546] | 0.376 | 0.590 [0.539, 0.636] | 0.527 | -0.014 [-0.071, 0.035] | 0.097 [0.029, 0.175] |
| level1_only | L2 concept | 0.492 [0.456, 0.524] | 0.458 | 0.585 [0.543, 0.622] | 0.547 | 0.004 [-0.080, 0.072] | 0.108 [0.034, 0.168] |
| level1_only | via emotion | 0.493 [0.448, 0.534] | 0.397 | 0.568 [0.526, 0.608] | 0.535 | -0.022 [-0.097, 0.040] | 0.099 [0.035, 0.172] |
| level12 | L2 concept | 0.488 [0.444, 0.519] | 0.468 | 0.549 [0.497, 0.587] | 0.542 | -0.037 [-0.130, 0.031] | 0.118 [0.038, 0.186] |
| level12 | via emotion | 0.496 [0.447, 0.539] | 0.415 | 0.584 [0.519, 0.634] | 0.538 | -0.036 [-0.127, 0.026] | 0.116 [0.030, 0.191] |
| level123 | L2 concept | 0.476 [0.441, 0.506] | 0.464 | 0.572 [0.525, 0.615] | 0.543 | -0.038 [-0.120, 0.026] | 0.106 [0.036, 0.174] |
| level123 | via emotion | 0.495 [0.445, 0.534] | 0.409 | 0.586 [0.533, 0.633] | 0.540 | -0.049 [-0.130, 0.015] | 0.099 [0.032, 0.160] |
| level123_l2 | L2 concept | 0.482 [0.442, 0.514] | 0.453 | 0.576 [0.528, 0.618] | 0.545 | -0.004 [-0.098, 0.067] | 0.100 [0.028, 0.152] |
| level123_l2 | via emotion | 0.496 [0.444, 0.542] | 0.372 | 0.595 [0.543, 0.642] | 0.538 | -0.009 [-0.103, 0.065] | 0.120 [0.050, 0.182] |
| bypass | L2 concept | 0.506 [0.470, 0.538] | 0.488 | 0.593 [0.537, 0.644] | 0.523 | 0.027 [-0.050, 0.090] | 0.080 [0.011, 0.150] |
| bypass | via emotion | 0.504 [0.463, 0.540] | 0.442 | 0.564 [0.516, 0.609] | 0.531 | 0.000 [-0.060, 0.067] | 0.058 [-0.014, 0.134] |

DREAMER coarse-emotion accuracy (**secondary**: labels mapped by class name onto the 7 training classes, unmapped excluded; chance ~ 0.167):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.192 [0.176, 0.207] | 0.175 |
| level1_only | 0.185 [0.167, 0.202] | 0.163 |
| level12 | 0.186 [0.162, 0.207] | 0.158 |
| level123 | 0.186 [0.162, 0.204] | 0.161 |
| level123_l2 | 0.177 [0.157, 0.194] | 0.160 |
| bypass | 0.213 [0.188, 0.232] | 0.167 |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | -0.024 [-0.079, +0.029] | 0.390 |
| CBM L2 concept vs flat | arousal | -0.018 [-0.065, +0.023] | 0.444 |
| CBM via emotion vs flat | valence | -0.005 [-0.021, +0.010] | 0.519 |
| CBM via emotion vs flat | arousal | -0.004 [-0.040, +0.032] | 0.816 |

Predicted dominance (trained on FACED pseudo-labels, exploratory (D3)) vs real DREAMER dominance, Spearman 0.027 [-0.054, 0.095].
