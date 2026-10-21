# Simulation 1: zero-shot cross-corpus generalisation (tag `dens12`)


Models trained on DENS only; temperature/thresholds frozen; no DEAP/DREAMER data used for any choice. Binary threshold = scale midpoint (0.5 after rescaling), ties dropped. CIs resample subjects.

**Metrics where the CBM is NOT significantly better than the flat baseline (H2):** deap arousal accuracy (CBM L2 concept: -0.001, CI [-0.002, +0.000]); deap valence accuracy (CBM via emotion: -0.013, CI [-0.023, -0.004]); deap arousal accuracy (CBM via emotion: -0.000, CI [-0.001, +0.001]); dreamer valence accuracy (CBM L2 concept: +0.023, CI [-0.011, +0.057]); dreamer arousal accuracy (CBM L2 concept: +0.003, CI [-0.001, +0.007]); dreamer valence accuracy (CBM via emotion: +0.003, CI [-0.018, +0.025]); dreamer arousal accuracy (CBM via emotion: -0.002, CI [-0.009, +0.003])


## DEAP (zero-shot, 32 subjects, 19133 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.470 [0.444, 0.494] | 0.431 | 0.584 [0.528, 0.640] | 0.373 | 0.024 [-0.015, 0.059] | 0.004 [-0.040, 0.046] |
| level1_only | L2 concept | 0.475 [0.452, 0.497] | 0.457 | 0.583 [0.527, 0.639] | 0.374 | 0.016 [-0.023, 0.055] | 0.008 [-0.042, 0.062] |
| level1_only | via emotion | 0.472 [0.450, 0.495] | 0.447 | 0.584 [0.527, 0.640] | 0.370 | 0.015 [-0.024, 0.051] | 0.003 [-0.043, 0.053] |
| level12 | L2 concept | 0.495 [0.472, 0.517] | 0.489 | 0.578 [0.527, 0.629] | 0.428 | 0.034 [-0.016, 0.085] | 0.028 [-0.010, 0.066] |
| level12 | via emotion | 0.470 [0.444, 0.495] | 0.431 | 0.584 [0.527, 0.640] | 0.369 | 0.030 [-0.013, 0.073] | 0.048 [0.011, 0.087] |
| level123 | L2 concept | 0.483 [0.457, 0.506] | 0.474 | 0.583 [0.527, 0.638] | 0.376 | 0.016 [-0.025, 0.060] | -0.022 [-0.068, 0.024] |
| level123 | via emotion | 0.457 [0.429, 0.483] | 0.393 | 0.584 [0.528, 0.640] | 0.369 | 0.002 [-0.038, 0.047] | 0.026 [-0.010, 0.062] |
| level123_l2 | L2 concept | 0.480 [0.453, 0.507] | 0.459 | 0.580 [0.526, 0.634] | 0.394 | 0.039 [-0.004, 0.081] | -0.004 [-0.048, 0.040] |
| level123_l2 | via emotion | 0.459 [0.429, 0.486] | 0.394 | 0.584 [0.528, 0.640] | 0.369 | 0.017 [-0.024, 0.060] | 0.042 [0.003, 0.085] |
| bypass | L2 concept | 0.477 [0.453, 0.500] | 0.463 | 0.584 [0.528, 0.640] | 0.372 | -0.002 [-0.039, 0.034] | 0.006 [-0.040, 0.051] |
| bypass | via emotion | 0.484 [0.463, 0.504] | 0.482 | 0.584 [0.527, 0.640] | 0.371 | -0.001 [-0.039, 0.038] | 0.004 [-0.041, 0.047] |

DEAP 4-quadrant accuracy (ties at 5 excluded; chance ~ 0.25):

| model | readout | quadrant acc [95% CI] |
|---|---|---|
| flat | via emotion | 0.261 [0.219, 0.305] |
| level1_only | L2 concept | 0.272 [0.233, 0.314] |
| level1_only | via emotion | 0.266 [0.225, 0.309] |
| level12 | L2 concept | 0.287 [0.253, 0.324] |
| level12 | via emotion | 0.264 [0.222, 0.309] |
| level123 | L2 concept | 0.278 [0.239, 0.320] |
| level123 | via emotion | 0.254 [0.213, 0.301] |
| level123_l2 | L2 concept | 0.272 [0.230, 0.313] |
| level123_l2 | via emotion | 0.253 [0.212, 0.300] |
| bypass | L2 concept | 0.269 [0.228, 0.312] |
| bypass | via emotion | 0.279 [0.240, 0.320] |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.013 [+0.000, +0.026] | 0.046 |
| CBM L2 concept vs flat | arousal | -0.001 [-0.002, +0.000] | 0.092 |
| CBM via emotion vs flat | valence | -0.013 [-0.023, -0.004] | 0.003 |
| CBM via emotion vs flat | arousal | -0.000 [-0.001, +0.001] | 0.785 |

Predicted dominance (trained on DENS self-ratings) vs real DEAP dominance, Spearman -0.012 [-0.080, 0.047].


## DREAMER (zero-shot, 23 subjects, 5817 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.503 [0.464, 0.537] | 0.476 | 0.625 [0.551, 0.696] | 0.400 | 0.021 [-0.033, 0.071] | 0.011 [-0.067, 0.091] |
| level1_only | L2 concept | 0.522 [0.485, 0.560] | 0.511 | 0.625 [0.550, 0.698] | 0.393 | 0.056 [-0.011, 0.121] | 0.048 [-0.031, 0.131] |
| level1_only | via emotion | 0.517 [0.483, 0.552] | 0.496 | 0.623 [0.548, 0.696] | 0.388 | 0.038 [-0.015, 0.090] | 0.076 [0.010, 0.143] |
| level12 | L2 concept | 0.533 [0.494, 0.571] | 0.531 | 0.625 [0.553, 0.690] | 0.467 | 0.078 [-0.006, 0.160] | 0.097 [0.032, 0.169] |
| level12 | via emotion | 0.510 [0.473, 0.548] | 0.472 | 0.623 [0.547, 0.697] | 0.384 | 0.066 [-0.010, 0.144] | 0.070 [0.001, 0.153] |
| level123 | L2 concept | 0.526 [0.488, 0.563] | 0.522 | 0.627 [0.552, 0.700] | 0.404 | 0.073 [-0.001, 0.141] | 0.028 [-0.054, 0.116] |
| level123 | via emotion | 0.507 [0.469, 0.546] | 0.451 | 0.623 [0.547, 0.697] | 0.384 | 0.033 [-0.028, 0.091] | 0.072 [0.003, 0.152] |
| level123_l2 | L2 concept | 0.517 [0.480, 0.554] | 0.513 | 0.627 [0.554, 0.696] | 0.431 | 0.066 [-0.018, 0.138] | -0.000 [-0.094, 0.091] |
| level123_l2 | via emotion | 0.509 [0.470, 0.546] | 0.463 | 0.623 [0.547, 0.698] | 0.386 | 0.037 [-0.034, 0.099] | 0.085 [0.020, 0.153] |
| bypass | L2 concept | 0.506 [0.474, 0.538] | 0.491 | 0.625 [0.548, 0.698] | 0.391 | 0.028 [-0.026, 0.078] | 0.063 [-0.020, 0.150] |
| bypass | via emotion | 0.511 [0.483, 0.540] | 0.510 | 0.625 [0.549, 0.698] | 0.394 | 0.007 [-0.043, 0.054] | 0.032 [-0.037, 0.110] |

DREAMER coarse-emotion accuracy (**secondary**, clip-level target emotion -> 6 classes, unmapped clips excluded):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.222 [0.202, 0.240] | 0.160 |
| level1_only | 0.257 [0.242, 0.270] | 0.154 |
| level12 | 0.241 [0.222, 0.261] | 0.160 |
| level123 | 0.190 [0.169, 0.209] | 0.151 |
| level123_l2 | 0.212 [0.191, 0.232] | 0.148 |
| bypass | 0.256 [0.237, 0.273] | 0.151 |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.023 [-0.011, +0.057] | 0.184 |
| CBM L2 concept vs flat | arousal | +0.003 [-0.001, +0.007] | 0.214 |
| CBM via emotion vs flat | valence | +0.003 [-0.018, +0.025] | 0.751 |
| CBM via emotion vs flat | arousal | -0.002 [-0.009, +0.003] | 0.587 |

Predicted dominance (trained on DENS self-ratings) vs real DREAMER dominance, Spearman 0.100 [0.006, 0.192].
