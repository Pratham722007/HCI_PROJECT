# Simulation 1: zero-shot cross-corpus generalisation (tag `12ch_random`)

> **PLACEHOLDER RUN: EEGPT weights were NOT loaded (random-init encoder).** Numbers below only show the pipeline runs end to end; they are not research results. Re-run `scripts/run_all.sh` with the real checkpoint.


Models trained on FACED only; temperature/thresholds frozen; no DEAP/DREAMER data used for any choice. Binary threshold = scale midpoint (0.5 after rescaling), ties dropped. CIs resample subjects.

**Metrics where the CBM is NOT significantly better than the flat baseline (H2):** deap arousal accuracy (CBM L2 concept: +0.002, CI [-0.021, +0.023]); deap arousal accuracy (CBM via emotion: -0.004, CI [-0.035, +0.026]); dreamer valence accuracy (CBM L2 concept: +0.003, CI [-0.039, +0.045]); dreamer arousal accuracy (CBM L2 concept: -0.005, CI [-0.032, +0.022]); dreamer valence accuracy (CBM via emotion: -0.002, CI [-0.013, +0.007]); dreamer arousal accuracy (CBM via emotion: -0.029, CI [-0.061, +0.002])


## DEAP (zero-shot, 32 subjects, 19133 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.441 [0.408, 0.473] | 0.309 | 0.539 [0.509, 0.569] | 0.502 | 0.001 [-0.047, 0.053] | 0.010 [-0.040, 0.061] |
| level1_only | L2 concept | 0.467 [0.438, 0.495] | 0.411 | 0.538 [0.496, 0.580] | 0.512 | 0.031 [-0.024, 0.085] | 0.021 [-0.049, 0.088] |
| level1_only | via emotion | 0.443 [0.412, 0.475] | 0.317 | 0.535 [0.500, 0.571] | 0.507 | 0.039 [-0.012, 0.091] | 0.010 [-0.046, 0.068] |
| level12 | L2 concept | 0.457 [0.432, 0.483] | 0.429 | 0.546 [0.516, 0.576] | 0.512 | -0.008 [-0.067, 0.053] | 0.011 [-0.035, 0.065] |
| level12 | via emotion | 0.444 [0.412, 0.474] | 0.329 | 0.554 [0.523, 0.584] | 0.498 | 0.012 [-0.039, 0.063] | 0.009 [-0.042, 0.064] |
| level123 | L2 concept | 0.466 [0.437, 0.494] | 0.426 | 0.541 [0.509, 0.569] | 0.519 | 0.015 [-0.036, 0.065] | 0.041 [-0.013, 0.097] |
| level123 | via emotion | 0.444 [0.412, 0.476] | 0.329 | 0.535 [0.509, 0.564] | 0.499 | 0.008 [-0.040, 0.055] | 0.001 [-0.067, 0.071] |
| level123_l2 | L2 concept | 0.464 [0.439, 0.489] | 0.438 | 0.538 [0.506, 0.571] | 0.519 | -0.001 [-0.051, 0.047] | 0.035 [-0.016, 0.089] |
| level123_l2 | via emotion | 0.441 [0.410, 0.472] | 0.328 | 0.562 [0.523, 0.600] | 0.492 | -0.009 [-0.058, 0.036] | 0.016 [-0.032, 0.068] |
| bypass | L2 concept | 0.453 [0.426, 0.481] | 0.394 | 0.558 [0.521, 0.597] | 0.490 | -0.034 [-0.089, 0.021] | 0.006 [-0.075, 0.085] |
| bypass | via emotion | 0.441 [0.408, 0.473] | 0.306 | 0.574 [0.529, 0.619] | 0.468 | -0.017 [-0.070, 0.037] | 0.011 [-0.047, 0.073] |

DEAP 4-quadrant accuracy (ties at 5 excluded; chance ~ 0.25):

| model | readout | quadrant acc [95% CI] |
|---|---|---|
| flat | via emotion | 0.226 [0.199, 0.254] |
| level1_only | L2 concept | 0.256 [0.225, 0.284] |
| level1_only | via emotion | 0.235 [0.209, 0.262] |
| level12 | L2 concept | 0.249 [0.230, 0.270] |
| level12 | via emotion | 0.231 [0.205, 0.258] |
| level123 | L2 concept | 0.237 [0.213, 0.259] |
| level123 | via emotion | 0.225 [0.197, 0.254] |
| level123_l2 | L2 concept | 0.249 [0.227, 0.270] |
| level123_l2 | via emotion | 0.234 [0.205, 0.264] |
| bypass | L2 concept | 0.245 [0.215, 0.272] |
| bypass | via emotion | 0.236 [0.201, 0.272] |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.024 [+0.009, +0.042] | 0.001 |
| CBM L2 concept vs flat | arousal | +0.002 [-0.021, +0.023] | 0.880 |
| CBM via emotion vs flat | valence | +0.003 [+0.000, +0.006] | 0.036 |
| CBM via emotion vs flat | arousal | -0.004 [-0.035, +0.026] | 0.829 |

Exploratory (D3): predicted dominance (trained on FACED pseudo-labels) vs real DEAP dominance, Spearman 0.029 [-0.063, 0.108].


## DREAMER (zero-shot, 23 subjects, 5817 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.493 [0.432, 0.549] | 0.335 | 0.539 [0.463, 0.605] | 0.469 | -0.024 [-0.105, 0.050] | -0.019 [-0.139, 0.090] |
| level1_only | L2 concept | 0.493 [0.450, 0.533] | 0.457 | 0.564 [0.503, 0.621] | 0.529 | 0.012 [-0.059, 0.084] | 0.109 [0.025, 0.186] |
| level1_only | via emotion | 0.498 [0.436, 0.555] | 0.354 | 0.539 [0.470, 0.600] | 0.493 | 0.015 [-0.055, 0.083] | 0.045 [-0.047, 0.116] |
| level12 | L2 concept | 0.482 [0.443, 0.522] | 0.457 | 0.539 [0.473, 0.597] | 0.516 | 0.005 [-0.069, 0.079] | 0.084 [-0.034, 0.165] |
| level12 | via emotion | 0.493 [0.436, 0.545] | 0.368 | 0.545 [0.472, 0.607] | 0.503 | -0.016 [-0.078, 0.048] | 0.050 [-0.074, 0.137] |
| level123 | L2 concept | 0.496 [0.460, 0.532] | 0.457 | 0.534 [0.476, 0.584] | 0.511 | 0.020 [-0.050, 0.088] | 0.059 [-0.052, 0.145] |
| level123 | via emotion | 0.490 [0.435, 0.543] | 0.357 | 0.510 [0.455, 0.555] | 0.483 | 0.019 [-0.047, 0.085] | -0.038 [-0.152, 0.060] |
| level123_l2 | L2 concept | 0.491 [0.461, 0.524] | 0.475 | 0.557 [0.505, 0.608] | 0.545 | 0.016 [-0.054, 0.088] | 0.129 [0.036, 0.202] |
| level123_l2 | via emotion | 0.494 [0.442, 0.544] | 0.375 | 0.562 [0.501, 0.622] | 0.522 | -0.005 [-0.078, 0.065] | 0.083 [-0.018, 0.157] |
| bypass | L2 concept | 0.487 [0.441, 0.525] | 0.450 | 0.604 [0.542, 0.662] | 0.494 | -0.020 [-0.111, 0.066] | 0.045 [-0.051, 0.138] |
| bypass | via emotion | 0.496 [0.435, 0.553] | 0.332 | 0.561 [0.484, 0.632] | 0.453 | -0.019 [-0.094, 0.058] | -0.017 [-0.125, 0.082] |

DREAMER coarse-emotion accuracy (**secondary**, loose 9->7 mapping, surprise clips excluded):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.154 [0.135, 0.171] | 0.145 |
| level1_only | 0.177 [0.157, 0.196] | 0.175 |
| level12 | 0.181 [0.160, 0.202] | 0.164 |
| level123 | 0.167 [0.152, 0.184] | 0.158 |
| level123_l2 | 0.176 [0.156, 0.195] | 0.170 |
| bypass | 0.167 [0.148, 0.187] | 0.156 |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.003 [-0.039, +0.045] | 0.858 |
| CBM L2 concept vs flat | arousal | -0.005 [-0.032, +0.022] | 0.764 |
| CBM via emotion vs flat | valence | -0.002 [-0.013, +0.007] | 0.709 |
| CBM via emotion vs flat | arousal | -0.029 [-0.061, +0.002] | 0.069 |

Exploratory (D3): predicted dominance (trained on FACED pseudo-labels) vs real DREAMER dominance, Spearman 0.064 [-0.024, 0.157].
