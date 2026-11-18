# Simulation 1: zero-shot cross-corpus generalisation (tag `dens12_ft`)


Models trained on DENS only; temperature/thresholds frozen; no DEAP/DREAMER data used for any choice. Binary threshold = scale midpoint (0.5 after rescaling), ties dropped. CIs resample subjects.

**Metrics where the CBM is NOT significantly better than the flat baseline (H2):** deap valence accuracy (CBM L2 concept: -0.007, CI [-0.017, +0.003]); deap arousal accuracy (CBM L2 concept: -0.001, CI [-0.003, +0.001]); deap valence accuracy (CBM via emotion: -0.024, CI [-0.039, -0.009]); deap arousal accuracy (CBM via emotion: +0.000, CI [-0.000, +0.001]); dreamer valence accuracy (CBM L2 concept: +0.022, CI [-0.005, +0.048]); dreamer arousal accuracy (CBM L2 concept: +0.002, CI [-0.001, +0.005]); dreamer valence accuracy (CBM via emotion: +0.008, CI [-0.024, +0.038]); dreamer arousal accuracy (CBM via emotion: -0.002, CI [-0.008, +0.003])


## DEAP (zero-shot, 32 subjects, 19133 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.479 [0.454, 0.502] | 0.457 | 0.584 [0.527, 0.640] | 0.371 | 0.018 [-0.020, 0.054] | 0.000 [-0.041, 0.038] |
| level123 | L2 concept | 0.472 [0.444, 0.499] | 0.435 | 0.583 [0.526, 0.638] | 0.378 | 0.032 [-0.016, 0.078] | -0.043 [-0.091, 0.011] |
| level123 | via emotion | 0.455 [0.425, 0.483] | 0.373 | 0.584 [0.528, 0.640] | 0.369 | 0.010 [-0.032, 0.055] | 0.021 [-0.024, 0.069] |

DEAP 4-quadrant accuracy (ties at 5 excluded; chance ~ 0.25):

| model | readout | quadrant acc [95% CI] |
|---|---|---|
| flat | via emotion | 0.271 [0.230, 0.313] |
| level123 | L2 concept | 0.267 [0.224, 0.312] |
| level123 | via emotion | 0.251 [0.208, 0.298] |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | -0.007 [-0.017, +0.003] | 0.158 |
| CBM L2 concept vs flat | arousal | -0.001 [-0.003, +0.001] | 0.352 |
| CBM via emotion vs flat | valence | -0.024 [-0.039, -0.009] | 0.002 |
| CBM via emotion vs flat | arousal | +0.000 [-0.000, +0.001] | 0.518 |

Predicted dominance (trained on DENS self-ratings) vs real DEAP dominance, Spearman 0.005 [-0.081, 0.084].


## DREAMER (zero-shot, 23 subjects, 5817 non-rejected windows)

| model | readout | valence acc [95% CI] | valence macro-F1 | arousal acc [95% CI] | arousal macro-F1 | Spearman V [CI] | Spearman A [CI] |
|---|---|---|---|---|---|---|---|
| flat | via emotion | 0.500 [0.468, 0.528] | 0.491 | 0.625 [0.549, 0.697] | 0.398 | 0.003 [-0.046, 0.048] | 0.028 [-0.044, 0.104] |
| level123 | L2 concept | 0.521 [0.480, 0.559] | 0.510 | 0.627 [0.552, 0.699] | 0.411 | 0.070 [-0.007, 0.136] | -0.000 [-0.092, 0.088] |
| level123 | via emotion | 0.507 [0.464, 0.549] | 0.435 | 0.623 [0.547, 0.697] | 0.384 | 0.034 [-0.024, 0.091] | 0.057 [-0.000, 0.123] |

DREAMER coarse-emotion accuracy (**secondary**, clip-level target emotion -> 6 classes, unmapped clips excluded):

| model | acc [95% CI] | macro-F1 |
|---|---|---|
| flat | 0.240 [0.220, 0.260] | 0.154 |
| level123 | 0.165 [0.146, 0.186] | 0.138 |

Paired subject bootstrap, CBM(level123) minus flat (accuracy):

| comparison | dim | diff [95% CI] | p |
|---|---|---|---|
| CBM L2 concept vs flat | valence | +0.022 [-0.005, +0.048] | 0.106 |
| CBM L2 concept vs flat | arousal | +0.002 [-0.001, +0.005] | 0.289 |
| CBM via emotion vs flat | valence | +0.008 [-0.024, +0.038] | 0.612 |
| CBM via emotion vs flat | arousal | -0.002 [-0.008, +0.003] | 0.520 |

Predicted dominance (trained on DENS self-ratings) vs real DREAMER dominance, Spearman 0.103 [-0.001, 0.202].
