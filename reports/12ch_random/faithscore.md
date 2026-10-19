# FaithScore (tag `12ch_random`, explainer `template`)

> **PLACEHOLDER RUN: EEGPT weights were NOT loaded (random-init encoder).** Numbers below only show the pipeline runs end to end; they are not research results. Re-run `scripts/run_all.sh` with the real checkpoint.


> **Template explainer:** deterministic text that restates the concepts, faithful by construction. These numbers only check the pipeline; run with `--backend qwen` for the real result.


Signal (a) = SST-2 sentiment label flips in the expected direction; (b) = explanation uses the new valence term. Placebo = L1 text-only change, any polarity change is a false positive. CI = subject bootstrap.

| dataset | model | perturbation | n | rate (a) [95% CI] | rate (b) [95% CI] |
|---|---|---|---|---|---|
| deap | CBM | valence flip (FaithScore) | 20 | 0.700 [0.529, 0.864] | 1.000 [1.000, 1.000] |
| deap | CBM | L1 placebo (false-positive rate) | 20 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| deap | flat | class-valence flip (FaithScore) | 20 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| faced | CBM | valence flip (FaithScore) | 60 | 0.883 [0.806, 0.963] | 1.000 [1.000, 1.000] |
| faced | CBM | L1 placebo (false-positive rate) | 60 | 0.050 [0.000, 0.100] | 0.000 [0.000, 0.000] |
| faced | flat | class-valence flip (FaithScore) | 60 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

The flat baseline has no concepts; its perturbation flips the class summary instead (asymmetry, see limitations).
Judge reliability: fill `human_faithful` (1/0) in `faithscore_manual_labels.csv`, then run `scripts/faithscore_agreement.py --tag 12ch_random`.

## Examples

**faced row 703** (true happy, CBM tenderness -> flip neutral)

- original: EEG evidence: theta power, frontal: low; theta power, temporal: low; alpha power, frontal: low. This pattern was read as positive valence with high arousal. The situation was appraised as novel/unexpected, goal-conducive, high coping, which is most consistent with tenderness (confidence 16%).
- valence flipped: EEG evidence: theta power, frontal: low; theta power, temporal: low; alpha power, frontal: low. This pattern was read as negative valence with high arousal. The situation was appraised as novel/unexpected, goal-obstructive, high coping, which is most consistent with neutral (confidence 15%).
- placebo: EEG evidence: theta power, frontal: low; theta power, temporal: low; alpha power, frontal: low. This pattern was read as positive valence with high arousal. The situation was appraised as novel/unexpected, goal-conducive, high coping, which is most consistent with tenderness (confidence 16%).

**faced row 14470** (true happy, CBM happy -> flip neutral)

- original: EEG evidence: theta power, frontal: high; theta power, temporal: high; alpha power, frontal: high. This pattern was read as positive valence with low arousal. The situation was appraised as novel/unexpected, goal-conducive, high coping, which is most consistent with happy (confidence 15%).
- valence flipped: EEG evidence: theta power, frontal: high; theta power, temporal: high; alpha power, frontal: high. This pattern was read as negative valence with low arousal. The situation was appraised as novel/unexpected, goal-conducive, high coping, which is most consistent with neutral (confidence 15%).
- placebo: EEG evidence: theta power, frontal: high; theta power, temporal: high; alpha power, frontal: high. This pattern was read as positive valence with low arousal. The situation was appraised as novel/unexpected, goal-conducive, high coping, which is most consistent with happy (confidence 15%).

**faced row 11154** (true tenderness, CBM disgust -> flip tenderness)

- original: EEG evidence: alpha power, frontal: high; alpha power, temporal: high; alpha power, parieto-occipital: high. This pattern was read as negative valence with high arousal. The situation was appraised as novel/unexpected, goal-obstructive, low coping, which is most consistent with disgust (confidence 17%).
- valence flipped: EEG evidence: alpha power, frontal: high; alpha power, temporal: high; alpha power, parieto-occipital: high. This pattern was read as positive valence with high arousal. The situation was appraised as novel/unexpected, goal-conducive, high coping, which is most consistent with tenderness (confidence 18%).
- placebo: EEG evidence: alpha power, frontal: high; alpha power, temporal: high; alpha power, parieto-occipital: high. This pattern was read as negative valence with high arousal. The situation was appraised as novel/unexpected, goal-obstructive, low coping, which is most consistent with disgust (confidence 17%).
