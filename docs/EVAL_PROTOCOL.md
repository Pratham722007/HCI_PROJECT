# Human evaluation protocol (Phase 10)

## Materials
- `reports/<tag>/human_eval/rater_form.html`: offline form (any browser). Each item shows explanation **A** and **B** of one
  EEG emotion prediction, each with the evidence it was generated from. Ratings auto-save in the browser; the rater
  presses **Download CSV** and returns `ratings_<id>.csv`.
- `items.json`: the pairs (A/B order randomised, seed in the key file). `key_PRIVATE.json`: which side is which system.
  **Never send the key file to raters.**
- About 150 pairs: FACED test windows stratified by emotion class and CBM correct/wrong, plus a smaller DEAP share
  (`scripts/explain.py --n 120 --n-deap 30`).

## Raters
At least 3, ideally clinicians or psychologists. Check whether your institution needs ethics approval for an expert
survey **before** contacting raters; record the decision here. Raters are told that both explanations come from AI systems,
not which system is which, and that ground truth is hidden.

## Criteria (1 = poor, 5 = excellent)
| criterion | definition shown to raters |
|---|---|
| faithfulness | The explanation only states things supported by the evidence shown, and its reasoning matches the evidence. |
| plausibility | The reasoning is psychologically/physiologically sensible. |
| clarity | Easy to read and understand. |
| actionability | It would help a clinician decide whether to trust or check this prediction. |

## Blinding limits
The CBM explanation cites concept evidence and the flat one cannot, so experienced raters may guess the system.
Architecture wording is scrubbed (`make_eval_set.py: scrub`), but this residual unblinding is reported as a limitation.

## Procedure
1. Pilot with 2-3 items with one rater; fix wording; regenerate the form if the text changed.
2. Send the form to each rater with their ID; collect CSVs into `reports/<tag>/human_eval/`.
3. `.venv/bin/python -E -P scripts/analyse_ratings.py --tag <tag>` writes `reports/<tag>/human_eval.md`: per-criterion
   means with item-bootstrap CIs, Wilcoxon signed-rank + rank-biserial effect size, Benjamini-Hochberg FDR across the
   four criteria, Krippendorff's alpha (interval) and Fleiss' kappa, and a split by CBM-correct vs wrong prediction.

## Consent text (template)
"You are invited to rate AI-generated explanations of emotion predictions from EEG recordings. No personal data about you
is collected beyond your ratings and an anonymous rater ID. Participation is voluntary; you may stop at any time and ask
for your ratings to be deleted. Ratings will be reported only in aggregate."
