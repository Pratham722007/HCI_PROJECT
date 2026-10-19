"""Serialise model outputs as structured text for the explainer (CBM: concepts; flat: class probabilities only).

The explainer must use ONLY these facts. L1 is summarised per region x band (bins from the subject-relative z-score:
low < -0.43 <= typical <= 0.43 < high), so the prompt stays short and every statement is checkable.
"""
import numpy as np

from hcbm.concepts.level1 import TERTILE_Z

REGIONS = {"frontal": ("F7", "F3", "FC5", "FC6", "F4", "F8"), "temporal": ("T7", "T8"), "parieto-occipital": ("P7", "P8", "O1", "O2")}
BANDS = ("theta", "alpha", "gamma")
SYSTEM = ("You explain an EEG emotion-recognition result to a clinician. Use ONLY the facts given. Do not invent brain "
          "regions, frequencies, numbers or psychological claims that are not listed. Write 2-4 plain sentences.")
POS_WORDS = ("positive", "pleasant")
NEG_WORDS = ("negative", "unpleasant")


def zbin(z):
    return "high" if z > TERTILE_Z else "low" if z < -TERTILE_Z else "typical"


def l1_summary(l1, concept_names):
    idx = {n: i for i, n in enumerate(concept_names)}
    out = []
    for b in BANDS:
        for r, chs in REGIONS.items():
            z = float(np.mean([l1[idx[f"{b}_{c}"]] for c in chs]))
            out.append((f"{b} power, {r}", zbin(z), z))
    faa = float(l1[idx["FAA"]])
    out.append(("frontal alpha asymmetry (right minus left)", zbin(faa), faa))
    return out


def word_valence(v):
    return "positive (pleasant)" if v > 0.5 else "negative (unpleasant)"


def l2_words(l2):
    return [("valence", word_valence(l2[0]), l2[0]), ("arousal", "high" if l2[1] > 0.5 else "low", l2[1]),
            ("dominance [pseudo-label, exploratory]", "high" if l2[2] > 0.5 else "low", l2[2])]


def l3_words(l3):
    names = (("novelty", "novel/unexpected", "familiar/expected"), ("goal conduciveness", "goal-conducive", "goal-obstructive"),
             ("coping potential", "high coping", "low coping"))
    return [(n, hi if v > 0 else lo, v) for (n, hi, lo), v in zip(names, l3)]


def cbm_facts(rec, concept_names, l1_override=None):
    l1 = l1_summary(rec["l1"], concept_names)
    if l1_override:
        l1 = [(n, l1_override.get(n, b), z) for n, b, z in l1]
    lines = ["Level 1, EEG band power (relative to this person's own baseline):"]
    lines += [f"- {n}: {b}" for n, b, _ in l1 if b != "typical"] or ["- all regions typical"]
    lines += ["Level 2, affective state:"] + [f"- {n}: {w} ({v:.2f} on 0-1)" for n, w, v in l2_words(rec["l2"])]
    lines += ["Level 3, appraisal:"] + [f"- {n}: {w} ({v:+.2f} on -1..1)" for n, w, v in l3_words(rec["l3"])]
    lines += [f"Predicted emotion: {rec['emotion']} (confidence {rec['confidence']:.0%})."]
    return "\n".join(lines)


def flat_facts(rec):
    top = sorted(rec["class_probs"].items(), key=lambda kv: -kv[1])[:3]
    return "\n".join(["A neural network mapped the EEG embedding directly to emotion classes.",
                      "Top class probabilities: " + ", ".join(f"{k} {v:.0%}" for k, v in top) + ".",
                      f"Predicted emotion: {rec['emotion']} (confidence {rec['confidence']:.0%})."])


def user_prompt(facts):
    return f"Facts:\n{facts}\n\nExplain why the system predicted this emotion, going from the evidence to the conclusion."
