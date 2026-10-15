"""H-CBM-EEG: Hierarchical Concept Bottleneck Models for EEG Emotion Recognition.

A research framework mapping EEGPT embeddings -> Neurophysiological (L1) ->
Affective (L2) -> Appraisal (L3) -> Emotion class, with Qwen natural-language explanations.
"""

__version__ = "0.1.0"

__all__ = [
    "io",
    "preprocess",
    "concepts",
    "models",
    "train",
    "eval",
    "explain",
    "hitl",
    "utils",
]
