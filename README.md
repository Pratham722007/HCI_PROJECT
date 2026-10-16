# H-CBM-EEG

Hierarchical Concept Bottleneck Models for EEG Emotion Recognition.

A research framework mapping EEGPT embeddings -> Neurophysiological concepts (L1) -> Affective dimensions (L2) -> Cognitive appraisal dimensions (L3) -> Emotion classification, with Qwen natural-language explanations.

## Protocol Highlights
- **Primary Dataset**: FACED (123 subjects, 28 clips, 9 emotion categories; 4 contaminated subjects excluded) for all training, cross-validation, and hyperparameter tuning. DENS is an optional secondary only.
- **Cross-Corpus Zero-Shot Testing**: DEAP and DREAMER evaluated frozen with zero fine-tuning.
- **Baseline**: Flat 2-layer MLP + GELU.
- **Shared Montage**: 12 channels (F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8) at 256 Hz.
