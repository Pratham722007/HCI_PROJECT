# Dataset Registry: H-CBM-EEG

| Dataset | Access Method | Licence Status | Expected Size | Local Path | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FACED** | Synapse `syn50614194` (Processed_data `syn50615881`) | Synapse DUC (check terms) | 6.2 GB | `data/raw/faced` (symlinks to `datasets/FACED`) | `downloaded` — pickles inspected; metadata in `datasets/FACED/METADATA`; **self-report ratings still missing (in `Data.zip`, syn50614821)** |
| **DENS** (`ds003751`) | OpenNeuro | CC0 | ~15-25 GB | `datasets/dens` | `partial` (3/40 raw subjects), optional secondary |
| **DEAP** | Web form / EULA request to dataset creators | Restricted Research EULA | ~3 GB (preprocessed .dat/.mat) | `datasets/deap` | `partial` — 16/32 subjects, .dat labels corrupted (use xls) |
| **DREAMER** | Access request to authors (IEEE Dataport / authors) | Restricted Research Agreement | ~1.5 GB (.mat format) | `datasets/DREAMER.mat` | `verified` (23 subj, 14 ch, 128 Hz) |

---

### Dataset Roles in H-CBM-EEG Protocol

- **DENS (OpenNeuro ds003751)**:
  - **Role**: Primary dataset used for all model training, validation, hyperparameter tuning, loss weighting, temperature scaling, and escalation thresholds.
  - **Montage**: 128-channel EGI cap (subsampled to the shared 14-channel montage during preprocessing).
  - **Splits**: Strict subject-wise 5-fold cross-validation and held-out test partition. Zero subject overlap across train and test sets.

- **DEAP**:
  - **Role**: Frozen out-of-domain cross-corpus evaluation benchmark (Simulation 1). Strictly no fine-tuning or hyperparameter tuning.
  - **Montage**: 32 EEG channels (mapped to the shared 14-channel montage).

- **DREAMER**:
  - **Role**: Frozen out-of-domain cross-corpus evaluation benchmark (Simulation 1). Strictly no fine-tuning or hyperparameter tuning.
  - **Montage**: 14 channels (Emotiv EPOC) matching the standardized 14-channel montage.

---

### Protocol Notice (Hard Rules)
1. Files under `data/raw/` are strictly read-only.
2. All datasets are converted to a single unified HDF5 schema (`data/processed/*.h5`).
