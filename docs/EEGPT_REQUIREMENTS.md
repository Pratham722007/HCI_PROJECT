# EEGPT requirements (Phase 2.7) — from the official repo, read 16 Oct 2026

Source: https://github.com/BINE022/EEGPT, commit `ea7282f698c406cd423ccbedbe9960853761144c` (Apache-2.0), cloned to `third_party/EEGPT` (gitignored).
Checkpoint: `eegpt_mcae_58chs_4s_large4E.ckpt` from the README's figshare link (`https://figshare.com/s/e37df4f8a907a866df4b`, path `Files/EEGPT/checkpoint/`). **Must be downloaded manually in a browser** (figshare blocks scripted downloads); place at `checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt`. Record its sha256 in the embeddings file (done automatically).

| Requirement | EEGPT | Our data | OK? |
|---|---|---|---|
| Sampling rate | 256 Hz | 256 Hz | yes |
| Window | 4 s = 1024 samples (patch 64, stride 64 -> 16 time patches) | 4 s, 1024 | yes |
| Channels | any subset of a fixed 62-name vocabulary (10-10 names, upper-case); ids passed as `chan_ids` in the SAME order as data rows | F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8, all in vocabulary (ids 7, 9, 17, 25, 43, 59, 61, 51, 33, 23, 13, 15) | yes |
| `img_size` | `[n_channels, 1024]` (any n_channels works; checkpoint trained with 58) | [12, 1024] | yes |
| Architecture (official linear-probe config) | embed_dim 512, embed_num (summary tokens) 4, depth 8, heads 8, patch 64 | same | yes |
| Parameters | README: "10M"; this config counts **25.3M** (incl. channel embedding) | – | note: the "large4E" checkpoint may be the larger variant; strict state-dict load will confirm |
| Output | `(B, 16 patches, 4 summary tokens, 512)` | we mean-pool patches -> **(B, 4, 512) = 2048-d** | **plan assumed 512-d: embedding_dim updated to 2048 (4 x 512)** |
| Weights loading | `state_dict` keys prefixed `target_encoder.` (official probe scripts) | strict load in `EEGPTEncoder` | yes |
| Amplitude scaling | **not specified**; pretraining pipeline feeds volts x 1e3 (i.e. mV); TUEG fine-tuning feeds uV / 100; downstream probe scripts feed raw uV | our windows are per-subject z-scored (unit std), physical units removed | **open: choose `input_scale` multiplier on FACED validation subjects only (candidates 0.01, 0.1, 1, 10); DEAP/DREAMER never used** |
| Reference / filtering | not specified for the checkpoint | CAR over 12 ch, 4-45 Hz | note as a limitation |

Preprocessing differences vs EEGPT's pretraining data (mixed 58-channel datasets): we use only 12 of 58 channels (no coverage of central/parietal midline channels) — Gate 3 probes whether the embedding still carries emotion signal.

CPU cost measured with random weights: 64 windows in 2.2 s -> FACED (23,324 windows) ~13 min on CPU, once, cached.
