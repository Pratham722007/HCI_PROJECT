# DEAP and DREAMER inspection (Phase 1.3) — 16 Oct 2026
Scripts: `scripts/inspect_deap.py`, `scripts/inspect_dreamer.py` (JSON alongside).

## DEAP
- **Only 16 of 32 subjects present** (`s01..s16.dat`). `participant_ratings.xls` covers all 32. Cross-corpus n is 16 subjects, 640 trials (≈9,600 windows), not 19,200.
- Each file: `data (40,40,8064) float64`, `labels (40,4)`; 0 NaN; 0 flat channels; 128 Hz, 63 s (3 s baseline = 384 samples). EEG std median 17.9 µV; 50 Hz PSD ~1e-7 (already filtered/notched); DC ≈ 0.24 µV.
- Trial order in `.dat` is **Experiment_id (video) order**, not presentation `Trial` order.
- **`.dat` labels are corrupted**: 198/640 trials (all 16 subjects) differ from the official `participant_ratings.xls`; every mismatch is exactly valence/arousal → 9 − x, giving impossible 0 ratings (7 subjects) and zero HVHA trials. **Use the xls ratings (matched on Experiment_id) as the only label source.** Using xls: V>5 53.1%, A>5 54.7%; quadrants LVLA 148 / LVHA 152 / HVLA 142 / HVHA 198; 9 trials have an exact 5 (threshold-tie).
- Channel names are not in files (Geneva order from the DEAP docs, assumed): all 12 shared channels present (indices F7=3, F3=2, FC5=4, T7=7, P7=11, O1=13, O2=31, P8=29, T8=25, FC6=21, F4=19, F8=20).
- Peripheral channels 32–39 must be dropped.

## DREAMER (`datasets/DREAMER.mat`, MATLAB v5, loads with scipy)
- 23 subjects × 18 clips = 414 trials; 14 channels AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4 (names in file); fs 128 Hz; all 12 shared channels present (idx 1–12). 0 NaN.
- **Stimulus lengths vary 67–394 s** (baseline fixed 61 s = 7808 samples); plan's "last 60 s" is feasible for all.
- **Large DC offset** (median |mean| ≈ 4,254 µV, std ≈174 µV) — DC removal mandatory before filtering/CAR. No 50 Hz problem (ratio 0.017).
- Ratings integer 1–5 for V/A/D. V>3: 39.4%, A>3: 43.7%; exact 3: 21.7% (V), 28.7% (A). The "≥3 vs >3" binarisation choice matters for ~¼ of trials → must be locked (plan says "3").
- **No emotion labels in the file**, only V/A/D. The 9-emotion → 7-class map needs the DREAMER paper's clip table (not on disk).
