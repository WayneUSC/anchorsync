# Changelog

## 0.1.0 — 2026-10-29

First release.

- Anchor detection for audio, accelerometer, generic multichannel (force/tactile) and video streams with
  constant-fraction onsets.
- Robust matching (offset voting + ICP-style refinement) tolerant to missed and spurious events and large offsets.
- Offset + drift clock fits with parameter covariance, offset-only fallback with a dual-anchor warning.
- Timing health: dropouts (jitter-aware), non-monotonic/duplicate timestamps, nominal-rate error.
- Resampling onto a shared timeline with dropout masking; fits saved/re-applied as JSON.
- Loaders: WAV, CSV/TSV (Phyphox exports), NPZ, video files (OpenCV, motion energy or brightness).
- CLI (`fit`, `check`, `detect`, `demo`), JSON/TOML configs, self-contained HTML report.
