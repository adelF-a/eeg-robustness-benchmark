# Development and Verification Record

[Project overview](../README.md) · [Methods](METHODS.md) · [Full V1 evidence](../artifacts/full_v1/README.md)

This record preserves the completed implementation milestones, scientific decisions, and
verification history. Dates below refer to those original runs, not a fresh execution.

Research question: *Does a compact neural EEG classifier outperform a CSP + logistic-regression
baseline on previously unseen subjects, and how robust and explainable are both models when EEG
channels are missing?*

Dataset: PhysioNet EEG Motor Movement/Imagery (EEGMMIDB / EEGBCI) via MNE. Runs 4, 8, 12.
Task: imagined left fist (T1) vs imagined right fist (T2). 64 channels, 160 Hz.

## Phase 1 — Inspect & plan
- [x] Inspect repository (empty git repo, no commits)
- [x] Identify tooling: Python 3.13 (miniforge), pip; Apple Silicon (MPS available after torch install)
- [x] Select dependency versions (numpy<2.3, torch>=2.2, mne>=1.6, sklearn>=1.3)
- [x] Record the implementation plan and verification history

## Phase 2 — Scaffold
- [x] Package structure `src/eeg_benchmark/`
- [x] Config loading (dataclasses + YAML)
- [x] CLI skeleton
- [x] Logging, seeds, hardware reporting
- [x] `.gitignore`, `pyproject.toml`, CI

## Phase 3 — Synthetic foundation
- [x] Synthetic EEG fixtures
- [x] Splits, metrics, CSP pipeline, neural model
- [x] Tests
- [x] Make CI-style tests pass before real data

## Phase 4 — Real-data pipeline
- [x] MNE download + validation
- [x] Preprocessing + caching
- [x] Real-data smoke test (network permitting)

## Phase 5 — Experiments
- [x] Preserve the original smoke snapshot as legacy evidence only
- [x] Inspect the original implementation for leakage, asymmetry, stale artifacts, and
  explanation/reporting confounds
- [x] Re-run the active smoke profile after protocol hardening
- [x] Full V1 profile executed on Kaggle and artifact-audited (109 requested, 106 valid)

## Phase 6 — Scientific protocol hardening
- [x] Authoritative data validation, strict run resolution, and explicit subject exclusions
- [x] Config-hashed manifests, folds, predictions, and isolated profile paths
- [x] Genuine grouped inner CV for CSP; deterministic mean-based selection
- [x] CNN validation-only epoch selection and fresh train+validation refit
- [x] Identical outer-training population for both final models and robustness imputation
- [x] Shared model-independent explanation cohort with true-label reference classes
- [x] Bootstrap explanations compared with the full-cohort estimate; configured count honoured
- [x] Rate-normalized confusion matrices, synchronized accelerator timing, explicit split report
- [x] Harden binary-artifact ignore rules for full, active-smoke, and legacy-smoke paths

## Phase 7 — Verification
- [x] Formatting (ruff format)
- [x] Linting (ruff check)
- [x] Full test suite (pytest)
- [x] Synthetic end-to-end test
- [x] Hardened real-data smoke experiment
- [x] Final package build
- [x] Verify active-smoke artifacts and generated report
- [x] Final documentation audit
- [x] Full V1 completeness, identity, finite-value, fold/seed-coverage, and report audit

## Actual verified outcomes
- **Legacy smoke (2026-07-18):** completed before the protocol audit. Its files are preserved in
  `results_smoke/` but its metrics must not support current scientific claims because the
  evaluation and explanation contracts subsequently changed.
- **Current hardened implementation (2026-07-20):** Ruff formatting and lint passed; the offline
  test suite passed; exact `python -m build --no-isolation` produced both sdist and wheel.
- **Active real-data smoke:** `run-all` completed in the isolated active directories. Validation
  accepted all 15 subject/run files (5 subjects, 225 epochs); both folds completed training,
  evaluation, robustness, explanations, reporting, and 10 figures. All prediction probabilities
  and persisted CSP filters/patterns/classifier coefficients were checked finite. This is a
  functional verification only, not inferential evidence.
- Apple Accelerate/MNE emitted the documented matrix RuntimeWarnings during CSP fitting, but no
  configuration failed and the post-run finite-value audit passed.
- **Reviewed full V1 (2026-07-28):** completed on Kaggle/Tesla T4 with config identity
  `4acfb870cf9b`. Validation inspected 109 subjects and retained 106, producing 4,748 epochs. All
  five outer folds, 25 CNN checkpoints, 1,080 CSP inner-CV rows, 22,260 robustness rows, 25 IG
  artifacts, and the expected explanation comparisons were present and finite.
- Mean subject-level balanced accuracy was 0.617 for CNN and 0.581 for CSP; paired difference
  `+0.035`, percentile-bootstrap CI `[0.021, 0.051]`. This is a modest within-protocol result,
  not a universal-superiority claim.
- **Future V2:** preregister a wider CSP grid because every fold selected the upper `C=10`
  boundary. Do not alter or replace V1 retrospectively.

## Key scientific decisions (documented)
- Subject-grouped outer CV and grouped inner CV using `StratifiedGroupKFold` (fallback
  `GroupKFold`); saved partitions are exact, complete, and config-stamped.
- Band-pass 7–30 Hz (mu + beta motor rhythms). Epoch window 1.0–4.0 s post-cue.
- Common-average reference. No baseline correction (CAR + narrow band already center data).
- Channel imputation for drop tests uses the final models' complete **outer-training** subject
  population only.
- CNN explanations use the same deterministic held-out examples for every seed, target the true
  class, and record correctness without filtering on it.
- Both final models are freshly refit on the same outer-training subjects. CNN normalization and
  class weights are recomputed on that population and saved with the checkpoint.
