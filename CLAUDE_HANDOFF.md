# Claude handoff: protocol hardening and full V1 review complete

Steps 1–7 and the full V1 run are complete. Preserve V1 as the reviewed experiment with config
identity `4acfb870cf9b`; documentation/report corrections do not justify retraining. Any change
to the scientific protocol must be versioned as V2. Do not treat the preserved `results_smoke/`
snapshot as evidence for the current protocol.

## Final Step 5–7 changes

- CNN explanations use a deterministic first-N-per-subject test cohort shared by every seed,
  target the true class, and record prediction correctness separately.
- Each configured explanation bootstrap is compared with the full-cohort importance estimate;
  the configured count is no longer capped.
- Confusion figures show per-true-class rates with seeds equally weighted.
- CUDA/MPS latency is synchronized; CSP timing is seeded and uses the configured repeat count.
- Reports include split parameters, conditional smoke warnings, and honest parameter definitions.
- Binary artifact ignores cover full, active-smoke, and legacy-smoke paths.
- README, PLAN, learning guide, full-profile wording, and package license metadata were updated.

## Verified on 2026-07-20

- `ruff format --check .`: passed
- `ruff check .`: passed
- `pytest -q`: passed (offline suite)
- `python -m build --no-isolation`: passed; sdist and wheel built without the old license warning
- Real-data `run-all --config configs/smoke.yaml`: passed into
  `results_smoke_active/` and `reports_smoke_active/`
- Smoke integrity: 15/15 subject-run validations OK; 225 epochs; 450 finite prediction rows;
  one config hash; all CSP persisted arrays finite; all 64 inner-CV rows successful; exact five
  bootstrap replicates per fold/seed; 10 figures rendered

Apple Accelerate/MNE still emits noisy matrix RuntimeWarnings during local smoke CSP fitting; the
run completed and the explicit finite-value audit passed. The full run completed on Kaggle/Tesla
T4 on 2026-07-28: 109 subjects inspected, 106 retained, 4,748 epochs, all expected
fold/seed/model/explanation artifacts present, and `Pipeline exit status: 0`. Reviewed balanced
accuracy was 0.617 for CNN versus 0.581 for CSP, paired difference `+0.035` with CI
`[0.021, 0.051]`.
