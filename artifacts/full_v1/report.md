# EEG Motor-Imagery Robustness Benchmark — Report

_Config profile:_ **full**<br>
_Device:_ cuda  _torch:_ 2.10.0+cu128  _mne:_ 1.12.1

> This report is generated automatically from files in `results/`. Numbers appear only when the corresponding stage has actually run.

## Research question

Does a compact neural EEG classifier outperform a CSP + logistic-regression baseline on previously unseen subjects, and how robust and explainable are both models when EEG channels are missing?

## Dataset & validation

- Subjects inspected: **109**; subjects with ≥1 failing run: **3**.
- Total T1 (left) epochs: **2396**, T2 (right) epochs: **2352**, rejected: **20**.
- Excluded/failed runs (subject, run, reason):
  - s88 run 4: unexpected sfreq 128.0 (expected 160.0)
  - s88 run 8: unexpected sfreq 128.0 (expected 160.0)
  - s88 run 12: unexpected sfreq 128.0 (expected 160.0)
  - s92 run 4: unexpected sfreq 128.0 (expected 160.0)
  - s92 run 8: unexpected sfreq 128.0 (expected 160.0)
  - s92 run 12: unexpected sfreq 128.0 (expected 160.0)
  - s100 run 4: unexpected sfreq 128.0 (expected 160.0)
  - s100 run 8: unexpected sfreq 128.0 (expected 160.0)
  - s100 run 12: unexpected sfreq 128.0 (expected 160.0)

## Leakage prevention

- Subject-grouped outer folds saved: **5** (`results/splits/`).
- Split method: **stratified_group_kfold**; outer folds: **5**; grouped inner folds: **3**; seed: **20260718**.
- Verified disjoint train/val/test subjects (see `tests/test_subject_splits.py`).
- CSP parameters are selected by mean performance across all grouped inner folds. CNN early stopping uses the first grouped inner validation fold; a freshly initialized CNN is then refit on the complete outer-training set.

## Results — generalisation to unseen subjects

Per-subject balanced accuracy (held-out test subjects):

| Model | mean | 95% CI | median |
|---|---|---|---|
| cnn | 0.617 | [0.593, 0.641] | 0.591 |
| csp | 0.581 | [0.560, 0.604] | 0.544 |

Paired CNN − CSP balanced accuracy: **+0.035** (95% CI [0.021, 0.051], n=106 subjects; CNN better on 64% of subjects).

Interpretation: the percentile-bootstrap CI lies entirely above 0, supporting a positive average paired CNN − CSP difference within this dataset and protocol. This does not establish universal superiority.

## Channel-drop robustness (test-time, no retraining)

Mean balanced accuracy under random channel loss:

| Model | fraction removed | balanced accuracy |
|---|---|---|
| cnn | 0.09 | 0.590 |
| cnn | 0.25 | 0.570 |
| cnn | 0.50 | 0.530 |
| csp | 0.09 | 0.561 |
| csp | 0.25 | 0.547 |
| csp | 0.50 | 0.524 |

## Explainability & explanation stability

Median explanation stability (Spearman of channel-importance vectors):

| Model | condition | median Spearman | median cosine |
|---|---|---|---|
| cnn | bootstrap | 0.668 | 0.850 |
| cnn | noise | 0.998 | 1.000 |
| cnn | seed | 0.504 | 0.891 |
| csp | fold | 0.786 | 0.981 |

Explanations are predictive attributions over channels, **not** proof of a neural mechanism. Low stability is reported as a legitimate negative result.

## Efficiency

| Model | reported fitted params | definition | size (KB) | latency p50 (ms) | latency p95 (ms) | device |
|---|---|---|---|---|---|---|
| cnn | 2162 | trainable_tensor_entries | 18.0 | 0.523 | 0.621 | cuda |
| csp | 391 | used_csp_filters_plus_logreg_coef_intercept | 67.3 | 0.728 | 0.845 | cpu |

CSP and CNN parameter counts use different model families and are descriptive, not a direct comparison of capacity. Accelerator timings are synchronized before wall-clock measurement.

CNN and CSP latency were measured on different devices/backends (cnn: cuda; csp: cpu). These are run-specific operational measurements and must not be interpreted as an intrinsic model-speed comparison.

## Limitations

- Motor-imagery EEG has low SNR; absolute accuracies can be modest.
- Explanations are correlational, not causal or clinical.
- The full profile evaluates one public dataset under one predefined protocol; there is no external-dataset replication.
- Cross-validation folds share training data. Subject-bootstrap intervals describe held-out-subject variation conditional on this run, not independent-study uncertainty.
- No clinical claims are made; this is a methods benchmark.
