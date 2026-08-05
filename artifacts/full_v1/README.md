# Reviewed full V1 evidence

This directory is the compact, versioned evidence package for the completed full-profile run.
Training ran on Kaggle with a Tesla T4 on 2026-07-28. The downloaded artifact archive was then
checked for identity, completeness, fold/seed coverage, finite numeric values, and consistency
between saved predictions and reported metrics.

This is a frozen V1 result. Corrections to report wording and figure labels were regenerated from
the saved CSV/JSON files on 2026-08-05; no model was retrained. Any change to preprocessing,
splits, hyperparameter search, model selection, or training belongs to a separately versioned V2.

## Identity and provenance

| Item | Value |
|---|---|
| Configuration identity | `4acfb870cf9b` |
| Requested / evaluated subjects | 109 / 106 |
| Excluded subjects | 88, 92, 100 (128 Hz; protocol required 160 Hz) |
| Held-out protocol | 5 outer subject-grouped folds, 3 grouped inner folds |
| CNN repetitions | 5 seeds per outer fold |
| Runtime | Python 3.12.13, PyTorch 2.10.0+cu128, MNE 1.12.1, scikit-learn 1.6.1 |
| Accelerator | Kaggle Tesla T4 |
| Runtime source archive SHA-256 | `ab79422a52bee848cf0220cd57f098e14cb96234178ef7bb6e51195db625c567` |
| Downloaded result archive SHA-256 | `6a39c4088a5ce828ffd00ee45f166576347d5467d3e9913c91de8979e1003012` |

The Kaggle notebook was
[`adelfarfarann/notebook7c1043425b`](https://www.kaggle.com/code/adelfarfarann/notebook7c1043425b).
Its visibility depends on the owner's Kaggle sharing settings.

Crucially, the Git commit containing this directory includes post-run reporting and documentation
corrections. It is therefore not byte-identical to the runtime source archive that trained V1.
The source-archive checksum above, not the later release commit alone, identifies the training
code uploaded to Kaggle.

## Reviewed result

| Model | Mean subject balanced accuracy | 95% subject-bootstrap CI |
|---|---:|---:|
| Compact EEGNet-like CNN | 0.617 | [0.593, 0.641] |
| CSP + logistic regression | 0.581 | [0.560, 0.604] |

The paired CNN − CSP difference was `+0.035`, with percentile-bootstrap CI
`[0.021, 0.051]` across 106 held-out subjects. CNN was better for 68 subjects, CSP for 37, and
the models tied for 1. This supports a modest average CNN advantage within this fixed dataset and
protocol; it does not establish universal model superiority.

At 50% random channel removal, balanced accuracy was 0.530 for CNN and 0.524 for CSP. CNN
explanation stability across seeds had median Spearman correlation 0.504. These results argue
against presenting either channel-drop robustness or explanation reproducibility as uniformly
strong.

## Artifact audit

The downloaded archive contained the expected:

- 5 outer split files with exact train/validation/test disjointness and one test appearance for
  every validation-eligible subject;
- 1,080 successful CSP inner-CV rows (`5 × 72 × 3`) and 5 fitted CSP models/artifacts;
- 25 CNN checkpoints (`5 folds × 5 seeds`) with `refit_epochs = best_epoch + 1` metadata;
- 28,488 finite prediction rows whose metrics recomputed exactly;
- 22,260 channel-drop robustness rows;
- 25 integrated-gradient tensors, 1,920 saved importance rows, and 1,335 stability comparisons.

The binary models and attribution tensors were inspected in the downloaded archive but are not
committed here. The full archive checksum preserves their identity.

## Contents

- [`report.md`](report.md): corrected report regenerated from the frozen V1 tabular artifacts.
- [`figures/`](figures/): corrected figures, including bounded class-count width, model-specific
  explanation estimands, and explicit device labels for latency.
- [`results/`](results/): selected tabular evidence, split definitions, configuration, and runtime
  metadata. Raw epoch-level predictions are intentionally omitted to keep the repository compact.
- `artifact_manifest.sha256`: checksums for every committed file in this evidence package except
  the manifest itself.
- [`RUN_PROVENANCE.json`](RUN_PROVENANCE.json): machine-readable provenance and audit summary.

## Limitations that remain

- Evaluation used one public dataset and has no independent external-dataset replication.
- Cross-validation test subjects are held out, but folds share training data. The subject
  bootstrap is conditional on this run and is not dataset-to-dataset uncertainty.
- Every fold selected the upper CSP boundary `C=10`; three selected the upper
  `n_components=8`. A wider grid is a preregistered V2 question, not a reason to replace V1.
- Subject effects were heterogeneous, and CNN explanation rankings were only moderately stable
  across seeds.
- CNN latency was measured on CUDA and CSP latency on CPU. The values are run-specific operational
  measurements, not a controlled model-speed comparison.
