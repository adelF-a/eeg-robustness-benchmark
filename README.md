# EEG Motor-Imagery Robustness Benchmark

[![CI](https://github.com/adelF-a/eeg-robustness-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/adelF-a/eeg-robustness-benchmark/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Can a compact neural network classify imagined left- versus right-hand movement in a
person it has never seen, and what happens when EEG channels fail?**

This benchmark compares **CSP + logistic regression** with a **compact EEGNet-like CNN**
using subject-held-out evaluation, missing-channel stress tests, and explanation-stability
analysis. It uses PhysioNet EEGMMIDB through MNE. This is a research benchmark; no clinical
claims are made.

[Results and evidence](artifacts/full_v1/README.md) ·
[Methods and reproduction](docs/METHODS.md)

## Results at a glance

The completed full V1 experiment contains **4,748 epochs from 106 subjects**, evaluated with
**5 outer subject-grouped folds** and **5 CNN seeds per fold**. Three of the 109 requested
subjects failed the prespecified sampling-rate check. Training ran on a Tesla T4 on
2026-07-28; the evidence bundle was reviewed and its reports updated on 2026-08-05.

| Model | Mean subject balanced accuracy | 95% subject-bootstrap CI |
|---|---:|---:|
| Compact EEGNet-like CNN | 61.7% | 59.3–64.1% |
| CSP + logistic regression | 58.1% | 56.0–60.4% |

The paired CNN advantage is **3.5 percentage points** (95% CI: **2.1–5.1**). The CNN
performed better for 68 subjects, CSP for 37, with one tie. This supports a modest average
advantage within this dataset and fixed protocol. The intervals describe held-out-subject
variation conditional on this run; the folds share training data.

![Each point compares the two models on one held-out subject](artifacts/full_v1/figures/subject_scatter.png)

**Sensor loss remains a weakness.** At 50% random channel removal, balanced accuracy falls
to 53.0% for the CNN and 52.4% for CSP. CNN explanation rankings have only moderate stability
across seeds (median Spearman correlation: 0.504).

![Balanced accuracy under random channel removal](artifacts/full_v1/figures/robustness_curve.png)

The [full report](artifacts/full_v1/report.md), [machine-readable results](artifacts/full_v1/results/),
[run provenance](artifacts/full_v1/RUN_PROVENANCE.json), and
[artifact checksums](artifacts/full_v1/artifact_manifest.sha256) are committed together.
Large checkpoints, raw EEG, epoch-level predictions, and attribution tensors are excluded
from Git; the evidence overview records what was audited and retained.

## Evaluation design

- **Task:** imagined left versus right fist, runs 4/8/12. `T1` maps to left, `T2` to right;
  rest events are excluded.
- **Preprocessing:** common-average reference, 7–30 Hz filtering, 1–4 s post-cue epochs.
- **Unseen subjects:** all epochs from a person stay together. Training, validation, and
  test subjects are disjoint during model selection; both final models are refit on the
  complete outer-training set before evaluation on untouched test subjects.
- **Model selection:** CSP uses all grouped inner folds; CNN epoch selection uses one grouped
  validation fold, followed by a fresh refit. Scaling and imputation use training data only.
- **Robustness:** both models receive the same random and regional channel masks at test time.
- **Explanations:** CSP patterns, CNN channel occlusion, and integrated gradients; stability
  is quantified across seeds, resampling, noise, or folds as appropriate to each model.

See [methods and protocol details](docs/METHODS.md) for the complete design and its limits.

## Quickstart

Python 3.10+ is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Installation needs network access. Once dependencies are installed, the tests use synthetic
data and run without an EEG download or GPU. CUDA and Apple MPS are used when available;
CPU is supported.

Run a small real-data functional check:

```bash
python -m eeg_benchmark.cli download --config configs/smoke.yaml
python -m eeg_benchmark.cli validate-data --config configs/smoke.yaml
python -m eeg_benchmark.cli run-all --config configs/smoke.yaml
```

The first command downloads data for five subjects. Outputs go to `results_smoke_active/`
and `reports_smoke_active/`. These smoke results verify execution; they are too small for
scientific conclusions. To review the project without running training, see the
[methods](docs/METHODS.md) and the [committed full-V1 evidence](artifacts/full_v1/README.md).

For a full experiment:

```bash
python -m eeg_benchmark.cli download --config configs/full.yaml
python -m eeg_benchmark.cli validate-data --config configs/full.yaml
python -m eeg_benchmark.cli run-all --config configs/full.yaml
```

The full run can take hours depending on hardware. The [methods guide](docs/METHODS.md)
documents individual stages, cached data, output paths, and reproduction metadata.

## Repository guide

| Location | What it contains |
|---|---|
| [`src/eeg_benchmark/`](src/eeg_benchmark/) | Data processing, models, training, evaluation, and reporting |
| [`configs/`](configs/) | Smoke and full experiment configurations |
| [`tests/`](tests/) | Offline tests for splits, fitting, explanations, reporting, and evidence integrity |
| [`artifacts/full_v1/`](artifacts/full_v1/) | Reviewed result tables, figures, saved splits, and provenance |
| [`docs/METHODS.md`](docs/METHODS.md) | Full scientific method and execution instructions |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Development milestones, scientific decisions, and verification history |

GitHub Actions runs Ruff and the offline test suite on Python 3.11 and 3.12.

## Limits and next experiment

- Results come from one public dataset and one split protocol; external replication remains open.
- Performance varies substantially between subjects. A channel-attribution map does not
  establish a physiological mechanism or a causal explanation.
- Every fold selected CSP's upper search boundary `C=10`; three selected `n_components=8`.
  A wider grid should be a separately specified V2 sensitivity experiment.
- Near-perfect explanation stability under the chosen small-noise perturbation may reflect
  a weak perturbation. It does not establish broad robustness.
- Published CNN latency used CUDA and CSP latency used CPU. Those measurements describe the
  tested deployments and cannot establish an intrinsic speed advantage.

The full V1 bundle is a frozen experiment. Protocol changes require a new version; its
results should remain available for comparison.

## License and citation

Code is released under the [MIT license](LICENSE). See [CITATION.cff](CITATION.cff)
for project citation metadata. Dataset details and attribution are provided by
[PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/).
