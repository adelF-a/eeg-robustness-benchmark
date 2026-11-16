# EEG Motor-Imagery Classification Benchmark — Methods and Reproduction

[Back to the project overview](../README.md)

Run the commands in this document from the repository root.

This document specifies the comparison of **CSP + logistic regression** with a
**compact EEGNet-like CNN** for imagined left- versus right-fist movement. Accuracy on
unseen participants, sensitivity to missing channels, and explanation stability are
measured separately; robustness and reliable explanations are not assumed.

> This is a research/methods benchmark. **No clinical claims are made.**

## 1. Research question

Does a compact CNN outperform CSP + logistic regression on previously unseen participants?
How much does channel loss reduce accuracy, and how stable are channel-level explanations?

We compare the two models on (a) generalisation to unseen subjects, (b) robustness to
channel dropout at test time, and (c) the stability of their channel-level explanations.

## 2. Why evaluation holds out whole participants

A random trial split can put recordings from the same participant in both training and
test sets. That measures performance on new trials from familiar participants, not the
intended task of generalising to a **new** person. Shared participant and recording
characteristics can make that result optimistic for the intended use case.

We therefore split by **subject group**: within each fold, no subject appears in more than one
of {train, validation, test}. All supervised steps (CSP, scaling, CNN normalisation,
hyperparameter and early-stopping choices) use training/validation subjects only. Split
integrity is asserted in code (`OuterFold.assert_disjoint`) and in tests
(`tests/test_subject_splits.py`).

## 3. Dataset and runs

PhysioNet **EEG Motor Movement/Imagery Dataset** (EEGMMIDB / EEGBCI), loaded through MNE.
The protocol requires 64 EEG channels and 160 Hz sampling. We use the **imagined left-vs-right
fist** runs **4, 8, 12** only.
Runs 6, 10, 14 (hands-vs-feet) are deliberately excluded — their `T1`/`T2` annotations mean
something different, and mixing them would corrupt the labels (the code refuses run numbers in
`{6, 10, 14}`).

Raw EDF files are downloaded to `data/raw/` and are **never** committed (`.gitignore`).

## 4. Event-label mapping

For runs 4/8/12:

| Annotation | Meaning | Our label |
|---|---|---|
| `T0` | rest | discarded |
| `T1` | imagined **left** fist | `0` (`left_fist`) |
| `T2` | imagined **right** fist | `1` (`right_fist`) |

## 5. Preprocessing pipeline

1. Standardise EEGBCI channel names; attach the standard 10-05 montage.
2. **Common-average reference** (neutral, subject-agnostic).
3. **Band-pass 7–30 Hz** — retains the mu (~8–13 Hz) and beta (~13–30 Hz) frequency bands.
   FIR, zero-phase; filtering skips annotated edges. This choice does not establish a
   physiological interpretation of the learned features.
4. Epoch `T1`/`T2` over **1.0–4.0 s post-cue** (avoids the cue transient, captures sustained
   imagery). No baseline correction (CAR + narrow band already remove drifts).

Processed epochs are cached under `data/processed/` keyed by a hash of the preprocessing
settings, so changing the band/window/reference invalidates the cache automatically.

## 6. CSP + logistic-regression baseline

`mne.decoding.CSP` (log-variance features) → `StandardScaler` → `LogisticRegression`.
CSP is supervised, so it is fitted on training subjects only. Hyperparameters (CSP
components, covariance regularisation, `C`, class weight) are selected by mean balanced
accuracy across **every grouped inner fold**. The selected pipeline is freshly refitted on the
complete outer-training population (train + validation). For interpretation we use CSP
**patterns** (forward model), keeping the per-component information rather than only an
aggregate. Per-fold scores, configuration means, failures, and hashes are saved.

## 7. Compact neural model

`CompactEEGNet` (`src/eeg_benchmark/models/compact_eegnet.py`): temporal conv → depthwise
spatial conv over channels → ELU → pooling → dropout → separable conv → pooling → global
average pool → 2-class linear head. Inspired by [EEGNet](https://arxiv.org/abs/1611.08024),
but not an exact reproduction; the published full-V1 model has **2,162 trainable parameters**.
Input normalisation uses **training-subject** statistics,
saved with the checkpoint. Training has two explicit phases: early stopping on one grouped
validation holdout selects a zero-based `best_epoch`; then a freshly initialised model and
optimizer are trained on train + validation for exactly `best_epoch + 1` epochs. The final
normalisation and class weights are recomputed from that complete outer-training population.
Multiple seeds are run in the full profile. Runs on CPU, CUDA, or Apple MPS.

## 8. Subject-held-out evaluation

- **Outer**: subject-grouped K-fold (`StratifiedGroupKFold`, `GroupKFold` fallback).
- **Inner for CSP**: every grouped inner fold is used for mean-based hyperparameter selection.
- **Validation for CNN**: the first grouped inner validation fold selects only the epoch count;
  the final model is refit on all outer-training subjects.
- Splits are saved below the active profile's result directory and reused by **both** models,
  so they see identical training and test subjects. Saved splits carry a config hash and are
  revalidated when loaded.

Metrics (balanced accuracy, macro-F1, ROC-AUC, class recalls, confusion) are computed at
epoch and **per-subject** level. The headline comparison is the **paired per-subject
difference** with a subject-level bootstrap CI. Individual CNN seeds are preserved.

For headline results, metrics are averaged across CNN seeds within each subject before
averaging across subjects. Every subject has equal weight. This is an average of model
scores, not an ensemble of their predictions. Balanced accuracy is
`(recall_left + recall_right) / 2`; its expected chance level is 50% for this binary task.
The bootstrap resamples subject-level scores or paired differences. It is conditional on
the recorded fits and splits; it does not include independent-dataset or full retraining
uncertainty.

## 9. Channel-drop robustness

Trained models are frozen; channel failure is applied **only to held-out test data** (no
retraining). Missing channels are imputed with **outer-training-subject channel means** (a dead
electrode whose value is unknown — never computed from test subjects). The **same masks** are
used for both models. Conditions:

- Random loss at 10 %, 25 %, 50 % (several seeded masks each).
- Regional loss: frontal, central/motor, left-hemisphere, right-hemisphere — derived from the
  actual channel names present.

## 10. Explanation methods

- **CSP**: per-component patterns + logistic-regression coefficients, plus a documented
  aggregate channel-importance vector.
- **CNN**: (1) **channel occlusion** — neutralise one channel, measure the drop in the
  reference-class probability; (2) **integrated gradients** (Captum if installed, else a
  self-contained implementation) from a zero baseline, aggregated to per-channel importance,
  with time-resolved maps saved for selected examples.

CNN explanations estimate the mean **true-class probability drop** over one deterministic,
model-independent held-out cohort: the first *N* epochs by `epoch_id` per test subject,
regardless of prediction correctness. The cohort is identical across seeds. Correctness is
saved as an annotation in `explanation_examples.csv`; it never controls selection.

## 11. Explanation-stability definition

Stability = how similar a model's channel-importance vector stays under small changes:
different **seeds** on the same cohort, **bootstrap** resamples of that cohort, small input
**noise** (CNN), and across **folds** (CSP). Each bootstrap replicate is compared with the
full-cohort importance vector, and `n_bootstrap_explain` is honoured exactly. Metrics are
Spearman rank correlation, cosine similarity, top-*k* Jaccard overlap, and sign agreement on
channel-aligned vectors. Low stability is a valid negative result, but it must be interpreted
for its recorded estimand and comparison condition.

## 12. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,explain for Captum integrated gradients
# CPU-only PyTorch (e.g. CI): pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Requires Python ≥ 3.10. GPU/MPS is used automatically when available but never required.

## 13. Smoke commands (fast functional check — NOT a scientific result)

```bash
python -m eeg_benchmark.cli download      --config configs/smoke.yaml
python -m eeg_benchmark.cli validate-data --config configs/smoke.yaml
python -m eeg_benchmark.cli run-all       --config configs/smoke.yaml
# or step by step:
python -m eeg_benchmark.cli prepare       --config configs/smoke.yaml
python -m eeg_benchmark.cli make-splits   --config configs/smoke.yaml
python -m eeg_benchmark.cli train-csp     --config configs/smoke.yaml
python -m eeg_benchmark.cli train-cnn     --config configs/smoke.yaml
python -m eeg_benchmark.cli evaluate      --config configs/smoke.yaml
python -m eeg_benchmark.cli stress-test   --config configs/smoke.yaml
python -m eeg_benchmark.cli explain       --config configs/smoke.yaml
python -m eeg_benchmark.cli report        --config configs/smoke.yaml
```

`make smoke` runs `run-all` on the smoke profile.

## 14. Full experiment commands

```bash
python -m eeg_benchmark.cli download      --config configs/full.yaml   # subjects 1–109
python -m eeg_benchmark.cli validate-data --config configs/full.yaml
python -m eeg_benchmark.cli run-all       --config configs/full.yaml   # or: make full
```

The full profile is configured for all valid subjects, 5 outer folds, 3 inner folds, 5 CNN
seeds, ≥1000 subject-level bootstraps, and the complete robustness/explanation suite. Expect a
multi-hour run on CPU (faster on GPU/MPS).

Full V1 was executed on Kaggle with a Tesla T4 on 2026-07-28 and its artifact bundle was audited.
Of 109 requested subjects, 106 passed validation and entered the five-fold subject-held-out
evaluation; subjects 88, 92, and 100 were excluded because their recordings were sampled at
128 Hz rather than the required 160 Hz. Any change to preprocessing, splits, model selection,
or training defines a new experiment rather than a correction to V1.

## 15. Results

Full-profile results write to `results/` and `reports/`. The smoke profile writes to
`results_smoke_active/` and `reports_smoke_active/`. These generated directories are excluded
from Git. The reporting code reads saved result files rather than hard-coded benchmark
values. The committed full-V1 bundle is the scientific evidence; small smoke runs only
verify pipeline execution.

### Reviewed full V1 result

Full V1 has configuration identity `4acfb870cf9b` and contains 4,748 epochs from 106
validation-eligible subjects.

| Model | Mean balanced accuracy | 95% subject-bootstrap CI |
|---|---:|---:|
| Compact EEGNet-like CNN | 0.617 | [0.593, 0.641] |
| CSP + logistic regression | 0.581 | [0.560, 0.604] |

The paired CNN − CSP difference was `+0.035` with percentile-bootstrap CI `[0.021, 0.051]`.
Because this interval is entirely above zero, the result supports a modest average CNN advantage
under this fixed protocol. Effects were heterogeneous: CNN performed better for 68 subjects, CSP
for 37, and the models tied for 1. This is evidence within this dataset and protocol, not
universal model superiority.

CNN recall averaged 63.3% for left imagery and 60.0% for right imagery, using the same
seed-then-subject averaging. Median subject balanced accuracy was 59.1%; 12 of 106 subjects
had seed-averaged scores below 50%. These are descriptive summaries of
[`subject_metrics.csv`](../artifacts/full_v1/results/subject_metrics.csv), not separate
significance tests for individual subjects. Classification remained modest and unreliable
for some participants.

At 50% random channel removal, mean balanced accuracy fell to 0.530 for CNN and 0.524 for CSP,
so the advantage narrowed substantially under severe channel loss. CNN explanation stability
was moderate across seeds (median Spearman 0.504); near-perfect noise stability may indicate
that the chosen perturbation was weak and should not be presented as decisive robustness
evidence.

The compact, versioned evidence bundle and exact provenance checksums are in
[`artifacts/full_v1/`](../artifacts/full_v1/README.md). Large checkpoints, raw predictions, and
attribution tensors remain outside Git.

Key artifacts: `predictions.csv`, `fold_metrics.csv`, `subject_metrics.csv`,
`model_comparison.csv`, `csp_inner_cv.csv`, `cnn_selection.csv`, `training_curves.csv`,
`robustness_metrics.csv`, `explanation_examples.csv`, `channel_importance.csv`,
`explanation_stability.csv`, and `efficiency.csv`.

## 16. Workflow

```
 raw EDF (PhysioNet, runs 4/8/12)
        │  download → validate-data → <results_dir>/data_validation.csv
        ▼
 preprocess (CAR, 7–30 Hz, 1–4 s epochs)  ──►  EpochBundle [epochs, channels, time]
        │  prepare (cache) → make-splits → <results_dir>/splits/*.json (subject-grouped)
        ▼
 ┌───────────────┐        ┌──────────────────────┐
 │ CSP + LogReg  │        │ CompactEEGNet (CNN)   │   train on train/val subjects only
 └──────┬────────┘        └──────────┬───────────┘
        └─────────► evaluate (per-subject, paired bootstrap) ◄─────────┘
                          │
        ┌─────────────────┼───────────────────────┐
        ▼                 ▼                        ▼
  stress-test        explain (occlusion,     report (figures +
  (channel drop)      IG, CSP patterns)       <reports_dir>/report.md)
        │                 │
        ▼                 ▼
 robustness_metrics  explanation_stability
```

## 17. Limitations

- Motor-imagery EEG has low SNR; absolute accuracies are modest and subject-variable.
- Explanations are **predictive** attributions over channels — not evidence of a neural
  mechanism, and not causal or clinical. A heatmap is not an explanation of physiology.
- A CNN producing a saliency map is **not** automatically "more explainable"; we treat
  explanation instability as a real finding.
- The smoke profile is a functional check only (too few subjects/seeds for inference).
- One public dataset and one predefined split protocol were evaluated; there is no
  external-dataset replication.
- Three requested subjects were excluded by the prespecified sampling-rate validation, so
  inferential results concern 106 subjects.
- CSP selected the largest tested `C=10` in every fold, and `n_components=8` in three folds.
  This boundary behavior motivates a separately preregistered V2 sensitivity experiment; the
  grid must not be expanded retrospectively to replace V1.
- Subject-level effects were heterogeneous, and explanation rankings were only moderately stable
  across CNN seeds.
- CNN latency was measured on a T4 GPU and CSP latency on CPU; those values describe the tested
  deployments, not intrinsic model-family speed.
- Earlier local Apple Silicon checks with NumPy 2.1.x / Accelerate emitted
  "encountered in matmul" RuntimeWarnings despite finite audited outputs. These warnings
  are filtered in the test settings in `pyproject.toml`. This is not a guarantee that every
  matrix warning is harmless; inspect fitting failures and non-finite outputs on a new run.

## 18. Ethical and scientific interpretation

This is a benchmark of methods, not a diagnostic or assistive device. Predictive performance
and channel attributions must not be read as claims about a person's neurology or intent.
Failed/excluded subjects and epochs are recorded with reasons below the active result
directory—never silently dropped.

## Reproduction

1. Install (§12). 2. `pytest` (synthetic, no download). 3. `download` + `validate-data`.
4. `run-all --config configs/smoke.yaml` (or `configs/full.yaml`). 5. Read the selected
profile's generated report.
Seeds, config, and environment/hardware metadata are saved alongside results
(`run_metadata.json`, `config_used.yaml`).

## Testing & CI

`pytest` runs an offline test suite (no network or GPU required), including a complete synthetic
end-to-end pipeline and regressions for identity, split integrity, explanation estimands,
reporting, device synchronization, and Git hygiene. GitHub Actions runs Ruff and the suite on
Python 3.11/3.12.

## License & citation

Code is released under [MIT](../LICENSE). See [CITATION.cff](../CITATION.cff) for project
citation metadata. Dataset terms and required acknowledgments are provided by PhysioNet.

## References

- Schalk, G. (2009). [EEG Motor Movement/Imagery Dataset, version 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/).
  PhysioNet. DOI: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P).
- Schalk et al. (2004). [BCI2000: a general-purpose brain-computer interface (BCI) system](https://pubmed.ncbi.nlm.nih.gov/15188875/).
- Pollard et al. (2026). [PhysioNet as a global platform for biomedical research](https://doi.org/10.1038/s44360-026-00096-z).
  Platform reference requested by the dataset page.
- Lawhern et al. (2018). [EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces](https://arxiv.org/abs/1611.08024).
  Architecture inspiration, not a claim of exact reproduction.
