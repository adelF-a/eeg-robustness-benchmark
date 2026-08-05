# Learning Guide — a 7-day walk through this project

A paced tour of the EEG robustness benchmark, **~45 minutes per day for one week**. Each day
follows one layer of the pipeline in the order data actually flows through it. For every day:
*Read* (files, with the concept), *Why it's built this way*, *Do* (a hands-on command or test),
and *Checkpoint* questions to confirm you understood it before moving on.

Setup once (already done in this repo): `source .venv/bin/activate`. Everything below uses the
fast **smoke** profile and the already-downloaded subjects 1–5, so no day waits on a big run.

The single best mental model to hold the whole week:

```
raw EDF ─▶ preprocess ─▶ EpochBundle[epochs,channels,time] ─▶ subject-grouped splits
        ─▶ {CSP+LogReg, CompactEEGNet} ─▶ evaluate ─▶ {robustness, explanations} ─▶ report
```

Everything is orchestrated by `src/eeg_benchmark/pipeline.py` (one function per stage) and
exposed by `src/eeg_benchmark/cli.py`. When lost, start at `pipeline.run_all` and read outward.

---

## Day 1 — The question, and the one idea that shapes everything

**Read (20 min)**
- `README.md` §1–2 and §8 — the research question and *why random epoch splitting leaks subjects*.
- `PLAN.md` — the "Key scientific decisions" and "Actual verified outcomes" sections.
- `src/eeg_benchmark/config.py` — skim the dataclasses; note that **every experimental knob is
  in YAML**, not hidden in code. Open `configs/smoke.yaml` beside it.

**Why it's built this way (10 min).** EEG epochs from one person are highly correlated (anatomy,
electrode placement). If the same subject's epochs land in both train and test, a model can
"recognise the person" instead of the task, and accuracy looks great but doesn't generalise.
The entire evaluation design exists to prevent this. Hold that thought all week.

**Do (10 min)**
```bash
python -m eeg_benchmark.cli --help
cat results_smoke_active/data_validation.csv
```

**Checkpoint (5 min).** In your own words: why would a 95%-accurate random-split result be
misleading? Where in `configs/smoke.yaml` would you change the number of outer folds?

---

## Day 2 — From raw EDF to an EpochBundle (the data layer)

**Read (25 min)** — package `src/eeg_benchmark/data/`
- `epochs.py` first: `EVENT_ID`, `LEFT_RIGHT_RUNS` vs `HANDS_FEET_RUNS`, the `EpochBundle`
  dataclass (the common currency: `X[epochs,channels,time]`, `y`, `groups`, `ch_names`), and
  `epochs_from_raw` (T1→left=0, T2→right=1). Note it **refuses** runs 6/10/14.
- `preprocessing.py`: `preprocess_raw` (standardise names → montage → common-average reference →
  7–30 Hz band-pass) and `build_dataset` (loops subjects, enforces channel order, caches).
- `download.py` and `validation.py` (skim): how EDFs are fetched and turned into the manifest.

**Why.** The band 7–30 Hz targets the mu/beta sensorimotor rhythms; the 1–4 s window skips the
cue transient. `EpochBundle` is the boundary between "MNE world" and "arrays world" — every model
consumes it, which is why tests can inject a synthetic one.

**Do**
```bash
python -c "
from eeg_benchmark.config import Config
from eeg_benchmark.data.preprocessing import load_subject_bundle
b = load_subject_bundle(Config.from_yaml('configs/smoke.yaml'), subject=1)
print('X', b.X.shape, 'classes', set(b.y.tolist()), 'first chans', b.ch_names[:5])
"
```

**Checkpoint.** What shape is `X` and what does each axis mean? Why does `epochs_from_raw` raise on
run 6? Where does the channel *order* get fixed across subjects?

---

## Day 3 — Subject-grouped splits (leakage prevention in code)

**Read (25 min)**
- `src/eeg_benchmark/data/splits.py`: `OuterFold` (+ `assert_disjoint`), `_grouped_kfold`
  (StratifiedGroupKFold with a GroupKFold fallback), and `make_outer_folds`—note the **nested**
  design: every inner fold partitions the complete outer-training subject set. The first inner
  validation fold is also the CNN's epoch-selection holdout.
- `tests/test_subject_splits.py`: the assertions that encode "no subject in two splits".

**Why.** This is Day 1's idea made mechanical. Config-stamped splits are saved below the active
result directory so CSP and the CNN see identical outer-training and test subjects.

**Do**
```bash
python -m eeg_benchmark.cli make-splits --config configs/smoke.yaml
cat results_smoke_active/splits/outer_fold_0.json
python -m pytest tests/test_subject_splits.py -q
```

**Checkpoint.** Given `outer_fold_0.json`, verify by eye that train/val/test subject lists don't
overlap. Why is the inner split taken from outer-train only, never the test subjects?

---

## Day 4 — Baseline: CSP + logistic regression

**Read (25 min)**
- `src/eeg_benchmark/models/csp_logreg.py`: `build_csp_pipeline` (CSP → StandardScaler → LogReg).
  Understand CSP as learning **spatial filters that maximise between-class variance ratio**.
- `src/eeg_benchmark/training/train_csp.py`: `train_csp_fold`—score every configuration on every
  grouped inner fold, select by mean balanced accuracy, refit on the complete outer-training
  population, then predict test once. It saves CSP **patterns** and LogReg coefficients.

**Why.** CSP is supervised, so it's fitted on training subjects only — the same leakage rule as
splits. Patterns (forward model) are preferred over raw filters for interpretation.

**Do**
```bash
python -m eeg_benchmark.cli train-csp --config configs/smoke.yaml
cat results_smoke_active/csp_selected_params.csv
head results_smoke_active/csp_inner_cv.csv # every config × inner-fold score
python -m pytest tests/test_csp_pipeline.py -q
```

**Checkpoint.** Which parameters are in the CSP grid (`configs/smoke.yaml`)? Why must CSP be fit
inside each fold rather than once on all data?

---

## Day 5 — The compact EEGNet-like CNN

**Read (25 min)**
- `src/eeg_benchmark/models/compact_eegnet.py`: `CompactEEGNet` — trace the blocks
  (temporal conv → depthwise spatial conv over channels → ELU/pool/dropout → separable conv →
  global average pool → 2-class head). `count_parameters` must stay < 100k.
- `src/eeg_benchmark/training/train_neural.py`: selection normalisation from the designated
  train subjects, **early stopping on grouped validation balanced accuracy**, then a fresh model,
  optimizer, normaliser, and class weights refit on train + validation for `best_epoch + 1`.

**Why.** Same leakage discipline, now for a neural net: the stopping epoch is selected without
test subjects, while the persisted CNN receives the same outer-training population as CSP.

**Do**
```bash
python -c "
from eeg_benchmark.models.compact_eegnet import CompactEEGNet, count_parameters
m = CompactEEGNet(n_channels=64, n_times=481, kernel_length=64)
print('params:', count_parameters(m))
"
python -m eeg_benchmark.cli train-cnn --config configs/smoke.yaml
python -m pytest tests/test_neural_model.py -q
```

**Checkpoint.** Why must selection normalisation exclude validation subjects? Why is it valid to
recompute final normalisation on train + validation only after the epoch count is frozen?

---

## Day 6 — Evaluation and channel-drop robustness

**Read (25 min)**
- `src/eeg_benchmark/evaluation/`: `metrics.py` (balanced accuracy, macro-F1, ROC-AUC, per-subject
  vs epoch level), `bootstrap.py` (subject-level CIs), `compare.py` (**paired per-subject**
  CNN−CSP difference — the headline result), `latency.py`.
- `src/eeg_benchmark/robustness/`: `channel_masks.py` (deterministic random + regional masks) and
  `channel_drop.py` (impute missing channels with **training** channel means, no retraining).

**Why.** The comparison is paired *per subject* because subjects are the independent unit. The same
masks are reused for both models; imputation never uses test data — leakage again.

**Do**
```bash
python -m eeg_benchmark.cli evaluate     --config configs/smoke.yaml
python -m eeg_benchmark.cli stress-test  --config configs/smoke.yaml
cat results_smoke_active/model_comparison.csv
python -m pytest tests/test_metrics.py tests/test_channel_masks.py -q
```

**Checkpoint.** Why bootstrap over subjects, not epochs? What does a paired-difference CI that
overlaps 0 tell you? Which channels does the "central" regional mask remove, and why those?

---

## Day 7 — Explanations, stability, reporting, and the whole loop

**Read (25 min)**
- `src/eeg_benchmark/explanations/`: `csp_patterns.py`, `occlusion.py` (drop-one-channel importance),
  `integrated_gradients.py` (Captum or built-in fallback), `stability.py` (Spearman / cosine /
  top-k Jaccard between channel-importance vectors — always channel-aligned).
- `src/eeg_benchmark/reporting/`: `figures.py` and `report.py` (reads result CSVs only — no
  hard-coded numbers).
- Finally re-read `src/eeg_benchmark/pipeline.py` top to bottom: now every function should feel
  familiar, and `run_all` is just Days 2→7 in sequence.

**Why.** Explanations are **predictive** attributions, not proof of neural mechanism. Every CNN
seed receives the same first-N-per-subject held-out cohort and targets the true class. Correctness
is annotated, not used to cherry-pick examples. Bootstrap vectors are compared with the full-set
estimate. The report reads only saved artifacts, so absent metrics remain pending.

**Do**
```bash
python -m eeg_benchmark.cli explain --config configs/smoke.yaml
python -m eeg_benchmark.cli report  --config configs/smoke.yaml
open reports_smoke_active/report.md
open reports_smoke_active/figures/
python -m pytest -q
```

**Checkpoint.** Trace one epoch from raw EDF to a row in
`results_smoke_active/predictions.csv`. Why would selecting only correctly predicted epochs
confound a cross-seed stability estimate? What would you change for a 20-subject pilot?

---

## Where to go after the week
- Study the reviewed full V1 summary in `README.md` and compare it with the smoke outputs. Explain
  why smoke evidence establishes pipeline operability while the 106-subject held-out result
  supports inferential interpretation.
- Read `tests/conftest.py` to see how the synthetic fixtures make the whole pipeline testable
  without any download — a good template for your own experiments.
- Open questions and verification status are listed in `PLAN.md`.

---

# Appendix A — Run the whole pipeline yourself, stage by stage (smoke set)

This is the hands-on companion to the week: run each stage **in order** on the 5-subject smoke
set (subjects 1–5 are already downloaded) and *look at the file each stage produces*. That loop —
**run a stage → inspect its output → read the code that made it** — is the fastest way to see how
an ML project is assembled. Every stage here is only seconds on the smoke set.

First, once per terminal:
```bash
cd /path/to/eeg-robustness-benchmark
source .venv/bin/activate
```

Note on output location: with `configs/smoke.yaml` these commands write to
`results_smoke_active/` and `reports_smoke_active/` — a fresh, isolated location. Your earlier
snapshot is preserved untouched in `results_smoke/` (historical evidence). The reviewed full V1
used `configs/full.yaml` and wrote to `results/` and `reports/`; its immutable archive identity is
recorded in `artifacts/full_v1/RUN_PROVENANCE.json`. These locations never mix because each
result directory is stamped with a config hash.
To reset between practice runs: `rm -rf results_smoke_active reports_smoke_active`.

**The whole thing in one shot** (do this once to see it flow, then go back and do it stage by
stage to actually learn):
```bash
python -m eeg_benchmark.cli run-all --config configs/smoke.yaml
```

## Stage-by-stage — what each command does and what to look at

**0. Get the data** — download EDFs (skips any already present, so this is instant for 1–5).
```bash
python -m eeg_benchmark.cli download --config configs/smoke.yaml
ls data/raw/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/       # the raw .edf files
```
Code: `src/eeg_benchmark/data/download.py`.

**1. Validate** — read every EDF, check channels/sfreq, count T1/T2 epochs, write a manifest.
```bash
python -m eeg_benchmark.cli validate-data --config configs/smoke.yaml
column -s, -t < results_smoke_active/data_validation.csv
```
Look for: 64 channels, 160 Hz, balanced T1/T2, 0 rejected, status `ok`.
Code: `src/eeg_benchmark/data/validation.py`.

**2. Prepare** — preprocess (CAR, 7–30 Hz, 1–4 s epochs) and cache the `EpochBundle`; record class
counts and environment metadata.
```bash
python -m eeg_benchmark.cli prepare --config configs/smoke.yaml
column -s, -t < results_smoke_active/class_counts.csv
python -c "import json; p='results_smoke_active/run_metadata.json'; d=json.load(open(p)); print(d['n_epochs'], 'epochs;', d['n_channels'], 'channels')"
```
Peek at the actual array the models will train on:
```bash
python -c "
from eeg_benchmark.config import Config
from eeg_benchmark.data.preprocessing import build_dataset
import numpy as np
b = build_dataset(Config.from_yaml('configs/smoke.yaml'))
print('X shape:', b.X.shape, '= (epochs, channels, time)')
vals, counts = np.unique(b.y, return_counts=True)
print('class counts:', dict(zip(vals.tolist(), counts.tolist())))
"
```
Code: `src/eeg_benchmark/data/preprocessing.py`, `data/epochs.py`.

**3. Split** — build subject-grouped outer/inner folds and save them to JSON.
```bash
python -m eeg_benchmark.cli make-splits --config configs/smoke.yaml
cat results_smoke_active/splits/outer_fold_0.json
```
Code: `src/eeg_benchmark/data/splits.py`. This is the leakage-prevention core — spend time here.

**4. Train the CSP baseline** — grid-search on inner-val, refit, predict test; save models + params.
```bash
python -m eeg_benchmark.cli train-csp --config configs/smoke.yaml
column -s, -t < results_smoke_active/csp_selected_params.csv
column -s, -t < results_smoke_active/csp_inner_cv.csv | head
ls results_smoke_active/models/csp_fold*.joblib
```
Code: `src/eeg_benchmark/training/train_csp.py`, `models/csp_logreg.py`.

**5. Train the CNN** — one model per fold × seed; early stopping; save checkpoints + training curves.
```bash
python -m eeg_benchmark.cli train-cnn --config configs/smoke.yaml
column -s, -t < results_smoke_active/training_curves.csv | head
column -s, -t < results_smoke_active/cnn_selection.csv
ls results_smoke_active/models/cnn_fold*_seed*.pt
```
Watch the log line printing the parameter count and best validation score.
Code: `src/eeg_benchmark/training/train_neural.py`, `models/compact_eegnet.py`.

**6. Evaluate** — merge predictions, compute epoch/subject metrics, paired CNN-vs-CSP, efficiency.
```bash
python -m eeg_benchmark.cli evaluate --config configs/smoke.yaml
column -s, -t < results_smoke_active/subject_metrics.csv
column -s, -t < results_smoke_active/model_comparison.csv
column -s, -t < results_smoke_active/efficiency.csv
```
Code: `src/eeg_benchmark/evaluation/{metrics,bootstrap,compare,latency}.py`.

**7. Stress-test** — apply channel-drop masks to test data only (no retraining); measure robustness.
```bash
python -m eeg_benchmark.cli stress-test --config configs/smoke.yaml
python -c "import pandas as pd; d=pd.read_csv('results_smoke_active/robustness_metrics.csv'); print(d.groupby(['model','condition','fraction'])['balanced_accuracy'].mean().round(3))"
head -c 400 results_smoke_active/channel_masks.json; echo
```
Code: `src/eeg_benchmark/robustness/{channel_masks,channel_drop}.py`.

**8. Explain** — CSP patterns + CNN occlusion/integrated-gradients importance, and their stability.
```bash
python -m eeg_benchmark.cli explain --config configs/smoke.yaml
python -c "import pandas as pd; d=pd.read_csv('results_smoke_active/channel_importance.csv'); print(d[d.model=='cnn'].groupby('channel')['importance'].mean().sort_values(ascending=False).head(8))"
column -s, -t < results_smoke_active/explanation_examples.csv | head
column -s, -t < results_smoke_active/explanation_stability.csv | head
```
Code: `src/eeg_benchmark/explanations/{csp_patterns,occlusion,integrated_gradients,stability}.py`.

**9. Report** — regenerate all figures and the Markdown report from the CSVs above.
```bash
python -m eeg_benchmark.cli report --config configs/smoke.yaml
open reports_smoke_active/report.md
open reports_smoke_active/figures/
```
Code: `src/eeg_benchmark/reporting/{figures,report}.py`.

# Appendix B — Ways to "see what a script does" without changing it

- **Read the logs.** Every stage logs what it's doing to the terminal (timestamps + messages).
  That narration comes from `src/eeg_benchmark/utils/logging.py`.
- **Inspect any smoke CSV** as a table:
  `column -s, -t < results_smoke_active/<file>.csv | less -S`.
- **Peek at arrays/objects interactively**: `python -i -c "from eeg_benchmark.config import Config; cfg=Config.from_yaml('configs/smoke.yaml')"` drops you into a Python shell with `cfg` loaded — then explore with `cfg.<tab>`.
- **Follow one function**: open the file under `src/eeg_benchmark/`, then find where it's called
  from in `pipeline.py`. `pipeline.run_all` is the map of the whole project.
- **Run a single test to see a component in isolation**, e.g. `python -m pytest tests/test_csp_pipeline.py -q -s` (`-s` shows prints).
- **Change a knob and re-run**: edit a value in `configs/smoke.yaml` (e.g. `neural.epochs: 5`),
  re-run the relevant stage, and watch the output change. Nothing is hidden in code — the config
  is the control panel.
