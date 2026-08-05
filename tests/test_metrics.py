"""Metric correctness on known examples."""

import numpy as np

from eeg_benchmark.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_diff
from eeg_benchmark.evaluation.metrics import compute_metrics, confusion


def test_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, y, y.astype(float))
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_known_confusion_and_balanced_accuracy():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])  # one false positive on class 0
    cm = confusion(y_true, y_pred)
    assert cm.tolist() == [[1, 1], [0, 2]]
    m = compute_metrics(y_true, y_pred)
    # class-0 recall = 1/2, class-1 recall = 2/2 -> balanced acc = 0.75
    assert abs(m["balanced_accuracy"] - 0.75) < 1e-9


def test_bootstrap_ci_contains_mean():
    vals = np.array([0.6, 0.7, 0.8, 0.65, 0.75])
    ci = bootstrap_ci(vals, n_boot=500, seed=0)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["n"] == 5


def test_paired_diff_sign():
    a = np.array([0.8, 0.9, 0.85])
    b = np.array([0.6, 0.7, 0.65])
    d = paired_bootstrap_diff(a, b, n_boot=500, seed=0)
    assert d["mean_diff"] > 0
    assert d["frac_a_better"] == 1.0
