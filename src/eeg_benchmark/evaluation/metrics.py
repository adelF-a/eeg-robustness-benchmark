"""Classification metrics at epoch and per-subject level."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> dict[str, float]:
    """Compute a standard set of binary-classification metrics.

    ``y_score`` is the probability of the positive class (label 1); if omitted ROC-AUC
    is NaN. Metrics are robust to a single-class subset (AUC/recall become NaN there).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, float] = {}
    out["accuracy"] = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    out["balanced_accuracy"] = (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
    # Class-wise recall (sensitivity). Class 0 = left fist, class 1 = right fist.
    out["recall_left"] = float(
        recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
    )
    out["recall_right"] = float(
        recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
    )
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix with fixed label order [0, 1]."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def per_subject_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for each (model, seed, fold, subject) group in a predictions frame.

    ``preds`` must contain columns: subject, true_label, predicted_label,
    class_probability, fold, model, seed.
    """
    rows = []
    keys = ["model", "seed", "fold", "subject"]
    for key, sub in preds.groupby(keys):
        m = compute_metrics(
            sub["true_label"].to_numpy(),
            sub["predicted_label"].to_numpy(),
            sub["class_probability"].to_numpy(),
        )
        m.update(dict(zip(keys, key, strict=True)))
        m["n_epochs"] = len(sub)
        rows.append(m)
    return pd.DataFrame(rows)


def fold_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Epoch-level metrics for each (model, seed, fold) group."""
    rows = []
    keys = ["model", "seed", "fold"]
    for key, sub in preds.groupby(keys):
        m = compute_metrics(
            sub["true_label"].to_numpy(),
            sub["predicted_label"].to_numpy(),
            sub["class_probability"].to_numpy(),
        )
        m.update(dict(zip(keys, key, strict=True)))
        m["n_epochs"] = len(sub)
        rows.append(m)
    return pd.DataFrame(rows)
