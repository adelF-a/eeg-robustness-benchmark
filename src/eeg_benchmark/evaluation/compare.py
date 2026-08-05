"""Paired per-subject comparison of CSP vs CNN."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import paired_bootstrap_diff


def paired_subject_table(
    subject_metrics: pd.DataFrame, metric: str = "balanced_accuracy"
) -> pd.DataFrame:
    """Pivot per-subject metrics to one row per subject with CSP and CNN columns.

    CNN seeds are averaged per subject first (so each subject contributes one CNN value).
    Uses the outer-test predictions, where each subject appears in exactly one fold.
    """
    df = subject_metrics.copy()
    agg = (
        df.groupby(["model", "subject"])[metric]
        .mean()
        .reset_index()
        .pivot(index="subject", columns="model", values=metric)
        .reset_index()
    )
    return agg


def compare_models(
    subject_metrics: pd.DataFrame,
    metric: str = "balanced_accuracy",
    model_a: str = "cnn",
    model_b: str = "csp",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 12345,
) -> dict:
    """Bootstrap the paired (model_a - model_b) difference across shared subjects."""
    tab = paired_subject_table(subject_metrics, metric)
    if model_a not in tab.columns or model_b not in tab.columns:
        return {"error": f"missing model column ({model_a} or {model_b})"}
    both = tab.dropna(subset=[model_a, model_b])
    res = paired_bootstrap_diff(
        both[model_a].to_numpy(), both[model_b].to_numpy(), n_boot=n_boot, ci=ci, seed=seed
    )
    res.update(
        {
            "metric": metric,
            "model_a": model_a,
            "model_b": model_b,
            "mean_a": float(np.nanmean(both[model_a])),
            "mean_b": float(np.nanmean(both[model_b])),
        }
    )
    return res
