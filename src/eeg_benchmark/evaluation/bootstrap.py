"""Subject-level bootstrap confidence intervals.

Resampling is done over *subjects*, not epochs, because subjects are the independent
statistical unit. This yields honest uncertainty for "how well does this generalise to a
new person".
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 12345
) -> dict[str, float]:
    """Bootstrap mean and CI of a 1-D array of per-subject values (NaNs dropped)."""
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "lo": float(np.quantile(boot, alpha)),
        "hi": float(np.quantile(boot, 1 - alpha)),
        "n": int(len(values)),
    }


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 12345
) -> dict[str, float]:
    """Bootstrap CI of the paired difference (a - b) across subjects.

    ``a`` and ``b`` are aligned per-subject metrics (same subject order). Pairs with a
    NaN on either side are dropped.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    diff = a[mask] - b[mask]
    if len(diff) == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return {
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "lo": float(np.quantile(boot, alpha)),
        "hi": float(np.quantile(boot, 1 - alpha)),
        "n": int(len(diff)),
        "frac_a_better": float(np.mean(diff > 0)),
    }
