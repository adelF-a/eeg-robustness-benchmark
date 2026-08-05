"""Channel-level importance derived from CSP patterns and classifier weights.

The per-component CSP patterns (forward model) are combined with the absolute logistic
regression coefficient of each component into an aggregate channel-importance vector.
This is a *predictive* summary, not a claim about neural sources — the component-level
patterns are always preserved so the aggregation does not hide the original information.
"""

from __future__ import annotations

import numpy as np


def csp_channel_importance(
    patterns: np.ndarray, logreg_coef: np.ndarray, n_channels: int
) -> np.ndarray:
    """Aggregate CSP patterns weighted by |LogReg coefficient| into per-channel scores.

    ``patterns`` has shape [n_components, n_channels]; ``logreg_coef`` has one weight per
    CSP feature/component. Returns a non-negative vector of length ``n_channels`` scaled
    to sum to 1 (so it is comparable across folds).
    """
    patterns = np.asarray(patterns)
    coef = np.abs(np.asarray(logreg_coef).ravel())
    n_comp = min(patterns.shape[0], len(coef))
    imp = np.zeros(n_channels, dtype=float)
    for c in range(n_comp):
        imp += coef[c] * np.abs(patterns[c, :n_channels])
    total = imp.sum()
    return imp / total if total > 0 else imp
