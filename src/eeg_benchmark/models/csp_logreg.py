"""CSP + logistic-regression baseline pipeline.

Common Spatial Patterns (CSP) learn spatial filters that maximise the variance ratio
between the two classes; log-variance of the filtered signals feeds a logistic
regression. CSP is *supervised*, so it is fitted on training subjects only.

Interpretation note: prefer CSP **patterns** (forward model, ``csp.patterns_``) over the
spatial **filters** (``csp.filters_``) for physiological reading — patterns show how
sources project onto the scalp, filters can have misleading signs/scales.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_csp_pipeline(
    n_components: int = 4,
    reg: Any | None = None,
    C: float = 1.0,
    class_weight: Any | None = None,
    random_state: int = 0,
) -> Pipeline:
    """Construct an sklearn Pipeline: CSP -> StandardScaler -> LogisticRegression.

    ``reg`` accepts ``None``, a float shrinkage in (0, 1], or ``"ledoit_wolf"``.
    CSP consumes arrays shaped [epochs, channels, time]; ``log`` variance features are used.
    """
    from mne.decoding import CSP

    csp = CSP(
        n_components=n_components,
        reg=reg,
        log=True,
        norm_trace=False,
        cov_est="concat",
    )
    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=1000,
        solver="liblinear",
        random_state=random_state,
    )
    return Pipeline([("csp", csp), ("scale", StandardScaler()), ("logreg", clf)])
