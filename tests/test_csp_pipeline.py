"""CSP + logistic-regression pipeline fits and learns on synthetic data."""

import numpy as np

from eeg_benchmark.models.csp_logreg import build_csp_pipeline


def test_csp_pipeline_fit_predict_above_chance(synthetic_bundle):
    b = synthetic_bundle
    # Split by subject to keep the evaluation honest even in this unit test.
    train = np.isin(b.groups, [1, 2, 3, 4])
    test = np.isin(b.groups, [5, 6])
    pipe = build_csp_pipeline(n_components=4, C=1.0)
    pipe.fit(b.X[train], b.y[train])
    acc = (pipe.predict(b.X[test]) == b.y[test]).mean()
    # Signal is lateralised and learnable; expect clearly above chance.
    assert acc > 0.7


def test_csp_exposes_filters_and_patterns(synthetic_bundle):
    b = synthetic_bundle
    pipe = build_csp_pipeline(n_components=4)
    pipe.fit(b.X, b.y)
    csp = pipe.named_steps["csp"]
    assert csp.filters_.shape[1] == b.n_channels
    assert csp.patterns_.shape[1] == b.n_channels
    assert pipe.predict_proba(b.X[:5]).shape == (5, 2)
