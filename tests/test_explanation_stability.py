"""Explanation-stability metrics return known values on controlled vectors."""

import numpy as np

from eeg_benchmark.explanations.csp_patterns import csp_channel_importance
from eeg_benchmark.explanations.stability import (
    compare_importances,
    cosine_similarity,
    spearman,
    top_k_jaccard,
)
from eeg_benchmark.pipeline import _select_test_indices


def test_identical_vectors_are_maximally_stable():
    v = np.array([0.1, 0.5, 0.2, 0.9, 0.05])
    m = compare_importances(v, v, k=2)
    assert abs(m["spearman"] - 1.0) < 1e-9
    assert abs(m["cosine_similarity"] - 1.0) < 1e-9
    assert m["top_k_jaccard"] == 1.0


def test_reversed_rank_gives_negative_spearman():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    assert abs(spearman(a, b) + 1.0) < 1e-9


def test_top_k_jaccard_known_value():
    a = np.array([9.0, 8.0, 1.0, 0.0])  # top-2 indices {0,1}
    b = np.array([9.0, 0.0, 8.0, 1.0])  # top-2 indices {0,2}
    # intersection {0}, union {0,1,2} -> 1/3
    assert abs(top_k_jaccard(a, b, 2) - 1 / 3) < 1e-9


def test_cosine_orthogonal():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_csp_channel_importance_alignment():
    # 2 components x 3 channels; component 0 loads channel 0, component 1 loads channel 2.
    patterns = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    coef = np.array([1.0, 1.0])
    imp = csp_channel_importance(patterns, coef, n_channels=3)
    assert imp.shape == (3,)
    assert np.isclose(imp.sum(), 1.0)
    assert imp[1] == 0.0  # middle channel unused
    assert imp[2] > imp[0]  # channel 2 has larger loading


def test_explanation_selection_is_deterministic_and_model_independent(small_bundle):
    first = _select_test_indices(small_bundle, n_per_subject=2)
    # Prediction correctness cannot affect a helper that receives only the held-out data.
    changed_labels = small_bundle.subset(np.arange(small_bundle.n_epochs))
    changed_labels.y = 1 - changed_labels.y
    second = _select_test_indices(changed_labels, n_per_subject=2)
    assert np.array_equal(first, second)

    for subject in np.unique(small_bundle.groups):
        chosen = first[small_bundle.groups[first] == subject]
        candidates = np.flatnonzero(small_bundle.groups == subject)
        expected = sorted(
            candidates,
            key=lambda i: (str(small_bundle.metadata.iloc[i]["epoch_id"]), int(i)),
        )[:2]
        assert chosen.tolist() == expected
