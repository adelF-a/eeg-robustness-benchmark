"""Deterministic channel masks and training-only imputation."""

import numpy as np

from eeg_benchmark.robustness.channel_drop import apply_mask, train_channel_means
from eeg_benchmark.robustness.channel_masks import make_masks
from tests.conftest import SYNTH_CH_NAMES


def test_random_masks_are_deterministic():
    a = make_masks(SYNTH_CH_NAMES, [0.25, 0.5], 2, [], seed=7)
    b = make_masks(SYNTH_CH_NAMES, [0.25, 0.5], 2, [], seed=7)
    assert [m.removed for m in a] == [m.removed for m in b]
    # A different seed generally yields a different selection.
    c = make_masks(SYNTH_CH_NAMES, [0.25], 1, [], seed=8)
    d = make_masks(SYNTH_CH_NAMES, [0.25], 1, [], seed=7)
    assert c[0].removed != d[0].removed or len(c[0].removed) == 0


def test_regional_masks_use_real_channel_names():
    masks = make_masks(SYNTH_CH_NAMES, [], 0, ["central"], seed=0)
    central = next(m for m in masks if m.name == "region_central")
    assert "C3" in central.removed and "C4" in central.removed
    assert all(c in SYNTH_CH_NAMES for c in central.removed)


def test_imputation_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(5.0, 1.0, (30, len(SYNTH_CH_NAMES), 40)).astype(np.float32)
    X_test = rng.normal(-3.0, 1.0, (10, len(SYNTH_CH_NAMES), 40)).astype(np.float32)
    fill = train_channel_means(X_train)  # from TRAIN only
    masks = make_masks(SYNTH_CH_NAMES, [], 0, ["central"], seed=0)
    mask = next(m for m in masks if m.name == "region_central")
    Xm = apply_mask(X_test, SYNTH_CH_NAMES, mask, fill)
    for c in mask.removed:
        i = SYNTH_CH_NAMES.index(c)
        # Masked channel equals the TRAIN mean (~5), never the test mean (~-3).
        assert np.allclose(Xm[:, i, :], fill[i])
        assert abs(Xm[:, i, :].mean() - 5.0) < 0.5
    # Untouched channels are unchanged.
    keep = SYNTH_CH_NAMES.index("Oz")
    assert np.allclose(Xm[:, keep, :], X_test[:, keep, :])
