"""Apply channel masks at test time using training-derived imputation.

Missing channels are replaced by a neutral value — the per-channel mean computed from
**training** subjects (never from test subjects). This mimics a dead electrode whose
signal is unknown, without leaking test information into the imputation.
"""

from __future__ import annotations

import numpy as np

from .channel_masks import ChannelMask


def train_channel_means(X_train: np.ndarray) -> np.ndarray:
    """Per-channel mean over training epochs/time, shape [n_channels, 1]."""
    return X_train.mean(axis=(0, 2), keepdims=False)[:, None]


def apply_mask(
    X: np.ndarray, ch_names: list[str], mask: ChannelMask, fill: np.ndarray
) -> np.ndarray:
    """Return a copy of X with masked channels replaced by ``fill`` values.

    ``fill`` is shaped [n_channels, 1] (broadcast over time). Only channels present in
    both ``ch_names`` and ``mask.removed`` are affected.
    """
    idx = [ch_names.index(c) for c in mask.removed if c in ch_names]
    Xm = X.copy()
    for i in idx:
        Xm[:, i, :] = fill[i]
    return Xm
