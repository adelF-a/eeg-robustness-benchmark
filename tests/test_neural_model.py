"""Compact CNN: forward-pass shape, parameter budget, and one-step training."""

import numpy as np
import torch

from eeg_benchmark.models.compact_eegnet import CompactEEGNet, count_parameters
from eeg_benchmark.training.train_neural import train_neural_fold


def test_forward_pass_shape():
    model = CompactEEGNet(n_channels=16, n_times=128, kernel_length=32)
    x = torch.randn(5, 1, 16, 128)
    out = model(x)
    assert out.shape == (5, 2)
    # Also accept [batch, channels, time] without explicit conv dim.
    assert model(torch.randn(5, 16, 128)).shape == (5, 2)


def test_parameter_budget_under_100k():
    model = CompactEEGNet(n_channels=64, n_times=481, kernel_length=64)
    assert count_parameters(model) < 100_000


def test_one_step_training_runs_and_predicts(small_bundle, smoke_config):
    b = small_bundle
    from eeg_benchmark.data.splits import make_outer_folds

    fold = make_outer_folds(b.y, b.groups, smoke_config)[0]
    res = train_neural_fold(b, fold, smoke_config, seed=0, device="cpu")
    assert res.n_params < 100_000
    assert res.test_pred.shape == res.test_proba.shape
    assert set(np.unique(res.test_pred)).issubset({0, 1})
    assert len(res.history["train_loss"]) >= 1
    assert res.standardizer is not None
