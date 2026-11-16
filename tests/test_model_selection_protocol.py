"""CSP inner CV and CNN validation-holdout selection with fresh train+val refitting.

Synthetic, offline regressions for model selection and final fitting.
"""

from __future__ import annotations

import joblib
import numpy as np
import torch

from eeg_benchmark.data.epochs import EpochBundle
from eeg_benchmark.data.splits import make_outer_folds
from eeg_benchmark.models.compact_eegnet import CompactEEGNet
from eeg_benchmark.training import train_csp as csp_mod
from eeg_benchmark.training import train_neural as nn_mod
from eeg_benchmark.training.reproducibility import set_seed
from eeg_benchmark.training.train_csp import train_csp_fold
from eeg_benchmark.training.train_neural import ChannelStandardizer, train_neural_fold
from eeg_benchmark.utils.identity import register_artifact


class _ZeroCSPPredictor:
    """Minimal persisted CSP stand-in for robustness-statistics tests."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


# --------------------------------------------------------------------------- #
# 1. CSP uses ALL inner folds, not inner[0].
# --------------------------------------------------------------------------- #
def test_csp_scores_every_inner_fold(synthetic_bundle, smoke_config, monkeypatch):
    fold = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)[0]
    assert len(fold.inner_folds) >= 2  # inner CV must have >1 fold to be meaningful

    seen_inner = []
    original = csp_mod._score_config_on_inner_fold

    def spy(bundle, in_tr, in_val, params, seed):
        seen_inner.append(tuple(in_val))
        return original(bundle, in_tr, in_val, params, seed)

    monkeypatch.setattr(csp_mod, "_score_config_on_inner_fold", spy)
    res = train_csp_fold(synthetic_bundle, fold, smoke_config)

    # Every configured inner fold's val set was scored (not just inner_folds[0]).
    inner_val_sets = {tuple(v) for _, v in fold.inner_folds}
    assert inner_val_sets <= set(seen_inner)
    assert res.n_inner_folds == len(fold.inner_folds)
    assert {r["inner_fold"] for r in res.inner_cv} == set(range(len(fold.inner_folds)))


# --------------------------------------------------------------------------- #
# 2. Mean inner score controls selection (not inner[0]).
# --------------------------------------------------------------------------- #
def test_csp_selection_uses_mean_not_first_fold(synthetic_bundle, smoke_config, monkeypatch):
    fold = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)[0]
    assert len(fold.inner_folds) == 2
    # Two configs differ only in n_components (2 vs 4).
    smoke_config.csp.n_components_grid = [2, 4]
    smoke_config.csp.reg_grid = [None]
    smoke_config.csp.C_grid = [1.0]
    smoke_config.csp.class_weight_grid = [None]

    val_index = {tuple(v): j for j, (_, v) in enumerate(fold.inner_folds)}
    # nc=2 wins inner-fold 0 but loses on the mean; nc=4 wins the mean.
    scores = {2: [0.90, 0.50], 4: [0.60, 0.86]}  # means: 0.70 vs 0.73

    def fake(bundle, in_tr, in_val, params, seed):
        return scores[params["n_components"]][val_index[tuple(in_val)]]

    monkeypatch.setattr(csp_mod, "_score_config_on_inner_fold", fake)
    res = train_csp_fold(synthetic_bundle, fold, smoke_config)
    assert res.params["n_components"] == 4  # mean-based selection, not inner[0]
    assert abs(res.val_score - 0.73) < 1e-9


def test_csp_tie_break_is_deterministic_grid_order(synthetic_bundle, smoke_config, monkeypatch):
    fold = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)[0]
    smoke_config.csp.n_components_grid = [2, 4]
    smoke_config.csp.reg_grid = [None]
    smoke_config.csp.C_grid = [1.0]
    smoke_config.csp.class_weight_grid = [None]
    # Identical means -> first configuration in grid order (nc=2) must win, every time.
    monkeypatch.setattr(csp_mod, "_score_config_on_inner_fold", lambda *a, **k: 0.7)
    for _ in range(3):
        res = train_csp_fold(synthetic_bundle, fold, smoke_config)
        assert res.params["n_components"] == 2


def test_csp_failed_config_is_fully_auditable(synthetic_bundle, smoke_config, monkeypatch):
    fold = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)[0]
    smoke_config.csp.n_components_grid = [2, 4]
    smoke_config.csp.reg_grid = [None]
    smoke_config.csp.C_grid = [1.0]
    smoke_config.csp.class_weight_grid = [None]

    def fake(bundle, in_tr, in_val, params, seed):
        if params["n_components"] == 2:
            raise ValueError("degenerate covariance")
        return 0.6

    monkeypatch.setattr(csp_mod, "_score_config_on_inner_fold", fake)
    res = csp_mod.train_csp_fold(synthetic_bundle, fold, smoke_config)

    failed = [r for r in res.inner_cv if r["n_components"] == 2]
    assert len(failed) == len(fold.inner_folds)
    assert failed[0]["fold_status"] == "failed"
    assert all(r["config_status"] == "failed" for r in failed)
    assert all(np.isnan(r["mean_inner_balanced_accuracy"]) for r in failed)
    assert "degenerate covariance" in failed[0]["failure_reason"]

    successful = [r for r in res.inner_cv if r["n_components"] == 4]
    assert len(successful) == len(fold.inner_folds)
    assert all(r["config_status"] == "ok" for r in successful)
    assert all(r["mean_inner_balanced_accuracy"] == 0.6 for r in successful)


# --------------------------------------------------------------------------- #
# 3. CNN refit uses train+val statistics.
# --------------------------------------------------------------------------- #
def test_cnn_refit_uses_train_val_normalization(small_bundle, smoke_config):
    fold = make_outer_folds(small_bundle.y, small_bundle.groups, smoke_config)[0]
    res = train_neural_fold(small_bundle, fold, smoke_config, seed=0, device="cpu")
    trv = small_bundle.subset(small_bundle.subject_mask(fold.outer_train_subjects))
    expected = ChannelStandardizer.fit(trv.X)
    assert np.allclose(res.standardizer.mean, expected.mean)
    assert np.allclose(res.standardizer.std, expected.std)
    # Distinct from train-only stats (proves it isn't the selection standardizer).
    tr_only = ChannelStandardizer.fit(
        small_bundle.subset(small_bundle.subject_mask(fold.train_subjects)).X
    )
    assert not np.allclose(res.standardizer.mean, tr_only.mean)


# --------------------------------------------------------------------------- #
# 4. Refit epoch count is exactly best_epoch + 1.
# --------------------------------------------------------------------------- #
def test_cnn_refit_epoch_count(small_bundle, smoke_config, monkeypatch):
    fold = make_outer_folds(small_bundle.y, small_bundle.groups, smoke_config)[0]
    calls = []
    original = nn_mod._run_training

    def spy(model, Xtr, ytr, cfg, device, max_epochs, val=None):
        calls.append({"max_epochs": max_epochs, "has_val": val is not None})
        return original(model, Xtr, ytr, cfg, device, max_epochs, val=val)

    monkeypatch.setattr(nn_mod, "_run_training", spy)
    res = train_neural_fold(small_bundle, fold, smoke_config, seed=0, device="cpu")

    assert res.refit_epochs == res.best_epoch + 1
    refit_calls = [c for c in calls if not c["has_val"]]
    assert len(refit_calls) == 1
    assert refit_calls[0]["max_epochs"] == res.best_epoch + 1


# --------------------------------------------------------------------------- #
# 5. Refit model is freshly initialized (fresh seed init, not warm-started).
# --------------------------------------------------------------------------- #
def test_cnn_refit_is_freshly_initialized(small_bundle, smoke_config):
    fold = make_outer_folds(small_bundle.y, small_bundle.groups, smoke_config)[0]
    res = train_neural_fold(small_bundle, fold, smoke_config, seed=3, device="cpu")

    set_seed(3)
    fresh = CompactEEGNet(
        n_channels=small_bundle.n_channels,
        n_times=small_bundle.n_times,
        n_classes=2,
        F1=smoke_config.neural.F1,
        D=smoke_config.neural.D,
        kernel_length=smoke_config.neural.kernel_length,
        dropout=smoke_config.neural.dropout,
    )
    fresh_state = fresh.state_dict()
    assert set(res.refit_init_state) == set(fresh_state)
    for k, v in fresh_state.items():
        assert torch.equal(res.refit_init_state[k], v.detach().cpu()), f"param {k} not fresh"


# --------------------------------------------------------------------------- #
# 6. Test subjects are untouched until the final prediction.
# --------------------------------------------------------------------------- #
def _corrupt_test_subjects(bundle, test_subjects):
    b = EpochBundle(
        X=bundle.X.copy(),
        y=bundle.y.copy(),
        groups=bundle.groups.copy(),
        ch_names=list(bundle.ch_names),
        sfreq=bundle.sfreq,
        metadata=bundle.metadata.copy(),
    )
    mask = np.isin(b.groups, list(test_subjects))
    rng = np.random.default_rng(123)
    b.X[mask] = rng.standard_normal(b.X[mask].shape).astype(np.float32) * 5.0
    return b


def test_test_subjects_do_not_affect_training(small_bundle, smoke_config):
    fold = make_outer_folds(small_bundle.y, small_bundle.groups, smoke_config)[0]
    res = train_neural_fold(small_bundle, fold, smoke_config, seed=0, device="cpu")

    corrupted = _corrupt_test_subjects(small_bundle, fold.test_subjects)
    res2 = train_neural_fold(corrupted, fold, smoke_config, seed=0, device="cpu")

    # Corrupting only the test subjects must not change selection, normalization or weights.
    assert res2.best_epoch == res.best_epoch
    assert np.allclose(res2.standardizer.mean, res.standardizer.mean)
    for k in res.state_dict:
        assert torch.allclose(res2.state_dict[k], res.state_dict[k], atol=1e-6), k


# --------------------------------------------------------------------------- #
# 7. Determinism.
# --------------------------------------------------------------------------- #
def test_cnn_training_is_deterministic(small_bundle, smoke_config):
    fold = make_outer_folds(small_bundle.y, small_bundle.groups, smoke_config)[0]
    a = train_neural_fold(small_bundle, fold, smoke_config, seed=1, device="cpu")
    b = train_neural_fold(small_bundle, fold, smoke_config, seed=1, device="cpu")
    assert a.best_epoch == b.best_epoch
    assert np.array_equal(a.test_pred, b.test_pred)
    for k in a.state_dict:
        assert torch.equal(a.state_dict[k], b.state_dict[k])


# --------------------------------------------------------------------------- #
# 8. Fairness + saved metadata (both models use the same outer train+val subjects).
# --------------------------------------------------------------------------- #
def test_saved_selection_and_refit_metadata(small_bundle, smoke_config):
    import json

    import pandas as pd

    from eeg_benchmark import pipeline

    pipeline.prepare(smoke_config, small_bundle)
    pipeline.make_splits(smoke_config, small_bundle)
    pipeline.train_csp(smoke_config, small_bundle)
    pipeline.train_cnn(smoke_config, small_bundle)
    rd = smoke_config.paths.results_dir

    csp = pd.read_csv(rd / "csp_selected_params.csv")
    for col in [
        "mean_inner_balanced_accuracy",
        "n_inner_folds",
        "outer_train_subjects",
        "config_hash",
    ]:
        assert col in csp.columns
    inner = pd.read_csv(rd / "csp_inner_cv.csv")
    assert inner["inner_fold"].nunique() >= 2  # all inner folds recorded
    for col in [
        "config_hash",
        "config_index",
        "config_status",
        "fold_status",
        "mean_inner_balanced_accuracy",
        "failure_reason",
    ]:
        assert col in inner.columns

    cnn = pd.read_csv(rd / "cnn_selection.csv")
    for col in ["best_epoch", "refit_epochs", "outer_train_subjects", "seed", "config_hash"]:
        assert col in cnn.columns
    assert (cnn["refit_epochs"] == cnn["best_epoch"] + 1).all()

    curves = pd.read_csv(rd / "training_curves.csv")
    assert set(curves["phase"]) == {"selection", "refit"}
    assert curves["config_hash"].notna().all()
    assert curves["n_training_subjects"].gt(0).all()
    selection = curves[curves["phase"] == "selection"]
    refit = curves[curves["phase"] == "refit"]
    assert selection["val_balanced_accuracy"].notna().all()
    assert refit["val_balanced_accuracy"].isna().all()

    # Fairness: CSP and CNN train on identical outer train+val subjects per fold.
    csp_sets = dict(zip(csp["fold"], csp["outer_train_subjects"], strict=False))
    cnn_sets = dict(zip(cnn["fold"], cnn["outer_train_subjects"], strict=False))
    for f in csp_sets:
        assert csp_sets[f] == cnn_sets[f]

    # Checkpoint carries selection/refit metadata + identity.
    ckpt = torch.load(rd / "models" / "cnn_fold0_seed0.pt", weights_only=False)
    for key in ["best_epoch", "refit_epochs", "outer_train_subjects", "seed", "config_hash"]:
        assert key in ckpt
    meta = json.loads((rd / "run_metadata.json").read_text())
    assert "validated_subjects" in meta


def test_stress_imputation_uses_outer_train_subjects(small_bundle, smoke_config, monkeypatch):
    """Channel-fill statistics must match the population used to fit final models."""
    from eeg_benchmark import pipeline
    from eeg_benchmark.data.splits import load_folds
    from eeg_benchmark.robustness import channel_drop

    # Make validation subjects visibly different so train-only and train+val means diverge.
    pipeline.prepare(smoke_config, small_bundle)
    pipeline.make_splits(smoke_config, small_bundle)
    folds = load_folds(smoke_config.paths.splits_dir)
    for fold in folds:
        mask = small_bundle.subject_mask(fold.val_subjects)
        small_bundle.X[mask] += 25.0

    models = smoke_config.paths.results_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        joblib.dump(_ZeroCSPPredictor(), models / f"csp_fold{fold.fold}.joblib")
    register_artifact(smoke_config, "csp_models", "test")
    register_artifact(smoke_config, "cnn_checkpoints", "test")

    seen = []
    original = channel_drop.train_channel_means

    def spy(X):
        seen.append(X.copy())
        return original(X)

    monkeypatch.setattr(channel_drop, "train_channel_means", spy)
    pipeline.stress_test(smoke_config, small_bundle)

    assert len(seen) == len(folds)
    for fold, observed in zip(folds, seen, strict=True):
        expected = small_bundle.subset(small_bundle.subject_mask(fold.outer_train_subjects)).X
        train_only = small_bundle.subset(small_bundle.subject_mask(fold.train_subjects)).X
        assert np.array_equal(observed, expected)
        assert not np.isclose(observed.mean(), train_only.mean())
