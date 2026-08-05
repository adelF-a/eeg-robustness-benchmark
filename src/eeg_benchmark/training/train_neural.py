"""Train the compact EEGNet-like model for one outer fold and seed.

Two clearly separated phases:

1. **Selection** — train on the outer-*train* subjects and early-stop on a grouped
   *validation holdout* (``fold.val_subjects``). This holdout is used *only* to pick the
   epoch count; it is a single grouped holdout, not inner cross-validation. ``best_epoch``
   is recorded zero-based.
2. **Refit** — reinitialise a *fresh* model (same seed) and train on the outer
   train+validation subjects for exactly ``best_epoch + 1`` epochs, with normalisation and
   class weights recomputed from train+validation. No optimiser state or weights are
   carried over from the selection model.

The refitted model is evaluated once on the untouched outer-test subjects. The outer-test
set never influences training, normalisation, early stopping or epoch selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Config
from ..data.epochs import EpochBundle
from ..data.splits import OuterFold
from ..evaluation.metrics import balanced_accuracy_score, f1_score
from ..models.compact_eegnet import CompactEEGNet, count_parameters
from ..training.reproducibility import set_seed
from ..utils.hardware import get_device
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChannelStandardizer:
    """Per-channel z-scoring using training-set statistics (saved with the model)."""

    mean: np.ndarray  # [n_channels]
    std: np.ndarray  # [n_channels]

    @classmethod
    def fit(cls, X: np.ndarray) -> ChannelStandardizer:
        mean = X.mean(axis=(0, 2))
        std = X.std(axis=(0, 2))
        std[std < 1e-8] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean[None, :, None]) / self.std[None, :, None]).astype(np.float32)


@dataclass
class NeuralFoldResult:
    fold: int
    seed: int
    test_pred: np.ndarray
    test_proba: np.ndarray
    test_subjects: list[int]
    n_params: int
    best_epoch: int  # zero-based epoch chosen on the validation holdout
    best_val_score: float
    refit_epochs: int  # exactly best_epoch + 1
    train_subjects: list[int]
    val_subjects: list[int]
    outer_train_subjects: list[int]
    train_time_s: float
    history: dict[str, list[float]] = field(default_factory=dict)
    refit_history: dict[str, list[float]] = field(default_factory=dict)
    state_dict: Any = None  # refitted model
    standardizer: ChannelStandardizer | None = None  # train+val statistics
    refit_init_state: Any = None  # refit model weights *before* training (fresh init)


def _loaders(X, y, batch_size, shuffle, device):
    import torch

    ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _predict(model, X, device, batch_size=256):
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            logits = model(xb)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    proba = np.concatenate(probs)
    return (proba >= 0.5).astype(int), proba


def _build_model(cfg: Config, bundle: EpochBundle, device):
    return CompactEEGNet(
        n_channels=bundle.n_channels,
        n_times=bundle.n_times,
        n_classes=2,
        F1=cfg.neural.F1,
        D=cfg.neural.D,
        kernel_length=cfg.neural.kernel_length,
        dropout=cfg.neural.dropout,
    ).to(device)


def _class_weight(y, device):
    import torch

    classes, counts = np.unique(y, return_counts=True)
    return torch.tensor((counts.sum() / (len(classes) * counts)).astype(np.float32), device=device)


def _run_training(model, Xtr, ytr, cfg, device, max_epochs, val=None):
    """Train ``model`` in place. With ``val=(Xval, yval)`` early-stops on validation
    balanced accuracy and restores the best weights; without it trains exactly
    ``max_epochs`` epochs. Returns (history, best_epoch, best_val_score)."""
    import torch

    criterion = torch.nn.CrossEntropyLoss(weight=_class_weight(ytr, device))
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.neural.lr, weight_decay=cfg.neural.weight_decay
    )
    loader = _loaders(Xtr, ytr, cfg.neural.batch_size, True, device)

    history = {"train_loss": [], "val_balanced_accuracy": [], "val_macro_f1": []}
    best_val, best_epoch, best_state, wait = -np.inf, -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        history["train_loss"].append(running / len(Xtr))

        if val is not None:
            Xval, yval = val
            val_pred, _ = _predict(model, Xval, device)
            val_bacc = (
                float(balanced_accuracy_score(yval, val_pred)) if len(np.unique(yval)) > 1 else 0.0
            )
            val_f1 = float(f1_score(yval, val_pred, average="macro", zero_division=0))
            history["val_balanced_accuracy"].append(val_bacc)
            history["val_macro_f1"].append(val_f1)
            monitor = (
                val_bacc if cfg.neural.early_stopping_metric == "balanced_accuracy" else val_f1
            )
            if monitor > best_val:
                best_val, best_epoch, wait = monitor, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg.neural.patience:
                    break

    if val is not None and best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch, float(best_val)


def train_neural_fold(
    bundle: EpochBundle, fold: OuterFold, cfg: Config, seed: int, device: str | None = None
) -> NeuralFoldResult:
    """Select an epoch count on the validation holdout, then refit fresh on train+val."""
    import time

    device = device or get_device()
    tr = bundle.subset(bundle.subject_mask(fold.train_subjects))
    va = bundle.subset(bundle.subject_mask(fold.val_subjects))
    te = bundle.subset(bundle.subject_mask(fold.test_subjects))
    trv = bundle.subset(bundle.subject_mask(fold.outer_train_subjects))

    t0 = time.perf_counter()

    # ---- Phase 1: selection (train on tr, early-stop on the val holdout) ----
    set_seed(seed)
    std_sel = ChannelStandardizer.fit(tr.X)  # training-subject stats only
    model_sel = _build_model(cfg, bundle, device)
    n_params = count_parameters(model_sel)
    history, best_epoch, best_val = _run_training(
        model_sel,
        std_sel.transform(tr.X),
        tr.y,
        cfg,
        device,
        max_epochs=cfg.neural.epochs,
        val=(std_sel.transform(va.X), va.y),
    )
    if best_epoch < 0:  # no epoch improved (e.g. degenerate val); fall back to 1 epoch
        best_epoch = 0
    refit_epochs = best_epoch + 1

    # ---- Phase 2: refit a FRESH model on train+val for exactly best_epoch+1 epochs ----
    set_seed(seed)  # fresh, reproducible init; independent of selection RNG usage
    std_refit = ChannelStandardizer.fit(trv.X)  # recompute from train+val
    model_refit = _build_model(cfg, bundle, device)
    refit_init_state = {k: v.detach().cpu().clone() for k, v in model_refit.state_dict().items()}
    refit_history, _, _ = _run_training(
        model_refit,
        std_refit.transform(trv.X),
        trv.y,  # class weights recomputed from train+val inside _run_training
        cfg,
        device,
        max_epochs=refit_epochs,
        val=None,  # no early stopping, no test, fixed epoch count
    )

    # ---- Evaluate the refitted model once on the untouched outer-test subjects ----
    test_pred, test_proba = _predict(model_refit, std_refit.transform(te.X), device)
    state_dict = {k: v.detach().cpu().clone() for k, v in model_refit.state_dict().items()}

    logger.info(
        "CNN fold %d seed %d: best_epoch=%d (0-based) -> refit %d epochs on %d train+val subjects",
        fold.fold,
        seed,
        best_epoch,
        refit_epochs,
        len(fold.outer_train_subjects),
    )
    return NeuralFoldResult(
        fold=fold.fold,
        seed=seed,
        test_pred=test_pred,
        test_proba=test_proba,
        test_subjects=list(fold.test_subjects),
        n_params=n_params,
        best_epoch=best_epoch,
        best_val_score=best_val,
        refit_epochs=refit_epochs,
        train_subjects=list(fold.train_subjects),
        val_subjects=list(fold.val_subjects),
        outer_train_subjects=list(fold.outer_train_subjects),
        train_time_s=float(time.perf_counter() - t0),
        history=history,
        refit_history=refit_history,
        state_dict=state_dict,
        standardizer=std_refit,
        refit_init_state=refit_init_state,
    )
