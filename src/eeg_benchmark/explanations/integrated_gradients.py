"""Integrated gradients attribution for the CNN, aggregated to channel level.

Uses Captum when installed, otherwise a self-contained Riemann-sum implementation.
Attributions are computed along a straight-line path from a documented baseline (zeros
in standardised space == the training channel mean) to the input. Time-resolved
attributions are aggregated to one score per channel by mean absolute value.
"""

from __future__ import annotations

import numpy as np


def _ig_manual(model, X, baseline, target, steps, device):
    import torch

    x = torch.from_numpy(X).float().to(device)
    base = torch.from_numpy(baseline).float().to(device)
    total = torch.zeros_like(x)
    for a in np.linspace(1.0 / steps, 1.0, steps):
        xi = (base + a * (x - base)).clone().requires_grad_(True)
        logits = model(xi)
        sel = logits.gather(1, target.view(-1, 1)).sum()
        grad = torch.autograd.grad(sel, xi)[0]
        total += grad
    attr = (x - base) * total / steps
    return attr.detach().cpu().numpy()


def integrated_gradients(
    model,
    X: np.ndarray,
    device: str = "cpu",
    baseline: str = "zeros",
    steps: int = 50,
    reference: str = "predicted",
    y_true: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (channel_importance [n_channels], time_resolved [n_epochs, n_channels, n_times]).

    channel_importance is the mean over epochs and time of |attribution|.
    """
    import torch

    model.eval()
    base = np.zeros_like(X) if baseline == "zeros" else np.full_like(X, np.mean(X))

    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
    if reference == "true":
        if y_true is None:
            raise ValueError("reference='true' requires y_true")
        target_np = np.asarray(y_true).astype(int)
    else:
        target_np = pred
    target = torch.from_numpy(target_np).long().to(device)

    try:
        from captum.attr import IntegratedGradients

        ig = IntegratedGradients(model)
        x = torch.from_numpy(X).float().to(device)
        b = torch.from_numpy(base).float().to(device)
        attr = ig.attribute(x, baselines=b, target=target, n_steps=steps)
        attr = attr.detach().cpu().numpy()
    except Exception:  # noqa: BLE001 - fall back to manual IG if Captum unavailable
        attr = _ig_manual(model, X, base, target, steps, device)

    # attr shape may be [n, 1, ch, t] or [n, ch, t]; squeeze the conv channel dim.
    attr = np.asarray(attr)
    if attr.ndim == 4:
        attr = attr[:, 0]
    channel_importance = np.abs(attr).mean(axis=(0, 2))
    total = channel_importance.sum()
    if total > 0:
        channel_importance = channel_importance / total
    return channel_importance, attr
