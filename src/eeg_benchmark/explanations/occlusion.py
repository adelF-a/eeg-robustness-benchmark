"""Channel-occlusion importance for the CNN.

For each channel, replace its signal with a neutral value and measure the drop in the
model's probability for the reference class (predicted or true). A large drop means the
channel mattered for that prediction. This is a model-behaviour explanation, not evidence
of a physiological mechanism.
"""

from __future__ import annotations

import numpy as np


def occlusion_importance(
    model,
    X: np.ndarray,
    device: str = "cpu",
    reference: str = "predicted",
    y_true: np.ndarray | None = None,
    fill: np.ndarray | None = None,
) -> np.ndarray:
    """Per-channel importance averaged over epochs in X.

    Parameters
    ----------
    X : standardised input [n_epochs, n_channels, n_times].
    reference : "predicted" uses each epoch's predicted class; "true" uses ``y_true``.
    fill : neutral per-channel value [n_channels] (default 0.0 == the standardised mean).

    Returns a vector [n_channels] of mean probability drops (higher = more important).
    """
    import torch

    model.eval()
    n_ch = X.shape[1]
    fill = np.zeros(n_ch, dtype=np.float32) if fill is None else np.asarray(fill, dtype=np.float32)

    with torch.no_grad():
        base_logits = model(torch.from_numpy(X).float().to(device))
        base_prob = torch.softmax(base_logits, dim=1).cpu().numpy()
    if reference == "true":
        if y_true is None:
            raise ValueError("reference='true' requires y_true")
        ref_class = np.asarray(y_true).astype(int)
    else:
        ref_class = base_prob.argmax(axis=1)
    base_ref = base_prob[np.arange(len(X)), ref_class]

    importance = np.zeros(n_ch, dtype=float)
    for ch in range(n_ch):
        Xo = X.copy()
        Xo[:, ch, :] = fill[ch]
        with torch.no_grad():
            logits = model(torch.from_numpy(Xo).float().to(device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
        occ_ref = prob[np.arange(len(X)), ref_class]
        importance[ch] = float(np.mean(base_ref - occ_ref))
    return importance
