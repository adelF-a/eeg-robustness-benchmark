"""Hardware and environment metadata capture for reproducibility."""

from __future__ import annotations

import platform
from typing import Any


def get_device(prefer: str = "auto") -> str:
    """Select a torch device string.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. Tests never require an
    accelerator, so CPU is always a valid fallback.
    """
    import torch

    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def environment_metadata() -> dict[str, Any]:
    """Collect Python / library / hardware versions to store with results."""
    meta: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    for mod in ["numpy", "scipy", "pandas", "sklearn", "mne", "torch", "matplotlib"]:
        try:
            m = __import__(mod)
            meta[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - metadata is best-effort
            meta[f"{mod}_version"] = "not-installed"

    try:
        import torch

        meta["device"] = get_device()
        meta["cuda_available"] = torch.cuda.is_available()
        meta["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        if torch.cuda.is_available():
            meta["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        meta["device"] = "cpu"
    return meta
