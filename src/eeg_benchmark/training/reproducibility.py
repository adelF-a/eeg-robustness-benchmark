"""Deterministic seeding across Python, NumPy and PyTorch."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed all RNGs used in the project.

    PyTorch determinism is best-effort: some CUDA/MPS kernels are not fully
    deterministic, so exact bit-reproducibility across devices is not guaranteed.
    We still seed everything and enable deterministic algorithms where supported.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
