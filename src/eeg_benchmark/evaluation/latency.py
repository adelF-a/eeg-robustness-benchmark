"""Inference-latency benchmarking with warm-up and percentile reporting.

Latency is measured at batch size 1 (single-trial online BCI setting). Device and backend
are recorded so CPU and accelerator timings are never conflated across models.
"""

from __future__ import annotations

import time

import numpy as np


def _synchronize_torch_device(torch_module, device: str) -> None:
    """Wait for queued accelerator work before recording elapsed wall time."""
    device_type = str(device).split(":", maxsplit=1)[0]
    if device_type == "cuda":
        torch_module.cuda.synchronize()
    elif device_type == "mps":
        torch_module.mps.synchronize()


def benchmark_torch_latency(
    model,
    input_shape: tuple[int, ...],
    device: str = "cpu",
    n_warmup: int = 10,
    n_repeats: int = 200,
) -> dict[str, float]:
    """Time a torch model's forward pass; returns p50/p95 in milliseconds."""
    import torch

    model.eval().to(device)
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        _synchronize_torch_device(torch, device)
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model(x)
            _synchronize_torch_device(torch, device)
            times.append((time.perf_counter() - t0) * 1e3)
    times = np.array(times)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
        "device": device,
        "n_repeats": n_repeats,
        "batch_size": input_shape[0],
    }


def benchmark_callable_latency(fn, x, n_warmup: int = 10, n_repeats: int = 200) -> dict[str, float]:
    """Time an arbitrary predict callable (e.g. sklearn pipeline) on a single sample."""
    for _ in range(n_warmup):
        fn(x)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1e3)
    times = np.array(times)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
        "device": "cpu",
        "n_repeats": n_repeats,
        "batch_size": 1,
    }
