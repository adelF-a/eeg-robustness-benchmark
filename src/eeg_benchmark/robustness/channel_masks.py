"""Deterministic channel-failure masks.

A mask lists which channels are treated as *lost*. The same masks are reused across
models and repeated comparisons for fairness. Random masks are seeded deterministically;
regional masks are derived from the actual channel names (missing names are simply not
included, never invented).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.epochs import region_channels


@dataclass
class ChannelMask:
    name: str  # e.g. "random_0.25_rep0" or "region_central"
    condition: str  # "random" or "regional"
    removed: list[str]  # channel names removed
    fraction: float  # fraction of channels removed


def make_masks(
    ch_names: list[str],
    random_fractions: list[float],
    n_random_masks: int,
    regions: list[str],
    seed: int = 0,
) -> list[ChannelMask]:
    """Build the full deterministic list of masks for an experiment."""
    masks: list[ChannelMask] = []
    n = len(ch_names)
    rng = np.random.default_rng(seed)
    for frac in random_fractions:
        k = int(round(frac * n))
        for rep in range(n_random_masks):
            idx = rng.choice(n, size=k, replace=False)
            removed = sorted(ch_names[i] for i in idx)
            masks.append(
                ChannelMask(
                    name=f"random_{frac:g}_rep{rep}",
                    condition="random",
                    removed=removed,
                    fraction=len(removed) / n,
                )
            )
    for region in regions:
        removed = region_channels(ch_names, region)
        masks.append(
            ChannelMask(
                name=f"region_{region}",
                condition="regional",
                removed=sorted(removed),
                fraction=len(removed) / n if n else 0.0,
            )
        )
    return masks


def masks_to_json(masks: list[ChannelMask]) -> list[dict]:
    return [
        {
            "name": m.name,
            "condition": m.condition,
            "removed": m.removed,
            "fraction": m.fraction,
        }
        for m in masks
    ]
