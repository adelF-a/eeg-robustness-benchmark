"""Epoch data structures, event mapping and channel-region definitions.

Event semantics (PhysioNet EEGMMIDB, imagined-movement runs 4, 8, 12 ONLY):

* ``T0`` — rest (discarded here)
* ``T1`` — imagined **left** fist
* ``T2`` — imagined **right** fist

These meanings are specific to the left-vs-right-hand runs. Runs 6, 10, 14 encode
hands-vs-feet and MUST NOT be mixed in; :func:`epochs_from_raw` rejects them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Runs whose T1/T2 mean left-fist / right-fist imagery.
LEFT_RIGHT_RUNS = frozenset({4, 8, 12})
# Runs whose T1/T2 mean both-hands / both-feet imagery — deliberately excluded.
HANDS_FEET_RUNS = frozenset({6, 10, 14})

# Binary label mapping used throughout the project.
EVENT_ID: dict[str, int] = {"left_fist": 0, "right_fist": 1}
# Map raw annotation descriptions to our binary labels for left-vs-right runs.
ANNOT_TO_LABEL: dict[str, int] = {"T1": EVENT_ID["left_fist"], "T2": EVENT_ID["right_fist"]}
LABEL_NAMES = {v: k for k, v in EVENT_ID.items()}


@dataclass
class EpochBundle:
    """In-memory epoched dataset shared across the pipeline.

    Attributes
    ----------
    X : float array [n_epochs, n_channels, n_times]
    y : int array [n_epochs] with values in {0, 1}
    groups : int array [n_epochs] of subject IDs
    ch_names : list[str] channel names, order matches axis 1 of X
    sfreq : sampling frequency (Hz)
    metadata : DataFrame with one row per epoch (subject, run, event, epoch_id)
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    ch_names: list[str]
    sfreq: float
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)
        self.groups = np.asarray(self.groups)
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3D [epochs, channels, time], got shape {self.X.shape}")
        n = self.X.shape[0]
        if not (len(self.y) == len(self.groups) == n):
            raise ValueError("X, y and groups must have matching length along axis 0")
        if self.X.shape[1] != len(self.ch_names):
            raise ValueError("ch_names length must equal X.shape[1]")

    @property
    def n_epochs(self) -> int:
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_times(self) -> int:
        return self.X.shape[2]

    def subset(self, mask: np.ndarray) -> EpochBundle:
        """Return a new bundle for the boolean/integer index ``mask``."""
        md = (
            self.metadata.iloc[mask].reset_index(drop=True) if len(self.metadata) else self.metadata
        )
        return EpochBundle(
            X=self.X[mask],
            y=self.y[mask],
            groups=self.groups[mask],
            ch_names=list(self.ch_names),
            sfreq=self.sfreq,
            metadata=md,
        )

    def subject_mask(self, subjects: list[int]) -> np.ndarray:
        s = {int(x) for x in subjects}
        return np.array([int(g) in s for g in self.groups], dtype=bool)


def epochs_from_raw(
    raw: Any,
    subject: int,
    run: int,
    tmin: float,
    tmax: float,
    baseline: Any | None,
    reject: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], float, int]:
    """Extract T1/T2 epochs from a preprocessed MNE Raw for one run.

    Returns ``(X, y, metadata, ch_names, sfreq, n_rejected)``. Raises ValueError if a
    hands-vs-feet run is passed, since its events have different meaning.
    """
    import mne

    if run in HANDS_FEET_RUNS:
        raise ValueError(
            f"Run {run} is a hands-vs-feet run; refusing to mix it with left-vs-right runs."
        )
    if run not in LEFT_RIGHT_RUNS:
        raise ValueError(
            f"Run {run} is not a recognised left-vs-right imagery run {sorted(LEFT_RIGHT_RUNS)}."
        )

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # Keep only annotations we understand (T1, T2). event_id maps description -> code.
    wanted = {desc: code for desc, code in event_id.items() if desc in ANNOT_TO_LABEL}
    if not wanted:
        raise ValueError(f"No T1/T2 annotations found for subject {subject} run {run}.")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=wanted,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks="eeg",
        preload=True,
        reject=reject,
        reject_by_annotation=True,
        verbose=False,
    )
    n_before = len(events[np.isin(events[:, 2], list(wanted.values()))])
    n_rejected = n_before - len(epochs)

    X = epochs.get_data(copy=True)
    # Map each retained epoch's event code back to our binary label.
    code_to_label = {code: ANNOT_TO_LABEL[desc] for desc, code in wanted.items()}
    y = np.array([code_to_label[c] for c in epochs.events[:, 2]], dtype=np.int64)

    ch_names = list(epochs.ch_names)
    md = pd.DataFrame(
        {
            "subject": subject,
            "run": run,
            "event": [LABEL_NAMES[int(lbl)] for lbl in y],
            "label": y,
            "epoch_id": [f"s{subject:03d}_r{run:02d}_e{i:03d}" for i in range(len(y))],
        }
    )
    return X.astype(np.float32), y, md, ch_names, float(epochs.info["sfreq"]), int(n_rejected)


# ---------------------------------------------------------------------------
# Channel regions — derived from actual (validated) channel names, never assumed.
# ---------------------------------------------------------------------------

_FRONTAL_PREFIXES = ("FP", "AF", "F")
_CENTRAL_PREFIXES = ("FC", "C", "CP")  # motor-region electrodes around C3/Cz/C4


def _num_in_name(name: str) -> int | None:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def _prefix(name: str) -> str:
    return re.match(r"[A-Za-z]+", name).group(0).upper()


def region_channels(ch_names: list[str], region: str) -> list[str]:
    """Return the subset of ``ch_names`` belonging to an anatomical ``region``.

    Regions: ``frontal``, ``central`` (motor), ``left`` (odd-numbered),
    ``right`` (even-numbered). Only names actually present are returned; missing
    expected names are simply absent, never fabricated.
    """
    region = region.lower()
    out: list[str] = []
    for ch in ch_names:
        pref = _prefix(ch)
        num = _num_in_name(ch)
        if region == "frontal":
            if pref in _FRONTAL_PREFIXES:
                out.append(ch)
        elif region == "central":
            if pref in _CENTRAL_PREFIXES:
                out.append(ch)
        elif region == "left":
            if num is not None and num % 2 == 1:
                out.append(ch)
        elif region == "right":
            if num is not None and num % 2 == 0:
                out.append(ch)
        else:
            raise ValueError(f"Unknown region {region!r}")
    return out
