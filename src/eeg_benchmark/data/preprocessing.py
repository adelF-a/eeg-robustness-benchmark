"""Deterministic EEG preprocessing.

Pipeline (all parameters configurable in ``configs/*.yaml``):

1. Standardise EEGBCI channel names (``FCz`` etc.) and attach the standard 10-05 montage.
2. Common-average reference (``average``) — a neutral, subject-agnostic reference.
3. Band-pass 7-30 Hz — captures the mu (~8-13 Hz) and beta (~13-30 Hz) sensorimotor
   rhythms whose event-related desynchronisation encodes motor imagery. FIR filtering
   respects annotated bad segments / recording edges.

No baseline correction is applied: common-average referencing plus a narrow band already
remove slow drifts, and baseline windows in continuous imagery runs are ambiguous.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..utils.logging import get_logger
from .epochs import EpochBundle, epochs_from_raw
from .runs import resolve_runs, subject_run_edf_map

logger = get_logger(__name__)


def preprocess_raw(raw: Any, l_freq: float, h_freq: float, reference: str) -> Any:
    """Standardise channels, set montage, reference and band-pass filter a Raw in place.

    Returns the same (mutated, preloaded) Raw for convenience.
    """
    import mne
    from mne.datasets import eegbci

    raw.load_data(verbose=False)
    # EEGBCI EDF channel names are non-standard (e.g. 'Fc5.'); fix to 10-05 names.
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn", verbose=False)

    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
    elif reference and reference != "none":
        raw.set_eeg_reference([reference], verbose=False)

    # FIR band-pass; skip_by_annotation avoids filtering across boundary/bad edges.
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        phase="zero",
        fir_design="firwin",
        skip_by_annotation="edge",
        verbose=False,
    )
    return raw


def _preproc_hash(cfg: Config, subjects: list[int]) -> str:
    """Stable hash of preprocessing-relevant settings, so caches invalidate on change."""
    key = {
        "subjects": sorted(subjects),
        "runs": sorted(cfg.data.runs),
        "l_freq": cfg.preprocessing.l_freq,
        "h_freq": cfg.preprocessing.h_freq,
        "reference": cfg.preprocessing.reference,
        "tmin": cfg.preprocessing.tmin,
        "tmax": cfg.preprocessing.tmax,
        "baseline": cfg.preprocessing.baseline,
        "reject": cfg.preprocessing.reject_criteria,
    }
    blob = json.dumps(key, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def load_subject_bundle(cfg: Config, subject: int) -> EpochBundle:
    """Load, preprocess and epoch all runs for one subject into an EpochBundle."""
    runs = resolve_runs(cfg)
    run_map = subject_run_edf_map(cfg, subject)
    Xs, ys, mds = [], [], []
    ch_names: list[str] | None = None
    sfreq: float | None = None
    import mne

    for run in runs:
        edf = run_map[run]
        try:
            raw = mne.io.read_raw_edf(edf, preload=False, verbose=False)
            raw = preprocess_raw(
                raw,
                l_freq=cfg.preprocessing.l_freq,
                h_freq=cfg.preprocessing.h_freq,
                reference=cfg.preprocessing.reference,
            )
            X, y, md, names, sf, _ = epochs_from_raw(
                raw,
                subject=subject,
                run=run,
                tmin=cfg.preprocessing.tmin,
                tmax=cfg.preprocessing.tmax,
                baseline=cfg.preprocessing.baseline,
                reject=cfg.preprocessing.reject_criteria,
            )
        except Exception as exc:  # noqa: BLE001 - fail loudly with context, never skip
            raise RuntimeError(
                f"Preprocessing failed for subject {subject} run {run}: {exc}"
            ) from exc
        if ch_names is None:
            ch_names, sfreq = names, sf
        elif names != ch_names:
            # Reorder to the reference channel order; refuse if channel sets differ.
            if set(names) != set(ch_names):
                raise ValueError(f"Subject {subject} run {run} channel set differs from reference.")
            order = [names.index(c) for c in ch_names]
            X = X[:, order, :]
        Xs.append(X)
        ys.append(y)
        mds.append(md)

    if not Xs:
        raise ValueError(f"No usable runs for subject {subject}.")
    return EpochBundle(
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0),
        groups=np.full(sum(len(y) for y in ys), subject, dtype=np.int64),
        ch_names=ch_names,
        sfreq=float(sfreq),
        metadata=pd.concat(mds, ignore_index=True),
    )


def build_dataset(cfg: Config, subjects: list[int] | None = None) -> EpochBundle:
    """Build (or load cached) an EpochBundle for the given subjects.

    Channel order is fixed to the first subject's order and enforced for all others.
    Cache files live under ``data/processed`` keyed by a preprocessing hash so changing
    the filter band, window or reference invalidates them automatically.
    """
    subjects = subjects or cfg.data.subjects
    h = _preproc_hash(cfg, subjects)
    cache = cfg.paths.processed_dir / f"bundle_{h}.npz"
    if cfg.preprocessing.cache and cache.exists():
        logger.info("Loading cached epochs from %s", cache)
        data = np.load(cache, allow_pickle=True)
        md = pd.DataFrame(data["metadata"], columns=list(data["metadata_cols"]))
        return EpochBundle(
            X=data["X"],
            y=data["y"],
            groups=data["groups"],
            ch_names=list(data["ch_names"]),
            sfreq=float(data["sfreq"]),
            metadata=md,
        )

    bundles = []
    ref_names: list[str] | None = None
    for subj in subjects:
        b = load_subject_bundle(cfg, subj)
        if ref_names is None:
            ref_names = b.ch_names
        elif b.ch_names != ref_names:
            order = [b.ch_names.index(c) for c in ref_names]
            b = EpochBundle(b.X[:, order, :], b.y, b.groups, ref_names, b.sfreq, b.metadata)
        bundles.append(b)

    bundle = EpochBundle(
        X=np.concatenate([b.X for b in bundles], axis=0),
        y=np.concatenate([b.y for b in bundles], axis=0),
        groups=np.concatenate([b.groups for b in bundles], axis=0),
        ch_names=ref_names,
        sfreq=bundles[0].sfreq,
        metadata=pd.concat([b.metadata for b in bundles], ignore_index=True),
    )
    if cfg.preprocessing.cache:
        cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            X=bundle.X,
            y=bundle.y,
            groups=bundle.groups,
            ch_names=np.array(bundle.ch_names),
            sfreq=bundle.sfreq,
            metadata=bundle.metadata.to_numpy(),
            metadata_cols=np.array(list(bundle.metadata.columns)),
        )
        logger.info("Cached epochs to %s", cache)
    return bundle
