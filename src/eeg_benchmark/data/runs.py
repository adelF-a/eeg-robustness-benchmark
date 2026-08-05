"""Strict run resolution and explicit run->EDF association.

Single source of truth for (a) which runs are permitted and (b) how each downloaded
EDF maps to its own run number. Previously a *filtered* run list was paired positionally
against an *unfiltered* EDF list, which could silently mislabel data (e.g. run-6
hands/feet epochs labelled as run 8). Here unsupported runs are rejected *before any
I/O*, and every EDF is associated with the run parsed from its filename rather than by
position.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import Config
from .epochs import LEFT_RIGHT_RUNS

# EEGBCI filenames look like 'S072R08.edf' -> subject 72, run 8.
_EDF_RUN_RE = re.compile(r"[Ss](\d+)[Rr](\d+)")


def resolve_runs(cfg: Config) -> list[int]:
    """Return configured runs, deduplicated and sorted, after validating them.

    Raises ``ValueError`` (before any download/preprocessing) if any run is not a
    recognised left-vs-right imagery run, so a bad config fails loudly rather than
    being silently filtered.
    """
    requested = list(cfg.data.runs)
    unsupported = [r for r in requested if r not in LEFT_RIGHT_RUNS]
    if unsupported:
        raise ValueError(
            f"Unsupported run(s) {sorted(set(unsupported))}: only left-vs-right imagery "
            f"runs {sorted(LEFT_RIGHT_RUNS)} are allowed. Refusing to proceed."
        )
    return sorted(set(requested))


def parse_subject_run_from_edf(path: str | Path) -> tuple[int, int]:
    """Parse ``(subject, run)`` from an EEGBCI EDF filename ('S072R08.edf' -> (72, 8))."""
    m = _EDF_RUN_RE.search(Path(path).stem)
    if not m:
        raise ValueError(f"Cannot parse subject/run from EDF filename: {path}")
    return int(m.group(1)), int(m.group(2))


def parse_run_from_edf(path: str | Path) -> int:
    """Parse the run number from an EEGBCI EDF filename ('S072R08.edf' -> 8)."""
    return parse_subject_run_from_edf(path)[1]


def subject_run_edf_map(cfg: Config, subject: int) -> dict[int, Path]:
    """Return ``{run: edf_path}`` for one subject, keyed by the run each EDF belongs to.

    Downloads via MNE if necessary. Hardened against a loader that returns files for the
    wrong subject, an unexpected run, a duplicate run, or a missing run — each raises
    ``ValueError`` rather than silently corrupting labels. Ordered by :func:`resolve_runs`.
    """
    from mne.datasets import eegbci

    runs = resolve_runs(cfg)
    paths = eegbci.load_data(
        subjects=subject,
        runs=runs,
        path=str(cfg.paths.raw_dir),
        update_path=False,
        verbose=False,
    )
    mapping: dict[int, Path] = {}
    for p in paths:
        subj_f, run_f = parse_subject_run_from_edf(p)
        if subj_f != subject:
            raise ValueError(
                f"EDF {p} is for subject {subj_f}, but subject {subject} was requested."
            )
        if run_f not in runs:
            raise ValueError(
                f"Subject {subject}: unexpected run {run_f} returned (requested {runs})."
            )
        if run_f in mapping:
            raise ValueError(
                f"Subject {subject}: duplicate EDF for run {run_f} ({mapping[run_f]} and {p})."
            )
        mapping[run_f] = Path(p)
    missing = [r for r in runs if r not in mapping]
    if missing:
        raise ValueError(
            f"Subject {subject}: missing EDF(s) for run(s) {missing} "
            f"(found runs {sorted(mapping)})."
        )
    return {r: mapping[r] for r in runs}
