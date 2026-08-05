"""Download PhysioNet EEGBCI EDF files into a configured data directory.

Raw EDFs are never committed to Git (see ``.gitignore``); they live under
``<data_dir>/raw`` via MNE's eegbci loader with an explicit ``path`` so the default
uncontrolled ``~/mne_data`` location is never used.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


def download(cfg: Config, subjects: list[int] | None = None) -> dict[int, list[Path]]:
    """Download EDF files for the requested subjects/runs.

    Returns a mapping ``subject -> [edf paths]`` in resolved-run order. Unsupported runs
    are rejected up front (before any I/O). Subjects that fail to download are logged and
    left empty here; :mod:`eeg_benchmark.data.validation` records them formally.
    """
    from .runs import resolve_runs, subject_run_edf_map

    subjects = subjects or cfg.data.subjects
    runs = resolve_runs(cfg)  # fail fast on unsupported runs, before any download
    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, list[Path]] = {}
    for subj in subjects:
        try:
            run_map = subject_run_edf_map(cfg, subj)
            out[subj] = [run_map[r] for r in runs]
            logger.info("Downloaded subject %d (%d files)", subj, len(out[subj]))
        except Exception as exc:  # noqa: BLE001 - report and continue, do not fabricate
            logger.error("Failed to download subject %d: %s", subj, exc)
            out[subj] = []
    return out
