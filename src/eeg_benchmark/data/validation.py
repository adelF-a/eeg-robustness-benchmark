"""Per-subject/run data validation producing a machine-readable manifest.

Writes ``data_validation.csv`` below the active profile's configured result directory,
recording for every subject and run: read status, sampling frequency, channel count, T1/T2
counts, rejected epochs and a reason for any failure. Subjects are never silently dropped.
"""

from __future__ import annotations

import json

import pandas as pd

from ..config import Config
from ..utils.identity import config_hash, ensure_run_manifest
from ..utils.logging import get_logger
from .epochs import epochs_from_raw
from .preprocessing import preprocess_raw
from .runs import resolve_runs, subject_run_edf_map

logger = get_logger(__name__)

MANIFEST_COLUMNS = [
    "subject",
    "run",
    "status",
    "n_channels",
    "sampling_frequency",
    "n_t1",
    "n_t2",
    "n_rejected",
    "reason",
]


def validate_subject_run(cfg: Config, subject: int, run: int, edf_path) -> dict:
    """Validate a single subject/run EDF; returns one manifest row as a dict."""
    import mne

    row = {
        "subject": subject,
        "run": run,
        "status": "ok",
        "n_channels": None,
        "sampling_frequency": None,
        "n_t1": 0,
        "n_t2": 0,
        "n_rejected": 0,
        "reason": "",
    }
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        row["n_channels"] = len(raw.ch_names)
        row["sampling_frequency"] = float(raw.info["sfreq"])

        # Sampling-frequency policy: strict fail (we do not resample). A subject recorded
        # at a different rate than sfreq_expected is excluded and recorded, never treated
        # as valid.
        if abs(row["sampling_frequency"] - cfg.data.sfreq_expected) > 1e-3:
            row["status"] = "fail"
            row["reason"] = (
                f"unexpected sfreq {row['sampling_frequency']} (expected {cfg.data.sfreq_expected})"
            )
            return row

        if len(raw.ch_names) != cfg.data.n_channels_expected:
            row["status"] = "fail"
            row["reason"] = (
                f"n_channels {len(raw.ch_names)} != expected {cfg.data.n_channels_expected}"
            )
            return row

        raw = preprocess_raw(
            raw,
            l_freq=cfg.preprocessing.l_freq,
            h_freq=cfg.preprocessing.h_freq,
            reference=cfg.preprocessing.reference,
        )
        _, y, _, ch_names, _, n_rej = epochs_from_raw(
            raw,
            subject=subject,
            run=run,
            tmin=cfg.preprocessing.tmin,
            tmax=cfg.preprocessing.tmax,
            baseline=cfg.preprocessing.baseline,
            reject=cfg.preprocessing.reject_criteria,
        )
        row["n_channels"] = len(ch_names)
        row["n_t1"] = int((y == 0).sum())
        row["n_t2"] = int((y == 1).sum())
        row["n_rejected"] = int(n_rej)
        if row["n_t1"] == 0 or row["n_t2"] == 0:
            row["status"] = "fail"
            row["reason"] = "missing one class"
    except Exception as exc:  # noqa: BLE001 - record failure reason, never fabricate data
        row["status"] = "fail"
        row["reason"] = f"{type(exc).__name__}: {exc}"
        logger.error("Validation failed for subject %d run %d: %s", subject, run, exc)
    return row


def validate_data(cfg: Config, subjects: list[int] | None = None) -> pd.DataFrame:
    """Validate all subjects/runs and write the active profile's validation CSV."""
    subjects = sorted(subjects or cfg.data.subjects)
    runs = resolve_runs(cfg)  # rejects unsupported runs before any I/O
    ensure_run_manifest(cfg)  # claim/verify the results_dir before writing artifacts

    rows: list[dict] = []
    for subj in subjects:
        try:
            run_map = subject_run_edf_map(cfg, subj)
        except Exception as exc:  # noqa: BLE001
            for run in runs:
                rows.append(
                    {
                        "subject": subj,
                        "run": run,
                        "status": "fail",
                        "n_channels": None,
                        "sampling_frequency": None,
                        "n_t1": 0,
                        "n_t2": 0,
                        "n_rejected": 0,
                        "reason": f"download/read error: {exc}",
                    }
                )
            continue
        # Each EDF is associated with its own run number (no positional assumptions).
        for run in runs:
            rows.append(validate_subject_run(cfg, subj, run, run_map[run]))

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.results_dir / "data_validation.csv"
    df.to_csv(out, index=False)

    # valid_subjects.json is the authoritative survivor set consumed by prepare/run-all.
    # It is stamped with identity so a stale or foreign manifest is rejected on load.
    valid = valid_subjects(df)
    excluded = excluded_subject_reasons(df, subjects)
    manifest = {
        "config_hash": config_hash(cfg),
        "requested_subjects": sorted(int(s) for s in subjects),
        "runs": runs,
        "valid": valid,
        "excluded": excluded.to_dict("records"),
        "n_requested": len(subjects),
    }
    (cfg.paths.results_dir / "valid_subjects.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    logger.info(
        "Wrote validation manifest %s (%d rows); %d valid, %d excluded",
        out,
        len(df),
        len(valid),
        excluded["subject"].nunique() if len(excluded) else 0,
    )
    return df


def valid_subjects(df: pd.DataFrame) -> list[int]:
    """Subjects with no failing run and at least one epoch of each class overall."""
    good: list[int] = []
    for subj, sub in df.groupby("subject"):
        if (sub["status"] == "fail").any():
            continue
        if sub["n_t1"].sum() > 0 and sub["n_t2"].sum() > 0:
            good.append(int(subj))
    return sorted(good)


def excluded_subject_reasons(df: pd.DataFrame, requested: list[int]) -> pd.DataFrame:
    """One row per (excluded subject, reason). Guarantees every requested subject that is
    not valid carries at least one machine-readable reason, so none disappears silently.
    """
    valid = set(valid_subjects(df))
    rows: list[dict] = []
    for subj in requested:
        if int(subj) in valid:
            continue
        sub = df[df["subject"] == subj]
        if sub.empty:
            rows.append({"subject": int(subj), "run": None, "reason": "no validation record"})
            continue
        failing = sub[sub["status"] == "fail"]
        if failing.empty:
            # Passed all runs but lacks both classes overall.
            rows.append(
                {"subject": int(subj), "run": None, "reason": "insufficient class coverage"}
            )
        else:
            for _, r in failing.iterrows():
                rows.append(
                    {
                        "subject": int(subj),
                        "run": int(r["run"]) if pd.notna(r["run"]) else None,
                        "reason": r["reason"] or "fail",
                    }
                )
    return pd.DataFrame(rows, columns=["subject", "run", "reason"])


def load_valid_subjects(cfg: Config) -> dict | None:
    """Return the parsed ``valid_subjects.json`` manifest, or None if absent."""
    p = cfg.paths.results_dir / "valid_subjects.json"
    return json.loads(p.read_text()) if p.exists() else None


def verify_valid_manifest(cfg: Config, manifest: dict, requested: list[int]) -> None:
    """Raise if a validation manifest's identity or requested-subject set does not match.

    Guards against consuming a stale/foreign ``valid_subjects.json`` (e.g. produced by a
    different config or a different subject request).
    """
    active = config_hash(cfg)
    if manifest.get("config_hash") != active:
        raise RuntimeError(
            f"valid_subjects.json was produced for config {manifest.get('config_hash')}, "
            f"not {active}. Re-run validate-data for this profile. No files were modified."
        )
    if set(manifest.get("requested_subjects", [])) != {int(s) for s in requested}:
        raise RuntimeError(
            "valid_subjects.json requested-subject set differs from the active config. "
            "Re-run validate-data for this profile. No files were modified."
        )


def resolve_valid_subjects(
    cfg: Config, subjects: list[int] | None = None
) -> tuple[list[int], pd.DataFrame]:
    """Return ``(valid_subjects, excluded_df)``, running validation if no manifest exists.

    Validation is authoritative: callers build datasets only on the survivor set, and an
    existing manifest is identity-checked before use (never blindly trusted).
    """
    requested = sorted(int(s) for s in (subjects or cfg.data.subjects))
    manifest = load_valid_subjects(cfg)
    if manifest is None:
        validate_data(cfg, subjects)
        manifest = load_valid_subjects(cfg)
    verify_valid_manifest(cfg, manifest, requested)
    excluded = pd.DataFrame(manifest["excluded"], columns=["subject", "run", "reason"])
    return list(manifest["valid"]), excluded
