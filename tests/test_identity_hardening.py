"""Hardening of identity, validation-gating and EDF mapping (post steps 1-3).

Covers: a bare directory must not be claimed if it already holds artifacts; no silent
fallback to all configured subjects; identity-aware validation manifest; artifact-level
identity for predictions; strict sampling-frequency policy; and robust run->EDF mapping.
No network access.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from eeg_benchmark import pipeline
from eeg_benchmark.config import Config
from eeg_benchmark.data import preprocessing, validation
from eeg_benchmark.data.runs import subject_run_edf_map
from eeg_benchmark.utils.identity import (
    config_hash,
    ensure_run_manifest,
    register_artifact,
    require_artifact,
)
from tests.conftest import make_synthetic_raw


def _cfg(tmp_path, subjects=(1, 2, 3), n_channels_expected=16, runs=(4,), sfreq_expected=160.0):
    return Config.from_dict(
        {
            "name": "smoke",
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "results_dir": str(tmp_path / "results"),
                "reports_dir": str(tmp_path / "reports"),
            },
            "data": {
                "subjects": list(subjects),
                "runs": list(runs),
                "n_channels_expected": n_channels_expected,
                "sfreq_expected": sfreq_expected,
            },
            "preprocessing": {"cache": False},
            "splits": {"outer_folds": 2, "inner_folds": 2},
        }
    )


# --------------------------------------------------------------------------- #
# Issue 1: a populated directory must not be silently claimed.
# --------------------------------------------------------------------------- #
def test_manifest_refuses_populated_dir_without_manifest(tmp_path):
    cfg = _cfg(tmp_path)
    (cfg.paths.results_dir / "models").mkdir(parents=True)
    (cfg.paths.results_dir / "predictions_csp.csv").write_text("legacy")  # stray artifact
    with pytest.raises(RuntimeError, match="Migrate"):
        ensure_run_manifest(cfg)


def test_manifest_allows_empty_or_gitkeep_only(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.results_dir.mkdir(parents=True)
    (cfg.paths.results_dir / ".gitkeep").write_text("")
    (cfg.paths.results_dir / "splits").mkdir()  # empty subdir is fine
    h = ensure_run_manifest(cfg)
    assert h == config_hash(cfg)
    assert (cfg.paths.results_dir / "run_manifest.json").exists()


# --------------------------------------------------------------------------- #
# Issue 2: no silent fallback; stages cannot bypass validation.
# --------------------------------------------------------------------------- #
def test_load_bundle_requires_validation_manifest(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(RuntimeError, match="validate-data"):
        pipeline.load_bundle(cfg)  # subjects=None, no manifest


def test_make_splits_cannot_bypass_validation(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(RuntimeError, match="validate-data"):
        pipeline.make_splits(cfg)  # no injected bundle, no validation


# --------------------------------------------------------------------------- #
# Issue 3: identity-aware validation manifest.
# --------------------------------------------------------------------------- #
def _fake_readers(monkeypatch, bad_subject=None, drop_to=15, sfreq_subject=None):
    def fake_map(cfg, subject):
        return {r: Path(f"S{subject:03d}R{r:02d}.edf") for r in cfg.data.runs}

    def fake_read(path, preload=False, verbose=False):
        subj = int(re.search(r"S(\d+)", str(path)).group(1))
        sf = 128.0 if sfreq_subject is not None and subj == sfreq_subject else 160.0
        raw = make_synthetic_raw(run=4, sfreq=sf)
        if bad_subject is not None and subj == bad_subject:
            raw = raw.copy().drop_channels(raw.ch_names[drop_to:])
        return raw

    monkeypatch.setattr(validation, "subject_run_edf_map", fake_map)
    monkeypatch.setattr(preprocessing, "subject_run_edf_map", fake_map)
    monkeypatch.setattr("mne.io.read_raw_edf", fake_read)


def test_valid_subjects_json_is_stamped(tmp_path, monkeypatch):
    _fake_readers(monkeypatch)
    cfg = _cfg(tmp_path, subjects=[1, 2], n_channels_expected=16)
    validation.validate_data(cfg)
    m = json.loads((cfg.paths.results_dir / "valid_subjects.json").read_text())
    assert m["config_hash"] == config_hash(cfg)
    assert m["requested_subjects"] == [1, 2]
    assert m["runs"] == [4]


def test_resolve_valid_subjects_rejects_foreign_manifest(tmp_path, monkeypatch):
    _fake_readers(monkeypatch)
    cfg = _cfg(tmp_path, subjects=[1, 2], n_channels_expected=16)
    cfg.paths.results_dir.mkdir(parents=True)
    ensure_run_manifest(cfg)
    (cfg.paths.results_dir / "valid_subjects.json").write_text(
        json.dumps(
            {
                "config_hash": "wrong",
                "requested_subjects": [1, 2],
                "runs": [4],
                "valid": [1, 2],
                "excluded": [],
                "n_requested": 2,
            }
        )
    )
    with pytest.raises(RuntimeError, match="validate-data"):
        validation.resolve_valid_subjects(cfg)


def test_resolve_valid_subjects_rejects_subject_set_mismatch(tmp_path, monkeypatch):
    _fake_readers(monkeypatch)
    cfg = _cfg(tmp_path, subjects=[1, 2, 3], n_channels_expected=16)
    cfg.paths.results_dir.mkdir(parents=True)
    ensure_run_manifest(cfg)
    (cfg.paths.results_dir / "valid_subjects.json").write_text(
        json.dumps(
            {
                "config_hash": config_hash(cfg),
                "requested_subjects": [1, 2],
                "runs": [4],
                "valid": [1, 2],
                "excluded": [],
                "n_requested": 2,
            }
        )
    )
    with pytest.raises(RuntimeError, match="requested"):
        validation.resolve_valid_subjects(cfg)


# --------------------------------------------------------------------------- #
# Issue 4: artifact-level identity.
# --------------------------------------------------------------------------- #
def test_require_artifact_missing_raises(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.results_dir.mkdir(parents=True)
    ensure_run_manifest(cfg)
    with pytest.raises(RuntimeError, match="producing stage"):
        require_artifact(cfg, "predictions_csp.csv")


def test_register_then_require_ok(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.results_dir.mkdir(parents=True)
    ensure_run_manifest(cfg)
    register_artifact(cfg, "predictions_csp.csv", "train_csp")
    require_artifact(cfg, "predictions_csp.csv")  # no raise


def test_evaluate_refuses_foreign_prediction_csv(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.ensure()
    ensure_run_manifest(cfg)
    rd = cfg.paths.results_dir
    df = pd.DataFrame(
        {
            "subject": [1, 1],
            "run": [4, 4],
            "epoch_id": ["a", "b"],
            "true_label": [0, 1],
            "predicted_label": [0, 1],
            "class_probability": [0.4, 0.6],
            "fold": [0, 0],
            "model": ["csp", "csp"],
            "seed": [0, 0],
            "config_hash": ["deadbeef", "deadbeef"],  # foreign
        }
    )
    df.to_csv(rd / "predictions_csp.csv", index=False)
    register_artifact(cfg, "predictions_csp.csv", "train_csp")
    with pytest.raises(RuntimeError, match="different config"):
        pipeline.evaluate(cfg)


# --------------------------------------------------------------------------- #
# Issue 5: sampling-frequency policy is fail (not silently valid).
# --------------------------------------------------------------------------- #
def test_sfreq_mismatch_fails_and_excludes(tmp_path, monkeypatch):
    _fake_readers(monkeypatch, sfreq_subject=2)  # subject 2 at 128 Hz, expected 160
    cfg = _cfg(tmp_path, subjects=[1, 2], n_channels_expected=16, sfreq_expected=160.0)
    df = validation.validate_data(cfg)
    assert validation.valid_subjects(df) == [1]
    bad = df[(df["subject"] == 2) & (df["status"] == "fail")]
    assert not bad.empty and "sfreq" in bad.iloc[0]["reason"]


# --------------------------------------------------------------------------- #
# Issue 6: run metadata distinguishes requested/validated/excluded.
# --------------------------------------------------------------------------- #
def test_run_metadata_distinguishes_subject_sets(smoke_config, synthetic_bundle):
    pipeline.prepare(smoke_config, synthetic_bundle)
    meta = json.loads((smoke_config.paths.results_dir / "run_metadata.json").read_text())
    subs = sorted(int(s) for s in synthetic_bundle.metadata["subject"].unique())
    assert meta["requested_subjects"] == subs
    assert meta["validated_subjects"] == subs
    assert meta["excluded_subjects"] == []


# --------------------------------------------------------------------------- #
# Issue 7: hardened EDF mapping.
# --------------------------------------------------------------------------- #
def _patch_load_data(monkeypatch, files):
    import mne.datasets.eegbci as eb

    monkeypatch.setattr(eb, "load_data", lambda **k: list(files))


def test_edf_map_rejects_duplicate_run(tmp_path, monkeypatch):
    _patch_load_data(monkeypatch, ["S001R04.edf", "S001R04.edf", "S001R08.edf", "S001R12.edf"])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        subject_run_edf_map(_cfg(tmp_path, runs=[4, 8, 12]), 1)


def test_edf_map_rejects_unexpected_run(tmp_path, monkeypatch):
    _patch_load_data(monkeypatch, ["S001R04.edf", "S001R06.edf", "S001R08.edf", "S001R12.edf"])
    with pytest.raises(ValueError, match="[Uu]nexpected run"):
        subject_run_edf_map(_cfg(tmp_path, runs=[4, 8, 12]), 1)


def test_edf_map_rejects_wrong_subject(tmp_path, monkeypatch):
    _patch_load_data(monkeypatch, ["S002R04.edf", "S001R08.edf", "S001R12.edf"])
    with pytest.raises(ValueError, match="subject"):
        subject_run_edf_map(_cfg(tmp_path, runs=[4, 8, 12]), 1)


# --------------------------------------------------------------------------- #
# Follow-up issue 3: evaluate requires a config_hash column (legacy rejected).
# --------------------------------------------------------------------------- #
def test_evaluate_requires_config_hash_column(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.ensure()
    ensure_run_manifest(cfg)
    rd = cfg.paths.results_dir
    # Registered prediction file, but WITHOUT a config_hash column (legacy shape).
    pd.DataFrame(
        {
            "subject": [1, 1],
            "run": [4, 4],
            "epoch_id": ["a", "b"],
            "true_label": [0, 1],
            "predicted_label": [0, 1],
            "class_probability": [0.4, 0.6],
            "fold": [0, 0],
            "model": ["csp", "csp"],
            "seed": [0, 0],
        }
    ).to_csv(rd / "predictions_csp.csv", index=False)
    register_artifact(cfg, "predictions_csp.csv", "train_csp")
    with pytest.raises(RuntimeError, match="config_hash"):
        pipeline.evaluate(cfg)


# --------------------------------------------------------------------------- #
# Follow-up issue 2: complete CSP artifact identity in stress_test/explain.
# --------------------------------------------------------------------------- #
def test_stress_test_requires_csp_models(tmp_path, small_bundle):
    cfg = _cfg(tmp_path)
    pipeline.make_splits(cfg, small_bundle)
    register_artifact(cfg, "cnn_checkpoints", "train_cnn")  # csp_models NOT registered
    with pytest.raises(RuntimeError, match="csp_models"):
        pipeline.stress_test(cfg, small_bundle)


def test_stress_test_fails_on_missing_fold_csp_model(tmp_path, small_bundle):
    cfg = _cfg(tmp_path)
    pipeline.make_splits(cfg, small_bundle)
    register_artifact(cfg, "cnn_checkpoints", "train_cnn")
    register_artifact(cfg, "csp_models", "train_csp")  # registered, but no joblib on disk
    with pytest.raises(RuntimeError, match="CSP model"):
        pipeline.stress_test(cfg, small_bundle)


def test_explain_requires_csp_artifacts(tmp_path, small_bundle):
    cfg = _cfg(tmp_path)
    pipeline.make_splits(cfg, small_bundle)
    register_artifact(cfg, "cnn_checkpoints", "train_cnn")  # csp_artifacts NOT registered
    with pytest.raises(RuntimeError, match="csp_artifacts"):
        pipeline.explain(cfg, small_bundle)


def test_explain_fails_on_missing_fold_artifact(tmp_path, small_bundle):
    cfg = _cfg(tmp_path)
    pipeline.make_splits(cfg, small_bundle)
    register_artifact(cfg, "cnn_checkpoints", "train_cnn")
    register_artifact(cfg, "csp_artifacts", "train_csp")  # registered, but no npz on disk
    with pytest.raises(RuntimeError, match="CSP artifact"):
        pipeline.explain(cfg, small_bundle)


# --------------------------------------------------------------------------- #
# Follow-up issue 1: active smoke profile writes to a fresh isolated location,
# not the legacy results_smoke evidence dir and not under the full results_dir.
# --------------------------------------------------------------------------- #
def test_smoke_profile_uses_fresh_isolated_paths():
    smoke = Config.from_yaml("configs/smoke.yaml")
    full = Config.from_yaml("configs/full.yaml")
    sr, fr = smoke.paths.results_dir, full.paths.results_dir
    assert sr != Path("results_smoke")  # preserve legacy evidence untouched
    assert sr != fr
    # Not nested under the full profile's results_dir (would trip recursive detection).
    assert not str(sr).startswith(str(fr) + "/")
    assert not str(smoke.paths.reports_dir).startswith(str(full.paths.reports_dir) + "/")
