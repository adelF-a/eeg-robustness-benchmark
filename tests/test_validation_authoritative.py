"""Validation is authoritative; exclusions always carry machine-readable reasons;
unexpected post-validation preprocessing failures fail loudly with subject/run context.
No network: MNE readers are monkeypatched.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from eeg_benchmark import pipeline
from eeg_benchmark.config import Config
from eeg_benchmark.data import preprocessing, validation
from tests.conftest import make_synthetic_bundle, make_synthetic_raw


def _cfg(tmp_path, subjects, n_channels_expected=16, runs=(4,)) -> Config:
    return Config.from_dict(
        {
            "name": "t",
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "results_dir": str(tmp_path / "results"),
                "reports_dir": str(tmp_path / "reports"),
            },
            "data": {
                "subjects": list(subjects),
                "runs": list(runs),
                "n_channels_expected": n_channels_expected,
            },
            "preprocessing": {"cache": False},
        }
    )


def test_channel_count_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mne.io.read_raw_edf", lambda path, preload=False, verbose=False: make_synthetic_raw(run=4)
    )
    # 16-channel synthetic raw: passes when expected=16, fails when expected=64.
    ok = validation.validate_subject_run(_cfg(tmp_path, [1], 16), 1, 4, "S001R04.edf")
    assert ok["status"] == "ok" and ok["n_t1"] > 0 and ok["n_t2"] > 0

    bad = validation.validate_subject_run(_cfg(tmp_path, [1], 64), 1, 4, "S001R04.edf")
    assert bad["status"] == "fail"
    assert "n_channels 16 != expected 64" in bad["reason"]


def _fake_readers(monkeypatch, bad_subject=None, drop_to=15):
    """Patch the EDF map + reader so subject id is encoded in the filename."""

    def fake_map(cfg, subject):
        from pathlib import Path

        return {r: Path(f"S{subject:03d}R{r:02d}.edf") for r in cfg.data.runs}

    def fake_read(path, preload=False, verbose=False):
        subj = int(re.search(r"S(\d+)", str(path)).group(1))
        raw = make_synthetic_raw(run=4)
        if bad_subject is not None and subj == bad_subject:
            raw = raw.copy().drop_channels(raw.ch_names[drop_to:])
        return raw

    monkeypatch.setattr(validation, "subject_run_edf_map", fake_map)
    monkeypatch.setattr(preprocessing, "subject_run_edf_map", fake_map)
    monkeypatch.setattr("mne.io.read_raw_edf", fake_read)


def test_validate_data_is_authoritative_and_records_exclusions(tmp_path, monkeypatch):
    _fake_readers(monkeypatch, bad_subject=2)  # subject 2 has 15 channels -> fail
    cfg = _cfg(tmp_path, [1, 2, 3], n_channels_expected=16)
    df = validation.validate_data(cfg)

    assert validation.valid_subjects(df) == [1, 3]
    manifest = json.loads((cfg.paths.results_dir / "valid_subjects.json").read_text())
    assert manifest["valid"] == [1, 3]
    excluded = {e["subject"]: e["reason"] for e in manifest["excluded"]}
    assert 2 in excluded and "n_channels" in excluded[2]


def test_no_subject_disappears_without_a_reason(tmp_path, monkeypatch):
    _fake_readers(monkeypatch, bad_subject=2)
    cfg = _cfg(tmp_path, [1, 2, 3], n_channels_expected=16)
    validation.validate_data(cfg)
    valid, excluded = validation.resolve_valid_subjects(cfg)
    covered = set(valid) | {int(s) for s in excluded["subject"]}
    assert covered == {1, 2, 3}


def test_resolve_valid_subjects_creates_manifest_when_absent(tmp_path, monkeypatch):
    _fake_readers(monkeypatch, bad_subject=None)
    cfg = _cfg(tmp_path, [1, 2], n_channels_expected=16)
    assert not (cfg.paths.results_dir / "valid_subjects.json").exists()
    valid, excluded = validation.resolve_valid_subjects(cfg)
    assert valid == [1, 2]
    assert (cfg.paths.results_dir / "valid_subjects.json").exists()


def test_build_dataset_fails_loudly_with_subject_run_context(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(
        preprocessing,
        "subject_run_edf_map",
        lambda cfg, subject: {r: Path(f"S{subject:03d}R{r:02d}.edf") for r in cfg.data.runs},
    )

    def boom(path, preload=False, verbose=False):
        raise ValueError("corrupt EDF")

    monkeypatch.setattr("mne.io.read_raw_edf", boom)
    cfg = _cfg(tmp_path, [7], n_channels_expected=16)
    with pytest.raises(RuntimeError, match=r"subject 7 run 4"):
        preprocessing.build_dataset(cfg, [7])


def test_prepare_consumes_valid_set_and_writes_excluded(tmp_path, monkeypatch):
    from eeg_benchmark.utils.identity import config_hash, ensure_run_manifest

    cfg = _cfg(tmp_path, [1, 2, 3], n_channels_expected=16)
    cfg.paths.ensure()
    ensure_run_manifest(cfg)
    manifest = {
        "config_hash": config_hash(cfg),
        "requested_subjects": [1, 2, 3],
        "runs": [4],
        "valid": [1, 3],
        "excluded": [{"subject": 2, "run": 4, "reason": "n_channels 15 != expected 16"}],
        "n_requested": 3,
    }
    (cfg.paths.results_dir / "valid_subjects.json").write_text(json.dumps(manifest))

    called = {}

    def fake_build(cfg_, subjects):
        called["subjects"] = list(subjects)
        return make_synthetic_bundle(n_subjects=2, epochs_per_class=4, n_times=64)

    monkeypatch.setattr(pipeline, "build_dataset", fake_build)
    pipeline.prepare(cfg)

    assert called["subjects"] == [1, 3]  # built only on survivors
    exc = pd.read_csv(cfg.paths.results_dir / "excluded_subjects.csv")
    assert 2 in exc["subject"].to_numpy()
    assert "n_channels" in exc.iloc[0]["reason"]
