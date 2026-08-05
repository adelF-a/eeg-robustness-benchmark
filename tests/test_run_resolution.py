"""Strict run resolution and explicit run->EDF association (step 1).

These tests guard against the previous positional-`zip` bug where a filtered run list
was paired against an unfiltered EDF list, silently mislabelling data (e.g. run-6
hands/feet epochs labelled as run 8). No network access: `eegbci.load_data` is mocked.
"""

from __future__ import annotations

import pytest

from eeg_benchmark.config import Config
from eeg_benchmark.data.runs import parse_run_from_edf, resolve_runs, subject_run_edf_map


def _cfg(runs, tmp_path) -> Config:
    return Config.from_dict(
        {
            "name": "t",
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "results_dir": str(tmp_path / "results"),
                "reports_dir": str(tmp_path / "reports"),
            },
            "data": {"subjects": [1], "runs": runs},
            "preprocessing": {"cache": False},
        }
    )


def test_resolve_runs_valid_sorted_deduped(tmp_path):
    assert resolve_runs(_cfg([12, 4, 8, 4], tmp_path)) == [4, 8, 12]


@pytest.mark.parametrize("runs", [[4, 6, 8], [6], [4, 8, 14], [3]])
def test_resolve_runs_rejects_unsupported(runs, tmp_path):
    with pytest.raises(ValueError, match="Unsupported run"):
        resolve_runs(_cfg(runs, tmp_path))


def test_parse_run_from_edf():
    assert parse_run_from_edf("/x/y/S072R08.edf") == 8
    assert parse_run_from_edf("S001R04.edf") == 4
    with pytest.raises(ValueError, match="Cannot parse"):
        parse_run_from_edf("/x/no_run_here.edf")


def test_subject_run_edf_map_is_order_independent(tmp_path, monkeypatch):
    """EDFs returned out of run order must still map to their correct run."""
    import mne.datasets.eegbci as eb

    returned = [
        str(tmp_path / "S001R12.edf"),
        str(tmp_path / "S001R04.edf"),
        str(tmp_path / "S001R08.edf"),
    ]
    calls = {}

    def fake_load_data(subjects, runs, path, update_path, verbose):
        calls["runs"] = list(runs)
        return list(returned)

    monkeypatch.setattr(eb, "load_data", fake_load_data)
    m = subject_run_edf_map(_cfg([4, 8, 12], tmp_path), 1)
    assert set(m) == {4, 8, 12}
    assert m[4].name == "S001R04.edf"
    assert m[8].name == "S001R08.edf"
    assert m[12].name == "S001R12.edf"
    # Only validated runs were ever requested.
    assert sorted(calls["runs"]) == [4, 8, 12]


def test_subject_run_edf_map_rejects_unsupported_before_io(tmp_path, monkeypatch):
    import mne.datasets.eegbci as eb

    def boom(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("load_data called despite unsupported run")

    monkeypatch.setattr(eb, "load_data", boom)
    with pytest.raises(ValueError, match="Unsupported run"):
        subject_run_edf_map(_cfg([4, 6, 8], tmp_path), 1)


def test_subject_run_edf_map_raises_on_missing_run(tmp_path, monkeypatch):
    import mne.datasets.eegbci as eb

    # Return only two of the three requested runs.
    monkeypatch.setattr(
        eb,
        "load_data",
        lambda **k: [str(tmp_path / "S001R04.edf"), str(tmp_path / "S001R08.edf")],
    )
    with pytest.raises(ValueError, match="missing EDF"):
        subject_run_edf_map(_cfg([4, 8, 12], tmp_path), 1)
