"""Generated runs stay out of Git; curated reviewed evidence remains eligible."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    [
        "results_smoke/models/fold0.pt",
        "results_smoke/csp_artifacts/fold0.npz",
        "results_smoke_active/models/fold0.joblib",
        "results_smoke_active/ig_fold0_seed0.npz",
        "results_smoke_active/subject_metrics.csv",
        "results/subject_metrics.csv",
        "reports/figures_smoke/confusion.png",
        "reports_smoke_active/figures/confusion.pdf",
        "full_run.log",
        "eeg-benchmark-kaggle.zip",
        ".claude/settings.local.json",
    ],
)
def test_generated_run_artifact_is_ignored(path):
    subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", path],
        cwd=ROOT,
        check=True,
    )


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/full_v1/README.md",
        "artifacts/full_v1/subject_metrics.csv",
        "src/eeg_benchmark/reporting/report.py",
    ],
)
def test_curated_evidence_and_source_remain_trackable(path):
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "--quiet",
            path,
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 1
