"""Integrity checks for the compact reviewed full-V1 evidence package."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "full_v1"
RESULTS = EVIDENCE / "results"
CONFIG_HASH = "4acfb870cf9b"


def test_full_v1_identity_and_cohort_are_consistent():
    provenance = json.loads((EVIDENCE / "RUN_PROVENANCE.json").read_text())
    run_manifest = json.loads((RESULTS / "run_manifest.json").read_text())
    valid_manifest = json.loads((RESULTS / "valid_subjects.json").read_text())

    assert provenance["identity"]["config_hash"] == CONFIG_HASH
    assert run_manifest["config_hash"] == CONFIG_HASH
    assert len(valid_manifest["requested_subjects"]) == 109
    assert len(valid_manifest["valid"]) == 106
    assert {item["subject"] for item in valid_manifest["excluded"]} == {88, 92, 100}


def test_full_v1_outer_splits_are_disjoint_and_cover_each_test_subject_once():
    valid = set(json.loads((RESULTS / "valid_subjects.json").read_text())["valid"])
    all_test_subjects = []

    split_paths = sorted((RESULTS / "splits").glob("outer_fold_*.json"))
    assert len(split_paths) == 5
    for path in split_paths:
        split = json.loads(path.read_text())
        train = set(split["train_subjects"])
        val = set(split["val_subjects"])
        test = set(split["test_subjects"])
        assert split["config_hash"] == CONFIG_HASH
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert train | val | test == valid
        assert set(split["outer_train_subject_ids"]) == train | val
        all_test_subjects.extend(test)

    assert len(all_test_subjects) == len(valid)
    assert set(all_test_subjects) == valid


def test_full_v1_expected_selection_and_stability_coverage():
    csp_cv = pd.read_csv(RESULTS / "csp_inner_cv.csv")
    csp_selected = pd.read_csv(RESULTS / "csp_selected_params.csv")
    cnn_selection = pd.read_csv(RESULTS / "cnn_selection.csv")
    robustness = pd.read_csv(RESULTS / "robustness_metrics.csv")
    importance = pd.read_csv(RESULTS / "channel_importance.csv")
    stability = pd.read_csv(RESULTS / "explanation_stability.csv")

    assert len(csp_cv) == 1080
    assert (csp_cv["fold_status"] == "ok").all()
    assert (csp_cv["config_status"] == "ok").all()
    assert len(csp_selected) == 5
    assert (csp_selected["C"] == 10.0).all()
    assert (csp_selected["n_components"] == 8).sum() == 3
    assert len(cnn_selection) == 25
    assert (cnn_selection["refit_epochs"] == cnn_selection["best_epoch"] + 1).all()
    assert len(robustness) == 22260
    assert len(importance) == 1920
    assert len(stability) == 1335


def test_full_v1_headline_metric_and_corrected_report_wording():
    comparison = pd.read_csv(RESULTS / "model_comparison.csv").set_index("metric")
    balanced = comparison.loc["balanced_accuracy"]
    report = (EVIDENCE / "report.md").read_text()

    assert balanced["n"] == 106
    assert abs(balanced["mean_diff"] - 0.03549028903827919) < 1e-12
    assert abs(balanced["lo"] - 0.021441596284833563) < 1e-12
    assert abs(balanced["hi"] - 0.05125610541871307) < 1e-12
    assert "CI lies entirely above 0" in report
    assert "different devices/backends" in report
    assert "Full-profile outputs require scientific review" not in report


def test_readme_classification_summary_matches_saved_subject_metrics():
    metrics = pd.read_csv(RESULTS / "subject_metrics.csv")
    cnn = metrics.loc[metrics["model"] == "cnn"].groupby("subject").mean(numeric_only=True)
    csp = metrics.loc[metrics["model"] == "csp"].groupby("subject").mean(numeric_only=True)
    readme = " ".join((ROOT / "README.md").read_text().split())

    assert f"**{len(cnn)} participants, {int(cnn['n_epochs'].sum()):,} trials" in readme
    for label, scores in [("Compact EEGNet-like CNN", cnn), ("CSP + logistic regression", csp)]:
        assert f"| {label} | {scores['balanced_accuracy'].mean():.1%} |" in readme
    assert f"**{cnn['recall_left'].mean():.1%} for left imagery**" in readme
    assert f"**{cnn['recall_right'].mean():.1%} for right imagery**" in readme
    assert f"median CNN balanced accuracy was **{cnn['balanced_accuracy'].median():.1%}**" in readme
    below_chance = int((cnn["balanced_accuracy"] < 0.5).sum())
    assert f"**{below_chance} of {len(cnn)} participants**" in readme
