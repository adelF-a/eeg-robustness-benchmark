"""Synthetic end-to-end run of the full pipeline; asserts expected artifacts exist."""

import pandas as pd

from eeg_benchmark import pipeline


def test_run_all_produces_artifacts(smoke_config, synthetic_bundle):
    # Greater than the removed legacy cap of 6: the exact configured count must survive.
    smoke_config.explanations.n_bootstrap_explain = 7
    pipeline.run_all(smoke_config, bundle=synthetic_bundle)
    rd = smoke_config.paths.results_dir

    expected = [
        "predictions.csv",
        "fold_metrics.csv",
        "subject_metrics.csv",
        "model_comparison.csv",
        "robustness_metrics.csv",
        "channel_masks.json",
        "channel_importance.csv",
        "explanation_examples.csv",
        "explanation_stability.csv",
        "efficiency.csv",
        "run_metadata.json",
        "config_used.yaml",
    ]
    for name in expected:
        assert (rd / name).exists(), f"missing artifact {name}"

    # Split files exist and both models produced predictions for every test subject.
    assert list((rd / "splits").glob("outer_fold_*.json"))
    preds = pd.read_csv(rd / "predictions.csv")
    assert set(preds["model"]) == {"csp", "cnn"}
    assert set(preds["true_label"]).issubset({0, 1})

    # Explanation examples are identical across seeds within each fold and are annotated
    # with correctness rather than selected by correctness.
    examples = pd.read_csv(rd / "explanation_examples.csv")
    assert examples["correct"].notna().all()
    for _, fold_examples in examples.groupby("fold"):
        cohorts = fold_examples.groupby("seed")["epoch_id"].apply(tuple)
        assert cohorts.nunique() == 1

    importance = pd.read_csv(rd / "channel_importance.csv")
    cnn_importance = importance[importance["model"] == "cnn"]
    assert set(cnn_importance["reference_class"]) == {"true_label"}
    assert set(cnn_importance["aggregation_population"]) == {"shared_outer_test_examples"}

    stability = pd.read_csv(rd / "explanation_stability.csv")
    bootstrap = stability[stability["condition"] == "bootstrap"]
    assert set(bootstrap["comparison"]) == {"bootstrap_vs_full"}
    assert set(bootstrap["reference_class"]) == {"true_label"}
    assert (
        bootstrap.groupby(["subject_or_fold", "seed_a"]).size()
        == smoke_config.explanations.n_bootstrap_explain
    ).all()
    assert set(bootstrap["bootstrap_replicate"]) == set(
        range(smoke_config.explanations.n_bootstrap_explain)
    )

    # Every subject appears exactly once as a test subject across outer folds (per model).
    csp = preds[preds["model"] == "csp"]
    assert csp["subject"].nunique() == synthetic_bundle.metadata["subject"].nunique()

    # Report renders.
    report_md = smoke_config.paths.reports_dir / "report.md"
    assert report_md.exists()
    assert "Research question" in report_md.read_text()


def test_no_test_subject_in_training(smoke_config, synthetic_bundle):
    """Guard against leakage: test subjects of each fold are absent from its train set."""
    import json

    pipeline.prepare(smoke_config, synthetic_bundle)
    pipeline.make_splits(smoke_config, synthetic_bundle)
    for p in (smoke_config.paths.splits_dir).glob("outer_fold_*.json"):
        d = json.loads(p.read_text())
        assert not set(d["test_subjects"]) & set(d["train_subjects"])
        assert not set(d["test_subjects"]) & set(d["val_subjects"])
