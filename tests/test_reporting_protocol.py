"""Regression tests for scientifically honest reporting and timing metadata."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from eeg_benchmark.evaluation.latency import _synchronize_torch_device
from eeg_benchmark.pipeline import _count_csp_parameters
from eeg_benchmark.reporting.figures import (
    _confusion_rate_matrix,
    _importance_aggregation_label,
    fig_class_counts,
)
from eeg_benchmark.reporting.report import _paired_ci_interpretation, build_report


def test_confusion_uses_rates_not_seed_multiplied_counts():
    base = pd.DataFrame(
        {
            "true_label": [0, 0, 1, 1],
            "predicted_label": [0, 1, 1, 1],
            "seed": 0,
        }
    )
    repeated = pd.concat([base.assign(seed=seed) for seed in [0, 1, 2]], ignore_index=True)
    assert np.allclose(_confusion_rate_matrix(base), _confusion_rate_matrix(repeated))
    assert np.allclose(_confusion_rate_matrix(base), [[0.5, 0.5], [0.0, 1.0]])


def test_confusion_gives_each_seed_equal_weight():
    perfect = pd.DataFrame({"true_label": [0, 1], "predicted_label": [0, 1], "seed": 0})
    wrong = pd.DataFrame(
        {
            "true_label": [0, 1] * 10,
            "predicted_label": [1, 0] * 10,
            "seed": 1,
        }
    )
    rates = _confusion_rate_matrix(pd.concat([perfect, wrong], ignore_index=True))
    assert np.allclose(rates, np.full((2, 2), 0.5))


def test_accelerator_synchronization_dispatches_by_device_type():
    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: calls.append("cuda")),
        mps=SimpleNamespace(synchronize=lambda: calls.append("mps")),
    )
    _synchronize_torch_device(fake_torch, "cuda:0")
    _synchronize_torch_device(fake_torch, "mps")
    _synchronize_torch_device(fake_torch, "cpu")
    assert calls == ["cuda", "mps"]


def test_csp_parameter_count_includes_used_filters_and_classifier():
    pipe = SimpleNamespace(
        named_steps={
            "csp": SimpleNamespace(filters_=np.zeros((5, 4)), n_components=2),
            "logreg": SimpleNamespace(coef_=np.zeros((1, 2)), intercept_=np.zeros(1)),
        }
    )
    assert _count_csp_parameters(pipe) == 2 * 4 + 2 + 1


def test_report_warning_is_conditional_and_split_protocol_is_explicit(smoke_config):
    split_file = smoke_config.paths.splits_dir / "outer_fold_0.json"
    split_file.write_text("{}")

    smoke_config.name = "smoke"
    smoke_text = build_report(smoke_config).read_text()
    assert "⚠️ The **smoke** profile" in smoke_text
    assert "grouped inner folds" in smoke_text
    assert str(smoke_config.splits.seed) in smoke_text

    smoke_config.name = "full"
    full_text = build_report(smoke_config).read_text()
    assert "⚠️ The **smoke** profile" not in full_text
    assert "there is no external-dataset replication" in full_text


def test_paired_ci_interpretation_covers_both_directions_and_overlap():
    positive = _paired_ci_interpretation(0.01, 0.05)
    negative = _paired_ci_interpretation(-0.05, -0.01)
    overlap = _paired_ci_interpretation(-0.01, 0.02)

    assert "entirely above 0" in positive
    assert "includes 0" not in positive
    assert "entirely below 0" in negative
    assert "includes 0" in overlap


def test_class_count_figure_width_is_bounded_for_full_cohort(tmp_path):
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()
    figures_dir.mkdir()
    pd.DataFrame(
        {
            "subject": range(1, 110),
            "n_t1": [22] * 109,
            "n_t2": [22] * 109,
        }
    ).to_csv(results_dir / "data_validation.csv", index=False)

    output = fig_class_counts(results_dir, figures_dir)

    import matplotlib.image as mpimg

    assert output is not None
    assert mpimg.imread(output).shape[1] <= 1800


def test_importance_labels_match_saved_estimands():
    csp = pd.DataFrame({"aggregation_population": ["outer_train_refit_model"]})
    cnn = pd.DataFrame({"aggregation_population": ["shared_outer_test_examples"]})

    assert "refit models" in _importance_aggregation_label(csp)
    assert "test subjects" not in _importance_aggregation_label(csp)
    assert "shared held-out examples" in _importance_aggregation_label(cnn)


def test_report_marks_cross_device_latency_as_non_comparable(smoke_config):
    smoke_config.name = "full"
    pd.DataFrame(
        {
            "model": ["cnn", "csp"],
            "n_params": [2162, 391],
            "parameter_definition": [
                "trainable_tensor_entries",
                "used_csp_filters_plus_logreg_coef_intercept",
            ],
            "size_kb": [18.0, 67.3],
            "latency_p50_ms": [0.52, 0.73],
            "latency_p95_ms": [0.62, 0.84],
            "device": ["cuda", "cpu"],
        }
    ).to_csv(smoke_config.paths.results_dir / "efficiency.csv", index=False)

    text = build_report(smoke_config).read_text()

    assert "different devices/backends" in text
    assert "must not be interpreted as an intrinsic model-speed comparison" in text
