"""Figure generation from saved result CSVs.

Every figure is produced only from result files on disk (no hard-coded numbers). Figures
that lack their source file are skipped with a warning, so partial runs still produce
whatever is available.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Missing %s; skipping figure", path)
        return None
    return pd.read_csv(path)


def fig_class_counts(results_dir: Path, figures_dir: Path) -> Path | None:
    df = _read(results_dir / "data_validation.csv")
    if df is None:
        return None
    plt = _mpl()
    agg = df.groupby("subject")[["n_t1", "n_t2"]].sum()
    n_subjects = len(agg)
    width = min(14.0, max(6.0, 0.14 * n_subjects))
    ax = agg.plot(kind="bar", figsize=(width, 4.5))
    if n_subjects:
        step = max(1, math.ceil(n_subjects / 20))
        positions = list(range(0, n_subjects, step))
        if positions[-1] != n_subjects - 1:
            positions.append(n_subjects - 1)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(agg.index[i]) for i in positions])
    ax.set_xlabel("Subject")
    ax.set_ylabel("Epoch count")
    ax.set_title("Class counts per subject (T1=left, T2=right)")
    ax.legend(["left fist", "right fist"])
    out = figures_dir / "class_counts.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def fig_subject_scatter(
    results_dir: Path, figures_dir: Path, metric="balanced_accuracy"
) -> Path | None:
    df = _read(results_dir / "subject_metrics.csv")
    if df is None:
        return None
    plt = _mpl()
    pivot = df.groupby(["model", "subject"])[metric].mean().unstack("model")
    if "csp" not in pivot or "cnn" not in pivot:
        logger.warning("subject_metrics missing csp/cnn columns; skipping scatter")
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pivot["csp"], pivot["cnn"], alpha=0.7)
    lims = [0, 1]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"CSP {metric}")
    ax.set_ylabel(f"CNN {metric}")
    ax.set_title("Per-subject CSP vs CNN (held-out test subjects)")
    out = figures_dir / "subject_scatter.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def fig_paired_diff(
    results_dir: Path, figures_dir: Path, metric="balanced_accuracy"
) -> Path | None:
    df = _read(results_dir / "subject_metrics.csv")
    if df is None:
        return None
    plt = _mpl()
    pivot = df.groupby(["model", "subject"])[metric].mean().unstack("model")
    if "csp" not in pivot or "cnn" not in pivot:
        return None
    diff = (pivot["cnn"] - pivot["csp"]).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diff, bins=15, edgecolor="k", alpha=0.8)
    ax.axvline(0, color="k", ls="--")
    ax.axvline(diff.mean(), color="r", ls="-", label=f"mean={diff.mean():.3f}")
    ax.set_xlabel(f"CNN − CSP {metric} (per subject)")
    ax.set_ylabel("Subjects")
    ax.set_title("Paired per-subject difference")
    ax.legend()
    out = figures_dir / "paired_difference.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def fig_subject_distribution(
    results_dir: Path, figures_dir: Path, metric="balanced_accuracy"
) -> Path | None:
    df = _read(results_dir / "subject_metrics.csv")
    if df is None:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, sub in df.groupby("model"):
        vals = sub.groupby("subject")[metric].mean().dropna()
        ax.hist(vals, bins=12, alpha=0.5, label=str(model))
    ax.set_xlabel(metric)
    ax.set_ylabel("Subjects")
    ax.set_title("Per-subject performance distribution")
    ax.legend()
    out = figures_dir / "subject_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def _confusion_rate_matrix(df: pd.DataFrame):
    """Mean row-normalized confusion rates, giving every seed equal weight."""
    import numpy as np
    from sklearn.metrics import confusion_matrix

    seed_groups = df.groupby("seed") if "seed" in df else [(0, df)]
    matrices = [
        confusion_matrix(sub["true_label"], sub["predicted_label"], labels=[0, 1], normalize="true")
        for _, sub in seed_groups
    ]
    return np.mean(matrices, axis=0)


def fig_confusion(results_dir: Path, figures_dir: Path) -> Path | None:
    df = _read(results_dir / "predictions.csv")
    if df is None:
        return None
    plt = _mpl()
    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models, strict=False):
        sub = df[df["model"] == model]
        cm = _confusion_rate_matrix(sub)
        im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(f"{model} (rates)")
        ax.set_xticks([0, 1], ["left", "right"])
        ax.set_yticks([0, 1], ["left", "right"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
        colorbar = fig.colorbar(im, ax=ax, fraction=0.046)
        colorbar.set_label("Rate within true class")
    out = figures_dir / "confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def fig_robustness_curve(results_dir: Path, figures_dir: Path) -> Path | None:
    df = _read(results_dir / "robustness_metrics.csv")
    if df is None:
        return None
    plt = _mpl()
    rnd = df[df["condition"] == "random"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, sub in rnd.groupby("model"):
        g = sub.groupby("fraction")["balanced_accuracy"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            g["fraction"] * 100, g["mean"], yerr=g["std"], marker="o", label=str(model), capsize=3
        )
    ax.set_xlabel("% channels removed (random)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Robustness to random channel loss (test-time, no retraining)")
    ax.legend()
    out = figures_dir / "robustness_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def fig_robustness_regional(results_dir: Path, figures_dir: Path) -> Path | None:
    df = _read(results_dir / "robustness_metrics.csv")
    if df is None:
        return None
    plt = _mpl()
    reg = df[df["condition"] == "regional"]
    if reg.empty:
        return None
    g = reg.groupby(["model", "mask"])["balanced_accuracy"].mean().unstack("model")
    ax = g.plot(kind="bar", figsize=(7, 4))
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Robustness to regional channel loss")
    out = figures_dir / "robustness_regional.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def fig_stability(results_dir: Path, figures_dir: Path) -> Path | None:
    df = _read(results_dir / "explanation_stability.csv")
    if df is None:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    data, labels = [], []
    for (model, cond), sub in df.groupby(["model", "condition"]):
        vals = sub["spearman"].dropna()
        if len(vals):
            data.append(vals)
            labels.append(f"{model}\n{cond}")
    if not data:
        return None
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("Spearman rank correlation")
    ax.set_title("Explanation stability")
    ax.axhline(0, color="k", ls="--", lw=0.7)
    out = figures_dir / "explanation_stability.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def fig_efficiency(results_dir: Path, figures_dir: Path) -> Path | None:
    df = _read(results_dir / "efficiency.csv")
    if df is None:
        return None
    plt = _mpl()
    devices = (
        df["device"].astype(str) if "device" in df else pd.Series("unspecified", index=df.index)
    )
    labels = [f"{model}\n({device})" for model, device in zip(df["model"], devices, strict=False)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, df["n_params"])
    axes[0].set_ylabel("Reported fitted parameter entries")
    axes[0].set_title("Model parameter count (definitions differ)")
    axes[1].bar(labels, df["latency_p50_ms"])
    axes[1].set_ylabel("Latency p50 (ms, batch=1)")
    axes[1].set_title("Observed batch-1 latency (run-specific)")
    out = figures_dir / "efficiency.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def _importance_aggregation_label(sub: pd.DataFrame) -> str:
    """Describe the saved importance estimand without inventing a subject weighting."""
    if "aggregation_population" not in sub:
        return "mean across saved importance vectors"
    populations = set(sub["aggregation_population"].dropna().astype(str))
    if populations == {"outer_train_refit_model"}:
        return "mean across outer-fold refit models"
    if populations == {"shared_outer_test_examples"}:
        return "mean across fold/seed vectors on shared held-out examples"
    return "mean across saved importance vectors"


def fig_importance_topomap(results_dir: Path, figures_dir: Path) -> Path | None:
    """Bar chart of channel importance for CSP and CNN (topomap needs montage; bar is robust)."""
    df = _read(results_dir / "channel_importance.csv")
    if df is None:
        return None
    plt = _mpl()
    models = df["model"].unique()
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3 * len(models)), squeeze=False)
    for ax, model in zip(axes[:, 0], models, strict=False):
        model_rows = df[df["model"] == model]
        importance = model_rows.groupby("channel")["importance"].mean().sort_values(ascending=False)
        ax.bar(importance.index[:20], importance.values[:20])
        ax.set_title(
            f"{model} top-20 channel importance ({_importance_aggregation_label(model_rows)})"
        )
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylabel("importance")
    out = figures_dir / "channel_importance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def generate_all(results_dir: Path, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in [
        fig_class_counts,
        fig_confusion,
        fig_subject_scatter,
        fig_paired_diff,
        fig_subject_distribution,
        fig_robustness_curve,
        fig_robustness_regional,
        fig_stability,
        fig_efficiency,
        fig_importance_topomap,
    ]:
        try:
            p = fn(results_dir, figures_dir)
            if p is not None:
                made.append(p)
                logger.info("Wrote figure %s", p)
        except Exception as exc:  # noqa: BLE001 - one bad figure shouldn't kill the report
            logger.error("Figure %s failed: %s", fn.__name__, exc)
    return made
