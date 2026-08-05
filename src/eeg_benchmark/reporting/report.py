"""Assemble the active profile's ``report.md`` from saved result files.

The report only prints numbers that exist in result CSVs. If a result file is missing,
the corresponding section is marked *pending* rather than filled with invented values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config
from ..evaluation.bootstrap import bootstrap_ci
from ..evaluation.compare import compare_models
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _is_smoke_profile(cfg: Config) -> bool:
    return cfg.name.lower().startswith("smoke")


def _paired_ci_interpretation(lo: float, hi: float) -> str:
    """Interpret a paired-difference interval without claiming universal superiority."""
    if lo > 0:
        return (
            "the percentile-bootstrap CI lies entirely above 0, supporting a positive "
            "average paired CNN − CSP difference within this dataset and protocol. "
            "This does not establish universal superiority."
        )
    if hi < 0:
        return (
            "the percentile-bootstrap CI lies entirely below 0, supporting a negative "
            "average paired CNN − CSP difference within this dataset and protocol. "
            "This does not establish universal superiority."
        )
    return (
        "the percentile-bootstrap CI includes 0, so these data do not establish an "
        "average paired difference at this sample size."
    )


def build_report(cfg: Config) -> Path:
    rd = cfg.paths.results_dir
    lines: list[str] = []
    a = lines.append

    meta = {}
    mpath = rd / "run_metadata.json"
    if mpath.exists():
        meta = json.loads(mpath.read_text())

    a("# EEG Motor-Imagery Robustness Benchmark — Report\n")
    a(f"_Config profile:_ **{cfg.name}**<br>")
    if meta:
        a(
            f"_Device:_ {meta.get('device', 'n/a')}  _torch:_ {meta.get('torch_version', 'n/a')}  "
            f"_mne:_ {meta.get('mne_version', 'n/a')}"
        )
    a(
        f"\n> This report is generated automatically from files in `{rd}/`. "
        "Numbers appear only when the corresponding stage has actually run.\n"
    )
    if _is_smoke_profile(cfg):
        a(
            "> ⚠️ The **smoke** profile is a functional check on very few subjects/seeds, "
            "not a scientific result.\n"
        )

    a("## Research question\n")
    a(
        "Does a compact neural EEG classifier outperform a CSP + logistic-regression baseline "
        "on previously unseen subjects, and how robust and explainable are both models when EEG "
        "channels are missing?\n"
    )

    # --- Data ---
    a("## Dataset & validation\n")
    dv = _load(rd / "data_validation.csv")
    if dv is not None:
        n_subj = dv["subject"].nunique()
        n_fail = dv[dv["status"] == "fail"]["subject"].nunique()
        a(f"- Subjects inspected: **{n_subj}**; subjects with ≥1 failing run: **{n_fail}**.")
        a(
            f"- Total T1 (left) epochs: **{int(dv['n_t1'].sum())}**, "
            f"T2 (right) epochs: **{int(dv['n_t2'].sum())}**, rejected: **{int(dv['n_rejected'].sum())}**."
        )
        failed = dv[dv["status"] == "fail"]
        if not failed.empty:
            a("- Excluded/failed runs (subject, run, reason):")
            for _, r in failed.iterrows():
                a(f"  - s{int(r['subject'])} run {int(r['run'])}: {r['reason']}")
    else:
        a("_Data validation pending — run `validate-data`._")

    # --- Splits ---
    a("\n## Leakage prevention\n")
    folds = sorted((rd / "splits").glob("outer_fold_*.json")) if (rd / "splits").exists() else []
    if folds:
        a(f"- Subject-grouped outer folds saved: **{len(folds)}** (`{rd / 'splits'}/`).")
        a(
            f"- Split method: **{cfg.splits.method}**; outer folds: "
            f"**{cfg.splits.outer_folds}**; grouped inner folds: "
            f"**{cfg.splits.inner_folds}**; seed: **{cfg.splits.seed}**."
        )
        a("- Verified disjoint train/val/test subjects (see `tests/test_subject_splits.py`).")
        a(
            "- CSP parameters are selected by mean performance across all grouped inner folds. "
            "CNN early stopping uses the first grouped inner validation fold; a freshly "
            "initialized CNN is then refit on the complete outer-training set."
        )
    else:
        a("_Splits pending — run `make-splits`._")

    # --- Main comparison ---
    a("\n## Results — generalisation to unseen subjects\n")
    sm = _load(rd / "subject_metrics.csv")
    if sm is not None:
        a("Per-subject balanced accuracy (held-out test subjects):\n")
        a("| Model | mean | 95% CI | median |")
        a("|---|---|---|---|")
        for model, sub in sm.groupby("model"):
            vals = sub.groupby("subject")["balanced_accuracy"].mean().to_numpy()
            ci = bootstrap_ci(
                vals, n_boot=cfg.bootstrap.n_boot, ci=cfg.bootstrap.ci, seed=cfg.bootstrap.seed
            )
            a(
                f"| {model} | {ci['mean']:.3f} | [{ci['lo']:.3f}, {ci['hi']:.3f}] | {ci.get('median', float('nan')):.3f} |"
            )
        cmp = compare_models(
            sm, n_boot=cfg.bootstrap.n_boot, ci=cfg.bootstrap.ci, seed=cfg.bootstrap.seed
        )
        if "mean_diff" in cmp:
            a(
                f"\nPaired CNN − CSP balanced accuracy: **{cmp['mean_diff']:+.3f}** "
                f"(95% CI [{cmp['lo']:.3f}, {cmp['hi']:.3f}], n={cmp['n']} subjects; "
                f"CNN better on {cmp['frac_a_better'] * 100:.0f}% of subjects)."
            )
            a(f"\nInterpretation: {_paired_ci_interpretation(cmp['lo'], cmp['hi'])}")
    else:
        a("_Model metrics pending — run `train-csp`, `train-cnn`, `evaluate`._")

    # --- Robustness ---
    a("\n## Channel-drop robustness (test-time, no retraining)\n")
    rob = _load(rd / "robustness_metrics.csv")
    if rob is not None:
        piv = (
            rob[rob["condition"] == "random"]
            .groupby(["model", "fraction"])["balanced_accuracy"]
            .mean()
            .reset_index()
        )
        a("Mean balanced accuracy under random channel loss:\n")
        a("| Model | fraction removed | balanced accuracy |")
        a("|---|---|---|")
        for _, r in piv.iterrows():
            a(f"| {r['model']} | {r['fraction']:.2f} | {r['balanced_accuracy']:.3f} |")
    else:
        a("_Robustness pending — run `stress-test`._")

    # --- Explanations & stability ---
    a("\n## Explainability & explanation stability\n")
    st = _load(rd / "explanation_stability.csv")
    if st is not None:
        a("Median explanation stability (Spearman of channel-importance vectors):\n")
        a("| Model | condition | median Spearman | median cosine |")
        a("|---|---|---|---|")
        for (model, cond), sub in st.groupby(["model", "condition"]):
            a(
                f"| {model} | {cond} | {sub['spearman'].median():.3f} | {sub['cosine_similarity'].median():.3f} |"
            )
        a(
            "\nExplanations are predictive attributions over channels, **not** proof of a neural "
            "mechanism. Low stability is reported as a legitimate negative result."
        )
    else:
        a("_Explanation stability pending — run `explain`._")

    # --- Efficiency ---
    a("\n## Efficiency\n")
    eff = _load(rd / "efficiency.csv")
    if eff is not None:
        a(
            "| Model | reported fitted params | definition | size (KB) | latency p50 (ms) | "
            "latency p95 (ms) | device |"
        )
        a("|---|---|---|---|---|---|---|")
        for _, r in eff.iterrows():
            a(
                f"| {r['model']} | {int(r['n_params'])} | "
                f"{r.get('parameter_definition', 'unspecified')} | "
                f"{r.get('size_kb', float('nan')):.1f} | "
                f"{r['latency_p50_ms']:.3f} | {r['latency_p95_ms']:.3f} | {r.get('device', 'cpu')} |"
            )
        a(
            "\nCSP and CNN parameter counts use different model families and are descriptive, "
            "not a direct comparison of capacity. Accelerator timings are synchronized before "
            "wall-clock measurement."
        )
        if "device" in eff and eff["device"].astype(str).nunique() > 1:
            observed = "; ".join(
                f"{row.model}: {row.device}" for row in eff[["model", "device"]].itertuples()
            )
            a(
                f"\nCNN and CSP latency were measured on different devices/backends ({observed}). "
                "These are run-specific operational measurements and must not be interpreted as "
                "an intrinsic model-speed comparison."
            )
        else:
            a(
                "\nLatency is environment- and backend-specific; it describes this run rather "
                "than an intrinsic model-speed ranking."
            )
    else:
        a("_Efficiency pending — run `evaluate`._")

    # --- Limitations ---
    a("\n## Limitations\n")
    a("- Motor-imagery EEG has low SNR; absolute accuracies can be modest.")
    a("- Explanations are correlational, not causal or clinical.")
    if _is_smoke_profile(cfg):
        a("- The smoke profile uses too few subjects/seeds for statistical inference.")
    else:
        a(
            "- The full profile evaluates one public dataset under one predefined protocol; "
            "there is no external-dataset replication."
        )
        a(
            "- Cross-validation folds share training data. Subject-bootstrap intervals describe "
            "held-out-subject variation conditional on this run, not independent-study uncertainty."
        )
    a("- No clinical claims are made; this is a methods benchmark.")

    out = cfg.paths.reports_dir / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    logger.info("Wrote report %s", out)
    return out
