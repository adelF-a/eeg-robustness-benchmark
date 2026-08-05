"""Train and select the CSP + logistic-regression baseline for one outer fold.

Hyperparameters are chosen by **genuine grouped inner cross-validation**: every
configuration is evaluated on every configured inner fold (preprocessing/CSP/scaling/
classifier fit independently inside each inner fold, on inner-train subjects only, scored
on inner-val subjects). Selection uses the mean inner-fold balanced accuracy, with
deterministic tie-breaking by grid order. The selected configuration is then refit on the
outer train+validation subjects and evaluated on the held-out outer-test subjects. The
outer-test set never influences selection.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import Config
from ..data.epochs import EpochBundle
from ..data.splits import OuterFold
from ..evaluation.metrics import balanced_accuracy_score
from ..models.csp_logreg import build_csp_pipeline
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CSPFoldResult:
    fold: int
    params: dict[str, Any]
    val_score: float  # mean inner-fold balanced accuracy of the selected configuration
    pipeline: Any
    csp_filters: np.ndarray
    csp_patterns: np.ndarray
    logreg_coef: np.ndarray
    test_pred: np.ndarray
    test_proba: np.ndarray
    test_subjects: list[int]
    outer_train_subjects: list[int]
    n_inner_folds: int
    inner_cv: list[dict] = field(default_factory=list)  # per-config per-inner-fold scores


def _param_grid(cfg: Config):
    for nc, reg, C, cw in itertools.product(
        cfg.csp.n_components_grid,
        cfg.csp.reg_grid,
        cfg.csp.C_grid,
        cfg.csp.class_weight_grid,
    ):
        yield {"n_components": nc, "reg": reg, "C": C, "class_weight": cw}


def _score_config_on_inner_fold(bundle, in_tr_subj, in_val_subj, params, seed) -> float:
    """Fit a fresh CSP pipeline on inner-train subjects, score on inner-val subjects."""
    tr = bundle.subset(bundle.subject_mask(in_tr_subj))
    va = bundle.subset(bundle.subject_mask(in_val_subj))
    pipe = build_csp_pipeline(
        n_components=params["n_components"],
        reg=params["reg"],
        C=params["C"],
        class_weight=params["class_weight"],
        random_state=seed,
    )
    pipe.fit(tr.X, tr.y)
    return float(balanced_accuracy_score(va.y, pipe.predict(va.X)))


def train_csp_fold(bundle: EpochBundle, fold: OuterFold, cfg: Config) -> CSPFoldResult:
    """Inner-CV grid search, refit on outer train+val, evaluate on outer test."""
    if not fold.inner_folds:
        raise ValueError(
            f"Fold {fold.fold} has no inner folds; regenerate splits with make-splits."
        )
    seed = cfg.splits.seed
    inner_cv: list[dict] = []
    best_mean, best_params = -np.inf, None

    # Iterate the grid in a fixed order so ties resolve deterministically (first wins).
    for config_index, params in enumerate(_param_grid(cfg)):
        per_fold: list[float] = []
        failure: str | None = None
        config_records: list[dict] = []
        for j, (in_tr_subj, in_val_subj) in enumerate(fold.inner_folds):
            if failure is not None:
                config_records.append(
                    {
                        **params,
                        "config_index": config_index,
                        "inner_fold": j,
                        "balanced_accuracy": float("nan"),
                        "fold_status": "not_run",
                        "failure_reason": failure,
                    }
                )
                continue
            try:
                s = _score_config_on_inner_fold(bundle, in_tr_subj, in_val_subj, params, seed)
            except Exception as exc:  # noqa: BLE001 - degenerate config, skip whole config
                logger.warning("CSP params %s failed on inner fold %d: %s", params, j, exc)
                failure = f"{type(exc).__name__}: {exc}"
                config_records.append(
                    {
                        **params,
                        "config_index": config_index,
                        "inner_fold": j,
                        "balanced_accuracy": float("nan"),
                        "fold_status": "failed",
                        "failure_reason": failure,
                    }
                )
                continue
            per_fold.append(s)
            config_records.append(
                {
                    **params,
                    "config_index": config_index,
                    "inner_fold": j,
                    "balanced_accuracy": s,
                    "fold_status": "ok",
                    "failure_reason": "",
                }
            )
        if failure is not None or len(per_fold) != len(fold.inner_folds):
            for record in config_records:
                record["config_status"] = "failed"
                record["mean_inner_balanced_accuracy"] = float("nan")
            inner_cv.extend(config_records)
            continue
        mean_score = float(np.mean(per_fold))
        for record in config_records:
            record["config_status"] = "ok"
            record["mean_inner_balanced_accuracy"] = mean_score
        inner_cv.extend(config_records)
        # Strictly-greater update => the first config achieving the maximum mean wins.
        if mean_score > best_mean:
            best_mean, best_params = mean_score, params

    if best_params is None:
        raise RuntimeError(f"No CSP configuration succeeded for fold {fold.fold}.")
    logger.info(
        "Fold %d selected CSP %s (mean inner bal-acc %.3f over %d folds)",
        fold.fold,
        best_params,
        best_mean,
        len(fold.inner_folds),
    )

    # Refit the selected configuration on the outer train+validation subjects.
    fit = bundle.subset(bundle.subject_mask(fold.outer_train_subjects))
    te = bundle.subset(bundle.subject_mask(fold.test_subjects))
    pipe = build_csp_pipeline(
        n_components=best_params["n_components"],
        reg=best_params["reg"],
        C=best_params["C"],
        class_weight=best_params["class_weight"],
        random_state=seed,
    )
    pipe.fit(fit.X, fit.y)

    csp = pipe.named_steps["csp"]
    logreg = pipe.named_steps["logreg"]
    test_proba = pipe.predict_proba(te.X)[:, 1]
    test_pred = pipe.predict(te.X)

    return CSPFoldResult(
        fold=fold.fold,
        params=best_params,
        val_score=float(best_mean),
        pipeline=pipe,
        csp_filters=np.asarray(csp.filters_),
        csp_patterns=np.asarray(csp.patterns_),
        logreg_coef=np.asarray(logreg.coef_).ravel(),
        test_pred=test_pred,
        test_proba=test_proba,
        test_subjects=list(fold.test_subjects),
        outer_train_subjects=list(fold.outer_train_subjects),
        n_inner_folds=len(fold.inner_folds),
        inner_cv=inner_cv,
    )
