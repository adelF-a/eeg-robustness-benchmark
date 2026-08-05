"""End-to-end experiment orchestration.

Each function corresponds to a CLI command and reads/writes explicit artifacts under the
active profile's configured result directory so stages can run independently. Models persist
(CSP as joblib, CNN as a torch checkpoint including its standardiser and architecture) so
robustness and explanation stages reuse the *same* fitted models — never retrained on
test data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data.epochs import EpochBundle
from .data.preprocessing import build_dataset
from .data.splits import OuterFold, load_folds, make_outer_folds, save_folds
from .evaluation.metrics import fold_metrics, per_subject_metrics
from .models.compact_eegnet import CompactEEGNet, count_parameters
from .utils.hardware import environment_metadata, get_device
from .utils.identity import config_hash, ensure_run_manifest, register_artifact, require_artifact
from .utils.logging import get_logger

logger = get_logger(__name__)

PRED_COLUMNS = [
    "subject",
    "run",
    "epoch_id",
    "true_label",
    "predicted_label",
    "class_probability",
    "fold",
    "model",
    "seed",
    "config_hash",
]


# --------------------------------------------------------------------------- #
# Bundle loading (real data). Tests inject synthetic bundles directly instead.
# --------------------------------------------------------------------------- #
def load_bundle(cfg: Config, subjects: list[int] | None = None) -> EpochBundle:
    """Build the epoch bundle for a real-data stage.

    When subjects are unspecified the authoritative survivor set (``valid_subjects.json``)
    is **required** — there is no silent fallback to all configured subjects, so a stage
    cannot preprocess data that was never validated. Identity is checked first.
    """
    ensure_run_manifest(cfg)
    if subjects is None:
        from .data.validation import load_valid_subjects, verify_valid_manifest

        manifest = load_valid_subjects(cfg)
        if manifest is None:
            raise RuntimeError(
                f"No validation manifest ({cfg.paths.results_dir / 'valid_subjects.json'}). "
                "Run 'validate-data' "
                "(or 'prepare'/'run-all') before this stage; refusing to preprocess without "
                "an authoritative valid-subject set."
            )
        verify_valid_manifest(cfg, manifest, sorted(int(s) for s in cfg.data.subjects))
        subjects = list(manifest["valid"])
    return build_dataset(cfg, subjects)


def _folds(cfg: Config) -> list[OuterFold]:
    """Load folds for the active config, verifying the run manifest and split stamps so a
    stale/foreign artifact is rejected rather than silently mixed."""
    ensure_run_manifest(cfg)
    return load_folds(cfg.paths.splits_dir, expected_hash=config_hash(cfg))


def _write_metadata(cfg: Config, extra: dict | None = None) -> None:
    meta = environment_metadata()
    meta["config_name"] = cfg.name
    meta["config_hash"] = config_hash(cfg)
    meta["runs"] = cfg.data.runs
    if extra:
        meta.update(extra)
    (cfg.paths.results_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )
    cfg.save(cfg.paths.results_dir / "config_used.yaml")


# --------------------------------------------------------------------------- #
# prepare / make-splits
# --------------------------------------------------------------------------- #
def prepare(cfg: Config, bundle: EpochBundle | None = None) -> EpochBundle:
    cfg.paths.ensure()
    ensure_run_manifest(cfg)
    if bundle is None:
        from .data.validation import resolve_valid_subjects

        requested = sorted(int(s) for s in cfg.data.subjects)
        valid, excluded = resolve_valid_subjects(cfg)
        excluded.to_csv(cfg.paths.results_dir / "excluded_subjects.csv", index=False)
        if not valid:
            raise RuntimeError(
                "No valid subjects after validation; see "
                f"{cfg.paths.results_dir / 'data_validation.csv'} and "
                f"{cfg.paths.results_dir / 'excluded_subjects.csv'}."
            )
        excluded_ids = sorted({int(s) for s in excluded["subject"]}) if len(excluded) else []
        logger.info("Prepare: %d valid subjects, %d excluded", len(valid), len(excluded))
        bundle = build_dataset(cfg, valid)
    else:
        # Injected bundle (tests/synthetic): its subjects are the validated set by definition.
        requested = sorted(int(s) for s in np.unique(bundle.groups))
        valid, excluded_ids = requested, []
    counts = (
        bundle.metadata.groupby(["subject", "event"]).size().reset_index(name="n")
        if len(bundle.metadata)
        else pd.DataFrame()
    )
    counts.to_csv(cfg.paths.results_dir / "class_counts.csv", index=False)
    _write_metadata(
        cfg,
        {
            "requested_subjects": requested,
            "validated_subjects": sorted(int(s) for s in valid),
            "excluded_subjects": excluded_ids,
            "n_epochs": bundle.n_epochs,
            "n_channels": bundle.n_channels,
            "n_times": bundle.n_times,
            "ch_names": bundle.ch_names,
        },
    )
    logger.info(
        "Prepared bundle: %d epochs, %d channels, %d times",
        bundle.n_epochs,
        bundle.n_channels,
        bundle.n_times,
    )
    return bundle


def make_splits(cfg: Config, bundle: EpochBundle | None = None) -> list[OuterFold]:
    ensure_run_manifest(cfg)
    bundle = bundle if bundle is not None else load_bundle(cfg)
    folds = make_outer_folds(bundle.y, bundle.groups, cfg)
    save_folds(folds, cfg.paths.splits_dir)
    return folds


# --------------------------------------------------------------------------- #
# CSP training
# --------------------------------------------------------------------------- #
def train_csp(cfg: Config, bundle: EpochBundle | None = None) -> pd.DataFrame:
    import joblib

    from .training.train_csp import train_csp_fold

    chash = config_hash(cfg)
    bundle = bundle if bundle is not None else load_bundle(cfg)
    folds = _folds(cfg)
    models_dir = cfg.paths.results_dir / "models"
    art_dir = cfg.paths.results_dir / "csp_artifacts"
    models_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    params_rows: list[dict] = []
    inner_cv_rows: list[dict] = []
    for fold in folds:
        res = train_csp_fold(bundle, fold, cfg)
        joblib.dump(res.pipeline, models_dir / f"csp_fold{fold.fold}.joblib")
        np.savez(
            art_dir / f"fold{fold.fold}.npz",
            filters=res.csp_filters,
            patterns=res.csp_patterns,
            logreg_coef=res.logreg_coef,
            ch_names=np.array(bundle.ch_names),
            config_hash=np.array(chash),
        )
        params_rows.append(
            {
                "fold": fold.fold,
                "mean_inner_balanced_accuracy": res.val_score,
                "n_inner_folds": res.n_inner_folds,
                "n_outer_train_subjects": len(res.outer_train_subjects),
                "outer_train_subjects": ";".join(str(s) for s in res.outer_train_subjects),
                "config_hash": chash,
                **res.params,
            }
        )
        for rec in res.inner_cv:
            inner_cv_rows.append({"fold": fold.fold, "config_hash": chash, **rec})
        te = bundle.subset(bundle.subject_mask(fold.test_subjects))
        for i in range(te.n_epochs):
            md = te.metadata.iloc[i]
            rows.append(
                {
                    "subject": int(md["subject"]),
                    "run": int(md["run"]),
                    "epoch_id": md["epoch_id"],
                    "true_label": int(te.y[i]),
                    "predicted_label": int(res.test_pred[i]),
                    "class_probability": float(res.test_proba[i]),
                    "fold": fold.fold,
                    "model": "csp",
                    "seed": 0,
                    "config_hash": chash,
                }
            )
    preds = pd.DataFrame(rows, columns=PRED_COLUMNS)
    preds.to_csv(cfg.paths.results_dir / "predictions_csp.csv", index=False)
    pd.DataFrame(params_rows).to_csv(cfg.paths.results_dir / "csp_selected_params.csv", index=False)
    pd.DataFrame(inner_cv_rows).to_csv(cfg.paths.results_dir / "csp_inner_cv.csv", index=False)
    for name, stage in [
        ("predictions_csp.csv", "train_csp"),
        ("csp_models", "train_csp"),
        ("csp_artifacts", "train_csp"),
    ]:
        register_artifact(cfg, name, stage)
    logger.info("CSP training complete: %d predictions", len(preds))
    return preds


# --------------------------------------------------------------------------- #
# CNN training
# --------------------------------------------------------------------------- #
def _cnn_ckpt_path(cfg: Config, fold: int, seed: int) -> Path:
    return cfg.paths.results_dir / "models" / f"cnn_fold{fold}_seed{seed}.pt"


def _save_cnn(cfg: Config, res, bundle: EpochBundle) -> None:
    import torch

    ckpt = {
        "state_dict": res.state_dict,
        "arch": {
            "n_channels": bundle.n_channels,
            "n_times": bundle.n_times,
            "F1": cfg.neural.F1,
            "D": cfg.neural.D,
            "kernel_length": cfg.neural.kernel_length,
            "dropout": cfg.neural.dropout,
        },
        "std_mean": res.standardizer.mean,
        "std_std": res.standardizer.std,
        "ch_names": bundle.ch_names,
        "config_hash": config_hash(cfg),
        "seed": res.seed,
        "best_epoch": res.best_epoch,
        "refit_epochs": res.refit_epochs,
        "train_subjects": res.train_subjects,
        "val_subjects": res.val_subjects,
        "outer_train_subjects": res.outer_train_subjects,
    }
    torch.save(ckpt, _cnn_ckpt_path(cfg, res.fold, res.seed))


def load_cnn(cfg: Config, fold: int, seed: int, device: str | None = None):
    """Reload a trained CNN + its standardiser from checkpoint.

    Verifies the checkpoint's identity stamp against the active config so a foreign/stale
    checkpoint is rejected rather than silently used.
    """
    import torch

    from .training.train_neural import ChannelStandardizer

    device = device or get_device()
    ckpt = torch.load(_cnn_ckpt_path(cfg, fold, seed), map_location=device, weights_only=False)
    stamp = ckpt.get("config_hash", "")
    active = config_hash(cfg)
    if stamp != active:
        raise RuntimeError(
            f"CNN checkpoint fold{fold} seed{seed} was trained for config '{stamp}' but the "
            f"active config hashes to '{active}'. Refusing to use a foreign checkpoint. "
            f"Re-run train-cnn for this profile. No files were modified."
        )
    model = CompactEEGNet(n_classes=2, **ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    std = ChannelStandardizer(mean=np.asarray(ckpt["std_mean"]), std=np.asarray(ckpt["std_std"]))
    return model, std, ckpt["ch_names"]


def train_cnn(cfg: Config, bundle: EpochBundle | None = None) -> pd.DataFrame:
    from .training.train_neural import train_neural_fold

    chash = config_hash(cfg)
    bundle = bundle if bundle is not None else load_bundle(cfg)
    folds = _folds(cfg)
    (cfg.paths.results_dir / "models").mkdir(parents=True, exist_ok=True)
    device = get_device()
    logger.info("Training CNN on device=%s", device)

    rows: list[dict] = []
    hist_rows: list[dict] = []
    sel_rows: list[dict] = []
    for fold in folds:
        for seed in cfg.neural.seeds:
            res = train_neural_fold(bundle, fold, cfg, seed=seed, device=device)
            _save_cnn(cfg, res, bundle)
            sel_rows.append(
                {
                    "fold": fold.fold,
                    "seed": seed,
                    "best_epoch": res.best_epoch,
                    "refit_epochs": res.refit_epochs,
                    "best_val_score": res.best_val_score,
                    "n_train_subjects": len(res.train_subjects),
                    "n_val_subjects": len(res.val_subjects),
                    "n_outer_train_subjects": len(res.outer_train_subjects),
                    "n_test_subjects": len(res.test_subjects),
                    "outer_train_subjects": ";".join(str(s) for s in res.outer_train_subjects),
                    "config_hash": chash,
                }
            )
            te = bundle.subset(bundle.subject_mask(fold.test_subjects))
            for i in range(te.n_epochs):
                md = te.metadata.iloc[i]
                rows.append(
                    {
                        "subject": int(md["subject"]),
                        "run": int(md["run"]),
                        "epoch_id": md["epoch_id"],
                        "true_label": int(te.y[i]),
                        "predicted_label": int(res.test_pred[i]),
                        "class_probability": float(res.test_proba[i]),
                        "fold": fold.fold,
                        "model": "cnn",
                        "seed": seed,
                        "config_hash": chash,
                    }
                )
            for ep, tl in enumerate(res.history["train_loss"]):
                hist_rows.append(
                    {
                        "fold": fold.fold,
                        "seed": seed,
                        "epoch": ep,
                        "phase": "selection",
                        "train_loss": tl,
                        "val_balanced_accuracy": res.history["val_balanced_accuracy"][ep],
                        "val_macro_f1": res.history["val_macro_f1"][ep],
                        "n_training_subjects": len(res.train_subjects),
                        "config_hash": chash,
                    }
                )
            for ep, tl in enumerate(res.refit_history["train_loss"]):
                hist_rows.append(
                    {
                        "fold": fold.fold,
                        "seed": seed,
                        "epoch": ep,
                        "phase": "refit",
                        "train_loss": tl,
                        "val_balanced_accuracy": np.nan,
                        "val_macro_f1": np.nan,
                        "n_training_subjects": len(res.outer_train_subjects),
                        "config_hash": chash,
                    }
                )
            logger.info(
                "CNN fold %d seed %d: %d params, best val %.3f @epoch %d",
                fold.fold,
                seed,
                res.n_params,
                res.best_val_score,
                res.best_epoch,
            )
    preds = pd.DataFrame(rows, columns=PRED_COLUMNS)
    preds.to_csv(cfg.paths.results_dir / "predictions_cnn.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(cfg.paths.results_dir / "training_curves.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(cfg.paths.results_dir / "cnn_selection.csv", index=False)
    register_artifact(cfg, "predictions_cnn.csv", "train_cnn")
    register_artifact(cfg, "cnn_checkpoints", "train_cnn")
    return preds


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def evaluate(cfg: Config, bundle: EpochBundle | None = None) -> pd.DataFrame:
    from .evaluation.compare import compare_models

    ensure_run_manifest(cfg)
    rd = cfg.paths.results_dir
    active = config_hash(cfg)
    parts = [p for p in [rd / "predictions_csp.csv", rd / "predictions_cnn.csv"] if p.exists()]
    if not parts:
        raise FileNotFoundError("No predictions found; run train-csp and train-cnn first.")
    frames = []
    for p in parts:
        require_artifact(cfg, p.name)  # refuse legacy/unregistered prediction inputs
        df = pd.read_csv(p)
        if "config_hash" not in df.columns:
            raise RuntimeError(
                f"{p.name} has no config_hash column (legacy prediction file). Re-run its "
                f"producing stage for this profile; refusing legacy input. No files modified."
            )
        if not (df["config_hash"].astype(str) == active).all():
            raise RuntimeError(
                f"{p.name} contains predictions produced under a different config than the "
                f"active one ({active}). Refusing to evaluate mixed results. No files modified."
            )
        frames.append(df)
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(rd / "predictions.csv", index=False)

    fm = fold_metrics(preds)
    sm = per_subject_metrics(preds)
    fm.to_csv(rd / "fold_metrics.csv", index=False)
    sm.to_csv(rd / "subject_metrics.csv", index=False)

    comps = []
    for metric in ["balanced_accuracy", "macro_f1", "roc_auc"]:
        c = compare_models(
            sm,
            metric=metric,
            n_boot=cfg.bootstrap.n_boot,
            ci=cfg.bootstrap.ci,
            seed=cfg.bootstrap.seed,
        )
        comps.append(c)
    pd.DataFrame(comps).to_csv(rd / "model_comparison.csv", index=False)

    _efficiency(cfg, bundle)
    logger.info("Evaluation complete: %d subject rows", len(sm))
    return sm


def _count_csp_parameters(pipe) -> int:
    """Count CSP filters actually used plus logistic-regression coefficients/intercept."""
    csp = pipe.named_steps["csp"]
    logreg = pipe.named_steps["logreg"]
    filters = np.asarray(csp.filters_)
    n_components = min(int(csp.n_components), filters.shape[0])
    return int(
        filters[:n_components].size
        + np.asarray(logreg.coef_).size
        + np.asarray(logreg.intercept_).size
    )


def _efficiency(cfg: Config, bundle: EpochBundle | None) -> None:
    import joblib

    from .evaluation.latency import benchmark_callable_latency, benchmark_torch_latency

    rd = cfg.paths.results_dir
    rows = []
    device = get_device()
    folds = _folds(cfg)
    f0 = folds[0].fold
    meta = (
        json.loads((rd / "run_metadata.json").read_text())
        if (rd / "run_metadata.json").exists()
        else {}
    )
    n_ch = meta.get("n_channels", bundle.n_channels if bundle else 64)
    n_t = meta.get("n_times", bundle.n_times if bundle else 481)

    # CNN
    ckpt = _cnn_ckpt_path(cfg, f0, cfg.neural.seeds[0])
    if ckpt.exists():
        model, _, _ = load_cnn(cfg, f0, cfg.neural.seeds[0], device=device)
        lat = benchmark_torch_latency(
            model,
            (1, 1, n_ch, n_t),
            device=device,
            n_warmup=cfg.latency.n_warmup,
            n_repeats=cfg.latency.n_repeats,
        )
        rows.append(
            {
                "model": "cnn",
                "n_params": count_parameters(model),
                "size_kb": ckpt.stat().st_size / 1024,
                "latency_p50_ms": lat["p50_ms"],
                "latency_p95_ms": lat["p95_ms"],
                "device": lat["device"],
                "parameter_definition": "trainable_tensor_entries",
            }
        )
    # CSP
    cpath = rd / "models" / f"csp_fold{f0}.joblib"
    if cpath.exists():
        pipe = joblib.load(cpath)
        rng = np.random.default_rng(cfg.splits.seed)
        x = rng.standard_normal((1, n_ch, n_t)).astype(np.float32)
        lat = benchmark_callable_latency(
            lambda z: pipe.predict(z),
            x,
            n_warmup=cfg.latency.n_warmup,
            n_repeats=cfg.latency.n_repeats,
        )
        rows.append(
            {
                "model": "csp",
                "n_params": _count_csp_parameters(pipe),
                "size_kb": cpath.stat().st_size / 1024,
                "latency_p50_ms": lat["p50_ms"],
                "latency_p95_ms": lat["p95_ms"],
                "device": "cpu",
                "parameter_definition": "used_csp_filters_plus_logreg_coef_intercept",
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(rd / "efficiency.csv", index=False)


# --------------------------------------------------------------------------- #
# stress-test (channel-drop robustness, no retraining)
# --------------------------------------------------------------------------- #
def _balanced_acc(y_true, y_pred):
    from .evaluation.metrics import balanced_accuracy_score

    return (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) > 1
        else float("nan")
    )


def stress_test(cfg: Config, bundle: EpochBundle | None = None) -> pd.DataFrame:
    import joblib
    import torch

    from .robustness.channel_drop import apply_mask, train_channel_means
    from .robustness.channel_masks import make_masks, masks_to_json

    require_artifact(cfg, "cnn_checkpoints")  # refuse legacy/foreign CNN models
    require_artifact(cfg, "csp_models")  # CSP baseline must be complete too
    bundle = bundle if bundle is not None else load_bundle(cfg)
    folds = _folds(cfg)
    rd = cfg.paths.results_dir
    device = get_device()

    masks = make_masks(
        bundle.ch_names,
        cfg.robustness.random_fractions,
        cfg.robustness.n_random_masks,
        cfg.robustness.regions,
        seed=cfg.splits.seed,
    )
    (rd / "channel_masks.json").write_text(json.dumps(masks_to_json(masks), indent=2))

    rows: list[dict] = []
    for fold in folds:
        # Both persisted models were refit on the complete outer-training population, so
        # channel-loss imputation must use statistics from that same train+validation set.
        tr = bundle.subset(bundle.subject_mask(fold.outer_train_subjects))
        te = bundle.subset(bundle.subject_mask(fold.test_subjects))
        fill = train_channel_means(tr.X)  # [n_ch, 1], training subjects only

        csp_path = rd / "models" / f"csp_fold{fold.fold}.joblib"
        if not csp_path.exists():
            raise RuntimeError(
                f"Missing CSP model for fold {fold.fold} ({csp_path}). Re-run train-csp for "
                f"this profile; refusing to stress-test with incomplete models. "
                f"No files were modified."
            )
        pipe = joblib.load(csp_path)
        cnns = []
        for seed in cfg.neural.seeds:
            if _cnn_ckpt_path(cfg, fold.fold, seed).exists():
                cnns.append((seed, *load_cnn(cfg, fold.fold, seed, device=device)))

        eval_conditions = [("intact", "intact", 0.0, None)] + [
            (m.name, m.condition, m.fraction, m) for m in masks
        ]
        for name, cond, frac, mask in eval_conditions:
            Xm = te.X if mask is None else apply_mask(te.X, bundle.ch_names, mask, fill)
            for subj in np.unique(te.groups):
                sm = te.groups == subj
                if sm.sum() == 0:
                    continue
                if pipe is not None:
                    pred = pipe.predict(Xm[sm])
                    rows.append(
                        {
                            "model": "csp",
                            "seed": 0,
                            "fold": fold.fold,
                            "mask": name,
                            "condition": cond,
                            "fraction": frac,
                            "subject": int(subj),
                            "balanced_accuracy": _balanced_acc(te.y[sm], pred),
                        }
                    )
                for seed, model, std, _ in cnns:
                    with torch.no_grad():
                        Xs = std.transform(Xm[sm])
                        logits = model(torch.from_numpy(Xs).float().to(device))
                        pred = logits.argmax(1).cpu().numpy()
                    rows.append(
                        {
                            "model": "cnn",
                            "seed": seed,
                            "fold": fold.fold,
                            "mask": name,
                            "condition": cond,
                            "fraction": frac,
                            "subject": int(subj),
                            "balanced_accuracy": _balanced_acc(te.y[sm], pred),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(rd / "robustness_metrics.csv", index=False)
    logger.info("Stress test complete: %d rows across %d masks", len(df), len(masks) + 1)
    return df


# --------------------------------------------------------------------------- #
# explain (CSP + CNN channel importance and stability)
# --------------------------------------------------------------------------- #
def _select_test_indices(te: EpochBundle, n_per_subject: int) -> np.ndarray:
    """Pick one model-independent held-out explanation cohort.

    The first ``n_per_subject`` epochs by ``epoch_id`` are selected for every test subject,
    regardless of any model's prediction. This makes the raw examples identical across CNN
    seeds and prevents correctness filtering from cherry-picking a different estimand per model.
    """
    if n_per_subject < 1:
        raise ValueError("n_examples_per_subject must be at least 1")
    idx: list[int] = []
    has_epoch_ids = len(te.metadata) == te.n_epochs and "epoch_id" in te.metadata
    for subj in sorted(int(s) for s in np.unique(te.groups)):
        candidates = np.flatnonzero(te.groups == subj).tolist()
        if has_epoch_ids:
            candidates.sort(key=lambda i: (str(te.metadata.iloc[i]["epoch_id"]), int(i)))
        idx.extend(int(i) for i in candidates[:n_per_subject])
    return np.array(idx, dtype=int)


def explain(cfg: Config, bundle: EpochBundle | None = None) -> None:
    import itertools

    from .explanations.csp_patterns import csp_channel_importance
    from .explanations.integrated_gradients import integrated_gradients
    from .explanations.occlusion import occlusion_importance
    from .explanations.stability import compare_importances

    require_artifact(cfg, "cnn_checkpoints")  # refuse legacy/foreign CNN models
    require_artifact(cfg, "csp_artifacts")  # CSP explanation inputs must be complete too
    bundle = bundle if bundle is not None else load_bundle(cfg)
    folds = _folds(cfg)
    rd = cfg.paths.results_dir
    device = get_device()
    ch = bundle.ch_names
    k = cfg.explanations.top_k

    imp_rows: list[dict] = []
    example_rows: list[dict] = []
    stab_rows: list[dict] = []
    csp_importances: dict[int, np.ndarray] = {}
    cnn_importances: dict[int, dict[int, np.ndarray]] = {}

    for fold in folds:
        # ---- CSP importance (one per fold; required for every fold) ----
        art = rd / "csp_artifacts" / f"fold{fold.fold}.npz"
        if not art.exists():
            raise RuntimeError(
                f"Missing CSP artifact for fold {fold.fold} ({art}). Re-run train-csp for this "
                f"profile; refusing to explain with incomplete artifacts. No files were modified."
            )
        d = np.load(art, allow_pickle=True)
        stamp = str(d["config_hash"]) if "config_hash" in d else ""
        if stamp != config_hash(cfg):
            raise RuntimeError(
                f"CSP artifact {art.name} was produced for config '{stamp}' but the active "
                f"config hashes to '{config_hash(cfg)}'. Refusing legacy input. "
                f"Re-run train-csp for this profile. No files were modified."
            )
        imp = csp_channel_importance(d["patterns"], d["logreg_coef"], len(ch))
        csp_importances[fold.fold] = imp
        for name, val in zip(ch, imp, strict=True):
            imp_rows.append(
                {
                    "model": "csp",
                    "fold": fold.fold,
                    "seed": 0,
                    "channel": name,
                    "importance": float(val),
                    "method": "csp_pattern_logreg_weight",
                    "reference_class": "not_applicable",
                    "aggregation_population": "outer_train_refit_model",
                    "n_examples": 0,
                    "config_hash": config_hash(cfg),
                }
            )

        # ---- CNN importance (per seed) ----
        te = bundle.subset(bundle.subject_mask(fold.test_subjects))
        idx = _select_test_indices(te, cfg.explanations.n_examples_per_subject)
        if len(idx) == 0:
            logger.warning("No held-out explanation examples selected for fold %d", fold.fold)
            continue
        y_sel = te.y[idx]
        cnn_importances.setdefault(fold.fold, {})
        for seed in cfg.neural.seeds:
            if not _cnn_ckpt_path(cfg, fold.fold, seed).exists():
                continue
            model, std, _ = load_cnn(cfg, fold.fold, seed, device=device)
            Xsel = std.transform(te.X[idx])
            import torch

            with torch.no_grad():
                selected_pred = (
                    model(torch.from_numpy(Xsel).float().to(device)).argmax(1).cpu().numpy()
                )
            for local_i, fold_i in enumerate(idx):
                md = te.metadata.iloc[fold_i] if len(te.metadata) else {}
                example_rows.append(
                    {
                        "fold": fold.fold,
                        "fold_epoch_index": int(fold_i),
                        "subject": int(te.groups[fold_i]),
                        "epoch_id": md.get("epoch_id", str(fold_i)),
                        "true_label": int(y_sel[local_i]),
                        "seed": seed,
                        "predicted_label": int(selected_pred[local_i]),
                        "correct": bool(selected_pred[local_i] == y_sel[local_i]),
                        "selection_rule": "first_n_by_epoch_id_per_subject",
                        "config_hash": config_hash(cfg),
                    }
                )
            occ = occlusion_importance(model, Xsel, device=device, reference="true", y_true=y_sel)
            cnn_importances[fold.fold][seed] = occ
            for name, val in zip(ch, occ, strict=True):
                imp_rows.append(
                    {
                        "model": "cnn",
                        "fold": fold.fold,
                        "seed": seed,
                        "channel": name,
                        "importance": float(val),
                        "method": "channel_occlusion_probability_drop",
                        "reference_class": "true_label",
                        "aggregation_population": "shared_outer_test_examples",
                        "n_examples": len(idx),
                        "config_hash": config_hash(cfg),
                    }
                )

            # Sampling stability: compare each epoch bootstrap with the full-cohort
            # importance estimand. The configured replicate count is honoured exactly.
            rng = np.random.default_rng(cfg.bootstrap.seed + fold.fold * 100_000 + seed)
            n_boot = cfg.explanations.n_bootstrap_explain
            for replicate in range(n_boot):
                bidx = rng.choice(len(idx), size=len(idx), replace=True)
                boot_imp = occlusion_importance(
                    model,
                    Xsel[bidx],
                    device=device,
                    reference="true",
                    y_true=y_sel[bidx],
                )
                m = compare_importances(occ, boot_imp, k=k)
                stab_rows.append(
                    {
                        "model": "cnn",
                        "subject_or_fold": f"fold{fold.fold}",
                        "condition": "bootstrap",
                        "seed_a": seed,
                        "seed_b": seed,
                        "comparison": "bootstrap_vs_full",
                        "bootstrap_replicate": replicate,
                        "reference_class": "true_label",
                        "n_examples": len(idx),
                        "config_hash": config_hash(cfg),
                        **m,
                    }
                )

            # input-noise stability
            noisy = Xsel + rng.normal(0, cfg.explanations.noise_std, Xsel.shape).astype(np.float32)
            occ_noisy = occlusion_importance(
                model, noisy, device=device, reference="true", y_true=y_sel
            )
            m = compare_importances(occ, occ_noisy, k=k)
            stab_rows.append(
                {
                    "model": "cnn",
                    "subject_or_fold": f"fold{fold.fold}",
                    "condition": "noise",
                    "seed_a": seed,
                    "seed_b": seed,
                    "comparison": "noisy_vs_full",
                    "bootstrap_replicate": np.nan,
                    "reference_class": "true_label",
                    "n_examples": len(idx),
                    "config_hash": config_hash(cfg),
                    **m,
                }
            )

            # integrated gradients on selected examples; save time-resolved for a few
            try:
                ig_imp, ig_attr = integrated_gradients(
                    model,
                    Xsel[: min(4, len(Xsel))],
                    device=device,
                    baseline=cfg.explanations.ig_baseline,
                    steps=cfg.explanations.ig_steps,
                    reference="true",
                    y_true=y_sel[: min(4, len(Xsel))],
                )
                np.savez(
                    rd / f"ig_fold{fold.fold}_seed{seed}.npz",
                    channel_importance=ig_imp,
                    time_resolved=ig_attr,
                    ch_names=np.array(ch),
                    reference_class=np.array("true_label"),
                    example_epoch_ids=np.array(
                        [
                            te.metadata.iloc[i].get("epoch_id", str(i))
                            if len(te.metadata)
                            else str(i)
                            for i in idx[: min(4, len(Xsel))]
                        ]
                    ),
                    config_hash=np.array(config_hash(cfg)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("IG failed fold %d seed %d: %s", fold.fold, seed, exc)

        # cross-seed stability within a fold
        seeds_here = list(cnn_importances[fold.fold].keys())
        for sa, sb in itertools.combinations(seeds_here, 2):
            m = compare_importances(
                cnn_importances[fold.fold][sa], cnn_importances[fold.fold][sb], k=k
            )
            stab_rows.append(
                {
                    "model": "cnn",
                    "subject_or_fold": f"fold{fold.fold}",
                    "condition": "seed",
                    "seed_a": sa,
                    "seed_b": sb,
                    "comparison": "seed_vs_seed_shared_cohort",
                    "bootstrap_replicate": np.nan,
                    "reference_class": "true_label",
                    "n_examples": len(idx),
                    "config_hash": config_hash(cfg),
                    **m,
                }
            )

    # cross-fold stability for CSP
    for fa, fb in itertools.combinations(sorted(csp_importances), 2):
        m = compare_importances(csp_importances[fa], csp_importances[fb], k=k)
        stab_rows.append(
            {
                "model": "csp",
                "subject_or_fold": f"folds{fa}-{fb}",
                "condition": "fold",
                "seed_a": 0,
                "seed_b": 0,
                "comparison": "fold_vs_fold_fitted_models",
                "bootstrap_replicate": np.nan,
                "reference_class": "not_applicable",
                "n_examples": 0,
                "config_hash": config_hash(cfg),
                **m,
            }
        )

    pd.DataFrame(imp_rows).to_csv(rd / "channel_importance.csv", index=False)
    example_cols = [
        "fold",
        "fold_epoch_index",
        "subject",
        "epoch_id",
        "true_label",
        "seed",
        "predicted_label",
        "correct",
        "selection_rule",
        "config_hash",
    ]
    pd.DataFrame(example_rows).reindex(columns=example_cols).to_csv(
        rd / "explanation_examples.csv", index=False
    )
    cols = [
        "model",
        "subject_or_fold",
        "condition",
        "seed_a",
        "seed_b",
        "comparison",
        "bootstrap_replicate",
        "reference_class",
        "n_examples",
        "config_hash",
        "spearman",
        "cosine_similarity",
        "top_k_jaccard",
        "sign_agreement",
    ]
    pd.DataFrame(stab_rows).reindex(columns=cols).to_csv(
        rd / "explanation_stability.csv", index=False
    )
    logger.info(
        "Explanations complete: %d importance rows, %d stability rows",
        len(imp_rows),
        len(stab_rows),
    )


# --------------------------------------------------------------------------- #
# report + run-all
# --------------------------------------------------------------------------- #
def report(cfg: Config) -> None:
    from .reporting.figures import generate_all
    from .reporting.report import build_report

    ensure_run_manifest(cfg)
    generate_all(cfg.paths.results_dir, cfg.paths.figures_dir)
    build_report(cfg)


def run_all(cfg: Config, bundle: EpochBundle | None = None) -> None:
    bundle = prepare(cfg, bundle)
    make_splits(cfg, bundle)
    train_csp(cfg, bundle)
    train_cnn(cfg, bundle)
    evaluate(cfg, bundle)
    stress_test(cfg, bundle)
    explain(cfg, bundle)
    report(cfg)
    logger.info("run-all complete for profile '%s'", cfg.name)
