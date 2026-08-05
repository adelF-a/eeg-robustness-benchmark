"""Config identity + profile isolation (step 3).

Two configurations that happen to share a name (or an output directory) must not mix:
artifacts are stamped with a config hash, and stages reject incompatible artifacts with a
precise, non-destructive recovery instruction.
"""

from __future__ import annotations

import json

import pytest

from eeg_benchmark import pipeline
from eeg_benchmark.config import Config
from eeg_benchmark.data.splits import load_folds, make_outer_folds, save_folds
from eeg_benchmark.utils.identity import check_run_manifest, config_hash, ensure_run_manifest


def _cfg(tmp_path, *, results="results", seed=20260718) -> Config:
    return Config.from_dict(
        {
            "name": "smoke",
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "results_dir": str(tmp_path / results),
                "reports_dir": str(tmp_path / "reports"),
            },
            "data": {"subjects": [1, 2, 3, 4], "runs": [4, 8, 12]},
            "preprocessing": {"cache": False},
            "splits": {"outer_folds": 2, "inner_folds": 2, "seed": seed},
        }
    )


def test_config_hash_ignores_paths_but_tracks_science(tmp_path):
    a = _cfg(tmp_path, results="results", seed=1)
    same_science = _cfg(tmp_path, results="OTHER_DIR", seed=1)  # only paths differ
    diff_science = _cfg(tmp_path, results="results", seed=2)  # a real setting differs
    assert config_hash(a) == config_hash(same_science)
    assert config_hash(a) != config_hash(diff_science)


def test_ensure_writes_manifest_and_check_rejects_mismatch(tmp_path):
    a = _cfg(tmp_path, seed=1)
    ensure_run_manifest(a)
    assert (a.paths.results_dir / "run_manifest.json").exists()
    assert check_run_manifest(a) == config_hash(a)  # same config: fine

    # A different config pointed at the same results_dir must be rejected, not merged.
    b = _cfg(tmp_path, seed=2)
    with pytest.raises(RuntimeError, match="make-splits"):
        check_run_manifest(b)
    with pytest.raises(RuntimeError, match="make-splits"):
        ensure_run_manifest(b)


def test_stale_folds_rejected(tmp_path, synthetic_bundle):
    a = _cfg(tmp_path, seed=1)
    b = _cfg(tmp_path, seed=2)
    folds = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, a)
    out = tmp_path / "splits"
    save_folds(folds, out)

    # Matching hash loads; mismatching hash raises with recovery guidance.
    assert load_folds(out, expected_hash=config_hash(a))
    with pytest.raises(RuntimeError, match="make-splits"):
        load_folds(out, expected_hash=config_hash(b))
    # No hash requested: backward-compatible load still works.
    assert load_folds(out)


def test_saved_folds_carry_config_hash(smoke_config, synthetic_bundle):
    pipeline.prepare(smoke_config, synthetic_bundle)
    pipeline.make_splits(smoke_config, synthetic_bundle)
    h = config_hash(smoke_config)
    for p in smoke_config.paths.splits_dir.glob("outer_fold_*.json"):
        d = json.loads(p.read_text())
        assert d["config_hash"] == h


def test_checkpoint_hash_verified(tmp_path):
    import numpy as np
    import torch

    from eeg_benchmark.models.compact_eegnet import CompactEEGNet

    cfg = _cfg(tmp_path)
    arch = {"n_channels": 4, "n_times": 16, "F1": 8, "D": 2, "kernel_length": 8, "dropout": 0.25}
    model = CompactEEGNet(n_classes=2, **arch)
    models_dir = cfg.paths.results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": arch,
            "std_mean": np.zeros(4, dtype=np.float32),
            "std_std": np.ones(4, dtype=np.float32),
            "ch_names": ["a", "b", "c", "d"],
            "config_hash": "deadbeefdead",  # wrong on purpose
        },
        models_dir / "cnn_fold0_seed0.pt",
    )
    with pytest.raises(RuntimeError, match="config"):
        pipeline.load_cnn(cfg, 0, 0, device="cpu")
