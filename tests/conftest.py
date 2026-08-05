"""Shared synthetic fixtures. Tests never download the PhysioNet dataset.

The synthetic generator injects a lateralised, variance-based class difference (higher
oscillatory power over left-central channels for class 0, right-central for class 1),
mimicking the contralateral event-related desynchronisation of real motor imagery. This
makes the signal genuinely learnable by both CSP (variance features) and the CNN, so
end-to-end tests exercise real training dynamics rather than noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg_benchmark.config import Config
from eeg_benchmark.data.epochs import EpochBundle

# 16 valid 10-05 names spanning frontal / central / left / right for region tests.
SYNTH_CH_NAMES = [
    "Fp1",
    "Fp2",
    "F3",
    "Fz",
    "F4",
    "FC3",
    "FC4",
    "C3",
    "Cz",
    "C4",
    "CP3",
    "CP4",
    "P3",
    "Pz",
    "P4",
    "Oz",
]
LEFT_CENTRAL = ["FC3", "C3", "CP3"]
RIGHT_CENTRAL = ["FC4", "C4", "CP4"]


def make_synthetic_bundle(
    n_subjects: int = 6,
    epochs_per_class: int = 20,
    n_times: int = 128,
    sfreq: float = 160.0,
    seed: int = 0,
) -> EpochBundle:
    """Build a class-separable synthetic EpochBundle with subject-specific variation."""
    rng = np.random.default_rng(seed)
    ch_names = list(SYNTH_CH_NAMES)
    n_ch = len(ch_names)
    left_idx = [ch_names.index(c) for c in LEFT_CENTRAL]
    right_idx = [ch_names.index(c) for c in RIGHT_CENTRAL]
    t = np.arange(n_times) / sfreq

    Xs, ys, gs, mds = [], [], [], []
    for subj in range(1, n_subjects + 1):
        subj_gain = 1.0 + 0.2 * rng.standard_normal()  # per-subject amplitude drift
        freq = 10.0 + rng.uniform(-1.0, 1.0)  # mu-band
        for label in (0, 1):
            for e in range(epochs_per_class):
                x = 0.5 * rng.standard_normal((n_ch, n_times)).astype(np.float32)
                active = left_idx if label == 0 else right_idx
                # Independent phase per active channel avoids perfectly collinear channels
                # (which would make the class covariance singular for CSP).
                for ci in active:
                    osc = subj_gain * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
                    x[ci] += (1.2 * osc).astype(np.float32)
                Xs.append(x)
                ys.append(label)
                gs.append(subj)
                mds.append(
                    {
                        "subject": subj,
                        "run": 4,
                        "event": "left_fist" if label == 0 else "right_fist",
                        "label": label,
                        "epoch_id": f"s{subj:03d}_r04_e{label}{e:03d}",
                    }
                )
    return EpochBundle(
        X=np.stack(Xs),
        y=np.array(ys),
        groups=np.array(gs),
        ch_names=ch_names,
        sfreq=sfreq,
        metadata=pd.DataFrame(mds),
    )


def make_synthetic_raw(run: int = 4, sfreq: float = 160.0, n_events: int = 6):
    """Build a small montaged MNE RawArray with T0/T1/T2 annotations for one run."""
    import mne

    ch_names = list(SYNTH_CH_NAMES)
    n_ch = len(ch_names)
    dur = 2.0  # seconds between events
    n_times = int(sfreq * dur * (n_events + 2))
    rng = np.random.default_rng(run)
    data = 1e-6 * rng.standard_normal((n_ch, n_times))  # volts
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"), on_missing="ignore", verbose=False
    )

    onsets, descs = [], []
    labels = ["T1", "T2"]
    for i in range(n_events):
        onsets.append(1.0 + i * dur)
        descs.append(labels[i % 2])
    annot = mne.Annotations(onset=onsets, duration=[0.1] * len(onsets), description=descs)
    raw.set_annotations(annot, verbose=False)
    return raw


@pytest.fixture
def synthetic_bundle() -> EpochBundle:
    return make_synthetic_bundle()


@pytest.fixture
def small_bundle() -> EpochBundle:
    return make_synthetic_bundle(n_subjects=4, epochs_per_class=10, n_times=96)


@pytest.fixture
def smoke_config(tmp_path) -> Config:
    """A tiny in-memory config pointing at a temp dir, for end-to-end tests."""
    raw = {
        "name": "test",
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "results_dir": str(tmp_path / "results"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "data": {"subjects": [1, 2, 3, 4], "runs": [4, 8, 12]},
        "preprocessing": {"cache": False},
        "splits": {
            "outer_folds": 2,
            "inner_folds": 2,
            "seed": 42,
            "method": "stratified_group_kfold",
        },
        "csp": {
            "n_components_grid": [2],
            "reg_grid": [None],
            "C_grid": [1.0],
            "class_weight_grid": [None],
        },
        "neural": {
            "seeds": [0, 1],
            "epochs": 2,
            "batch_size": 16,
            "kernel_length": 32,
            "patience": 5,
        },
        "robustness": {
            "random_fractions": [0.25],
            "n_random_masks": 1,
            "regions": ["frontal", "central", "left", "right"],
        },
        "explanations": {
            "n_examples_per_subject": 3,
            "ig_steps": 4,
            "top_k": 4,
            "n_bootstrap_explain": 3,
        },
        "bootstrap": {"n_boot": 50, "ci": 0.95, "seed": 1},
        "latency": {"n_warmup": 1, "n_repeats": 3, "batch_size": 1},
    }
    cfg = Config.from_dict(raw)
    cfg.paths.ensure()
    return cfg
