"""Exploratory data visualisation to build intuition about the EEG signals.

Read-only: no training, no model selection. Uses a small illustrative sample of subjects
(default 1-5, which are cached) and writes figures to reports/figures/exploration/.

Run:  python scripts/explore_data.py --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from eeg_benchmark.config import Config
from eeg_benchmark.data.preprocessing import build_dataset

MOTOR_CHANNELS = ["C3", "Cz", "C4"]  # C3=left motor, C4=right motor (contralateral control)
MU_BAND = (8.0, 13.0)


def _mne_info(ch_names, sfreq):
    import mne

    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
    return info


def fig_class_counts(bundle, out: Path):
    md = bundle.metadata
    counts = md.groupby(["subject", "event"]).size().unstack("event").fillna(0)
    ax = counts.plot(kind="bar", figsize=(8, 4))
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of epochs")
    ax.set_title("Class counts per subject (illustrative sample)")
    plt.tight_layout()
    plt.savefig(out / "explore_class_counts.png", dpi=120)
    plt.close()


def fig_class_average_traces(bundle, out: Path):
    """Class-averaged waveform on motor channels — the 'evoked' view."""
    t = np.arange(bundle.n_times) / bundle.sfreq
    idx = [bundle.ch_names.index(c) for c in MOTOR_CHANNELS if c in bundle.ch_names]
    fig, axes = plt.subplots(len(idx), 1, figsize=(8, 2.2 * len(idx)), sharex=True)
    for ax, ci, name in zip(np.atleast_1d(axes), idx, MOTOR_CHANNELS, strict=False):
        left = bundle.X[bundle.y == 0, ci, :].mean(axis=0)
        right = bundle.X[bundle.y == 1, ci, :].mean(axis=0)
        ax.plot(t, left * 1e6, label="left-fist imagery")
        ax.plot(t, right * 1e6, label="right-fist imagery")
        ax.set_ylabel(f"{name}\n(µV)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time within epoch (s, post-cue)")
    fig.suptitle("Class-averaged signal on motor channels")
    plt.tight_layout()
    plt.savefig(out / "explore_class_traces.png", dpi=120)
    plt.close()


def fig_psd(bundle, out: Path):
    """Power spectral density on C3/C4 for each class — shows mu/beta and lateralisation."""
    idx = [bundle.ch_names.index(c) for c in ["C3", "C4"] if c in bundle.ch_names]
    fig, axes = plt.subplots(1, len(idx), figsize=(5 * len(idx), 4), sharey=True)
    for ax, ci, name in zip(np.atleast_1d(axes), idx, ["C3", "C4"], strict=False):
        for label, lbl in [(0, "left"), (1, "right")]:
            X = bundle.X[bundle.y == label, ci, :]
            f, p = welch(X, fs=bundle.sfreq, nperseg=min(128, bundle.n_times), axis=1)
            ax.semilogy(f, p.mean(axis=0), label=f"{lbl} imagery")
        ax.axvspan(*MU_BAND, color="grey", alpha=0.15, label="mu band")
        ax.set_xlim(1, 40)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{name}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Power (V²/Hz)")
    fig.suptitle("Power spectrum on motor channels by class")
    plt.tight_layout()
    plt.savefig(out / "explore_psd.png", dpi=120)
    plt.close()


def fig_bandpower_topomap(bundle, out: Path):
    """Scalp maps of mu-band power for each class + their difference."""
    import mne

    info = _mne_info(bundle.ch_names, bundle.sfreq)

    # mu-band power per channel = variance of band-limited signal (data already 7-30 Hz,
    # here we additionally isolate 8-13 Hz via Welch integration).
    def band_power(X):  # X: [epochs, ch, time]
        f, p = welch(X, fs=bundle.sfreq, nperseg=min(128, bundle.n_times), axis=2)
        m = (f >= MU_BAND[0]) & (f <= MU_BAND[1])
        return p[:, :, m].mean(axis=(0, 2))  # -> [ch]

    left = band_power(bundle.X[bundle.y == 0])
    right = band_power(bundle.X[bundle.y == 1])
    diff = left - right

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, title in zip(
        axes, [left, right, diff], ["Left imagery", "Right imagery", "Left − Right"], strict=False
    ):
        im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap="RdBu_r")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Mu-band (8–13 Hz) power topography by class")
    plt.tight_layout()
    plt.savefig(out / "explore_bandpower_topomap.png", dpi=120)
    plt.close()


def fig_csp_patterns(bundle, out: Path):
    """What CSP learns: spatial patterns that best separate the two classes."""
    from mne.decoding import CSP

    info = _mne_info(bundle.ch_names, bundle.sfreq)
    csp = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False)
    csp.fit(bundle.X, bundle.y)
    fig = csp.plot_patterns(info, components=[0, 1, 2, 3], ch_type="eeg", show=False)
    fig.suptitle("CSP spatial patterns (top discriminative components)")
    fig.savefig(out / "explore_csp_patterns.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--subjects", default="1,2,3,4,5", help="comma-separated illustrative sample")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    subjects = [int(s) for s in args.subjects.split(",")]
    out = cfg.paths.figures_dir / "exploration"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Building illustrative bundle for subjects {subjects} ...")
    bundle = build_dataset(cfg, subjects)
    print(f"  X {bundle.X.shape} (epochs, channels, time), sfreq {bundle.sfreq} Hz")

    fig_class_counts(bundle, out)
    fig_class_average_traces(bundle, out)
    fig_psd(bundle, out)
    fig_bandpower_topomap(bundle, out)
    fig_csp_patterns(bundle, out)
    print(f"Wrote figures to {out}/:")
    for p in sorted(out.glob("explore_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
