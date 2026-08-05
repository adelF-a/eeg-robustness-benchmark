"""Preprocessing output shape, channel-order preservation, and common-average reference."""

import numpy as np

from eeg_benchmark.data.epochs import epochs_from_raw, region_channels
from eeg_benchmark.data.preprocessing import preprocess_raw
from tests.conftest import SYNTH_CH_NAMES, make_synthetic_raw


def test_epoch_shape_and_channel_order():
    raw = make_synthetic_raw(run=4, n_events=6)
    X, y, md, ch_names, sfreq, _ = epochs_from_raw(
        raw, subject=3, run=4, tmin=0.0, tmax=1.0, baseline=None
    )
    assert X.ndim == 3  # [epochs, channels, time]
    assert X.shape[1] == len(ch_names)
    assert ch_names == SYNTH_CH_NAMES  # order preserved from the Raw
    assert sfreq == raw.info["sfreq"]


def test_common_average_reference_zeroes_channel_mean():
    raw = make_synthetic_raw(run=4)
    raw = preprocess_raw(raw, l_freq=7.0, h_freq=30.0, reference="average")
    data = raw.get_data()
    # After CAR the instantaneous mean across channels should be ~0.
    assert np.allclose(data.mean(axis=0), 0.0, atol=1e-10)


def test_region_channels_are_anatomically_sensible():
    frontal = region_channels(SYNTH_CH_NAMES, "frontal")
    central = region_channels(SYNTH_CH_NAMES, "central")
    left = region_channels(SYNTH_CH_NAMES, "left")
    right = region_channels(SYNTH_CH_NAMES, "right")
    assert "Fp1" in frontal and "C3" not in frontal
    assert "C3" in central and "C4" in central
    assert "C3" in left and "C4" not in left  # odd = left
    assert "C4" in right and "C3" not in right  # even = right
    # Midline channels (Cz, Fz) belong to neither hemisphere.
    assert "Cz" not in left and "Cz" not in right
