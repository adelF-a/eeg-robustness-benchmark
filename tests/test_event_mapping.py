"""T1/T2 event mapping for left-vs-right runs; refusal of hands-vs-feet runs."""

import numpy as np
import pytest

from eeg_benchmark.data.epochs import EVENT_ID, epochs_from_raw
from tests.conftest import make_synthetic_raw


@pytest.mark.parametrize("run", [4, 8, 12])
def test_t1_t2_map_to_left_right(run):
    raw = make_synthetic_raw(run=run)
    X, y, md, ch_names, sfreq, n_rej = epochs_from_raw(
        raw, subject=1, run=run, tmin=0.0, tmax=1.0, baseline=None
    )
    assert set(np.unique(y)).issubset({EVENT_ID["left_fist"], EVENT_ID["right_fist"]})
    # Synthetic raw alternates T1,T2 -> labels 0,1,0,1...
    assert (y == 0).sum() > 0 and (y == 1).sum() > 0
    assert list(md["label"]) == list(y)
    assert ch_names == raw.ch_names


@pytest.mark.parametrize("run", [6, 10, 14])
def test_hands_feet_runs_are_rejected(run):
    raw = make_synthetic_raw(run=4)  # data content irrelevant; the run number is the guard
    with pytest.raises(ValueError, match="hands-vs-feet"):
        epochs_from_raw(raw, subject=1, run=run, tmin=0.0, tmax=1.0, baseline=None)
