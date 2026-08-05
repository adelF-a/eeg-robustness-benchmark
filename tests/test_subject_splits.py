"""Subject-grouped split integrity: determinism, disjointness, class presence."""

import copy

import numpy as np
import pytest

from eeg_benchmark.data.splits import load_folds, make_outer_folds, save_folds


def test_folds_are_disjoint_by_subject(synthetic_bundle, smoke_config):
    folds = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)
    assert len(folds) == smoke_config.splits.outer_folds
    for f in folds:
        f.assert_disjoint()  # raises if any overlap
        tr, va, te = set(f.train_subjects), set(f.val_subjects), set(f.test_subjects)
        assert tr and va and te


def test_no_subject_leakage_across_all_folds(synthetic_bundle, smoke_config):
    folds = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)
    # Every subject appears as test in exactly one fold; never train+test in same fold.
    for f in folds:
        overlap = set(f.test_subjects) & (set(f.train_subjects) | set(f.val_subjects))
        assert not overlap


def test_both_classes_present_in_each_split(synthetic_bundle, smoke_config):
    b = synthetic_bundle
    folds = make_outer_folds(b.y, b.groups, smoke_config)
    for f in folds:
        for subjects in [f.train_subjects, f.val_subjects, f.test_subjects]:
            mask = np.isin(b.groups, subjects)
            assert set(np.unique(b.y[mask])) == {0, 1}


def test_deterministic_split_generation(synthetic_bundle, smoke_config):
    a = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)
    c = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)
    for fa, fc in zip(a, c, strict=True):
        assert fa.train_subjects == fc.train_subjects
        assert fa.val_subjects == fc.val_subjects
        assert fa.test_subjects == fc.test_subjects


def test_save_load_roundtrip(synthetic_bundle, smoke_config, tmp_path):
    folds = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)
    out = tmp_path / "splits"
    save_folds(folds, out)
    loaded = load_folds(out)
    assert [f.test_subjects for f in loaded] == [f.test_subjects for f in folds]


def test_inner_partition_must_equal_complete_outer_train(synthetic_bundle, smoke_config):
    fold = copy.deepcopy(
        make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)[0]
    )
    inner_train, inner_val = fold.inner_folds[1]
    fold.inner_folds[1] = (inner_train[1:], inner_val)
    with pytest.raises(AssertionError, match="does not equal outer-train"):
        fold.assert_disjoint()


def test_each_outer_train_subject_is_inner_validation_once(synthetic_bundle, smoke_config):
    fold = copy.deepcopy(
        make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)[0]
    )
    # Duplicate one otherwise valid partition: unions stay complete, but validation coverage
    # is now repeated for some subjects and absent for others.
    fold.inner_folds[1] = copy.deepcopy(fold.inner_folds[0])
    with pytest.raises(AssertionError, match="inner validation"):
        fold.assert_disjoint()


def test_load_revalidates_saved_split_integrity(synthetic_bundle, smoke_config, tmp_path):
    folds = make_outer_folds(synthetic_bundle.y, synthetic_bundle.groups, smoke_config)
    inner_train, inner_val = folds[0].inner_folds[1]
    folds[0].inner_folds[1] = (inner_train[1:], inner_val)
    out = tmp_path / "malformed_splits"
    save_folds(folds, out)

    with pytest.raises(AssertionError, match="does not equal outer-train"):
        load_folds(out)
