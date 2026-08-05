"""Subject-grouped outer/inner cross-validation splits.

Splitting epochs at random leaks subject identity: epochs from one person are highly
correlated (anatomy, electrode placement, session), so a classifier can memorise
subject-specific quirks and appear to generalise when it does not. We therefore split
by *subject* — no subject appears in more than one of {train, val, test} within a fold.

Uses ``StratifiedGroupKFold`` (keeps class balance across folds while respecting groups)
with a ``GroupKFold`` fallback. Splits are written to JSON so every model uses identical
subjects, and integrity assertions guarantee disjoint sets.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OuterFold:
    fold: int
    train_subjects: list[int]
    val_subjects: list[int]
    test_subjects: list[int]
    seed: int
    method: str
    config_hash: str = ""  # identity stamp; verified on load to prevent cross-profile mixing
    # Explicit outer-training population, stored independently from the designated CNN
    # train/validation holdout so malformed inner folds cannot redefine it implicitly.
    outer_train_subject_ids: list[int] = field(default_factory=list)
    # All grouped inner folds over the outer-train subjects, as (inner_train, inner_val)
    # subject-id lists. Used for genuine inner CV (CSP). ``val_subjects`` above is the
    # single designated grouped holdout (inner_folds[0].val) used only for CNN early
    # stopping — deliberately distinct from inner CV.
    inner_folds: list = field(default_factory=list)

    @property
    def outer_train_subjects(self) -> list[int]:
        """Subjects both final models train on (outer train + validation)."""
        if self.outer_train_subject_ids:
            return sorted(int(s) for s in self.outer_train_subject_ids)
        # Backward-compatible fallback for pre-step-4 split files.
        return sorted(set(self.train_subjects) | set(self.val_subjects))

    def assert_disjoint(self) -> None:
        tr, va, te = set(self.train_subjects), set(self.val_subjects), set(self.test_subjects)
        assert not (tr & va), f"train∩val not empty in fold {self.fold}: {tr & va}"
        assert not (tr & te), f"train∩test not empty in fold {self.fold}: {tr & te}"
        assert not (va & te), f"val∩test not empty in fold {self.fold}: {va & te}"
        outer_train = set(self.outer_train_subjects)
        assert tr | va == outer_train, (
            f"designated train∪val does not equal outer-train in fold {self.fold}"
        )
        inner_val_counts: Counter[int] = Counter()
        for j, pair in enumerate(self.inner_folds):
            itr, ival = set(pair[0]), set(pair[1])
            assert not (itr & ival), f"inner fold {j} train∩val overlap in fold {self.fold}"
            assert not (itr & te), f"inner fold {j} train leaks test in fold {self.fold}"
            assert not (ival & te), f"inner fold {j} val leaks test in fold {self.fold}"
            assert itr | ival == outer_train, (
                f"inner fold {j} train∪val does not equal outer-train in fold {self.fold}"
            )
            inner_val_counts.update(ival)
        if self.inner_folds:
            assert set(inner_val_counts) == outer_train, (
                f"inner validation folds do not cover outer-train in fold {self.fold}"
            )
            repeated = {s: n for s, n in inner_val_counts.items() if n != 1}
            assert not repeated, (
                f"outer-train subjects must appear in inner validation exactly once in fold "
                f"{self.fold}: {repeated}"
            )


def _grouped_kfold(y, groups, n_splits, seed, method):
    """Yield (train_idx, test_idx) using a stratified-group or plain group KFold."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    if method == "stratified_group_kfold":
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(splitter.split(np.zeros(len(y)), y, groups))
        except Exception as exc:  # noqa: BLE001 - fall back cleanly
            logger.warning("StratifiedGroupKFold failed (%s); using GroupKFold", exc)
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def make_outer_folds(y: np.ndarray, groups: np.ndarray, cfg: Config) -> list[OuterFold]:
    """Build nested subject-grouped folds.

    For each outer split, the outer-train portion is further split (subject-grouped)
    to carve out an inner validation set. Only outer-train subjects are used to pick the
    validation fold, so test subjects never influence model selection.
    """
    from ..utils.identity import config_hash

    seed = cfg.splits.seed
    method = cfg.splits.method
    chash = config_hash(cfg)
    outer = _grouped_kfold(y, groups, cfg.splits.outer_folds, seed, method)

    folds: list[OuterFold] = []
    for i, (tr_idx, te_idx) in enumerate(outer):
        train_val_subjects = np.unique(groups[tr_idx])
        test_subjects = np.unique(groups[te_idx])

        # Grouped inner CV within outer-train subjects only (test never participates).
        mask = np.isin(groups, train_val_subjects)
        y_in, g_in = y[mask], groups[mask]
        inner = _grouped_kfold(y_in, g_in, cfg.splits.inner_folds, seed + 1, method)
        inner_folds = [
            (
                sorted(int(s) for s in np.unique(g_in[in_tr])),
                sorted(int(s) for s in np.unique(g_in[in_val])),
            )
            for in_tr, in_val in inner
        ]
        # The first inner fold defines the CNN early-stopping holdout (val_subjects);
        # CSP uses every inner fold for genuine inner cross-validation.
        train_subjects, val_subjects = inner_folds[0]

        fold = OuterFold(
            fold=i,
            train_subjects=sorted(int(s) for s in train_subjects),
            val_subjects=sorted(int(s) for s in val_subjects),
            test_subjects=sorted(int(s) for s in test_subjects),
            seed=seed,
            method=method,
            config_hash=chash,
            outer_train_subject_ids=sorted(int(s) for s in train_val_subjects),
            inner_folds=inner_folds,
        )
        fold.assert_disjoint()
        folds.append(fold)
    return folds


def save_folds(folds: list[OuterFold], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in folds:
        p = out_dir / f"outer_fold_{f.fold}.json"
        with p.open("w") as fh:
            json.dump(asdict(f), fh, indent=2)
        paths.append(p)
    logger.info("Saved %d split files to %s", len(folds), out_dir)
    return paths


def load_folds(out_dir: Path, expected_hash: str | None = None) -> list[OuterFold]:
    """Load saved outer folds.

    If ``expected_hash`` is given, every fold must carry that identity stamp; otherwise a
    stale/foreign split file raises with a recovery instruction (no files are modified).
    """
    folds = []
    for p in sorted(Path(out_dir).glob("outer_fold_*.json")):
        with p.open() as fh:
            d = json.load(fh)
        fold = OuterFold(**d)
        if expected_hash is not None and fold.config_hash != expected_hash:
            raise RuntimeError(
                f"Split file {p.name} was generated for config '{fold.config_hash}' but "
                f"the active config hashes to '{expected_hash}'. Refusing to mix profiles. "
                f"Re-run make-splits for this profile into a clean splits dir. "
                f"No files were modified."
            )
        # Treat saved split JSON as untrusted input: a matching identity stamp does not
        # prove that subject partitions are still complete and leakage-free.
        fold.assert_disjoint()
        folds.append(fold)
    if not folds:
        raise FileNotFoundError(f"No split files found in {out_dir}. Run make-splits first.")
    return folds
