"""Configuration schema and YAML loading.

All experimental choices live in YAML config files (``configs/*.yaml``) rather than
hidden constants. This module parses a config into typed dataclasses and resolves
paths. A copy of the raw config dict is retained so it can be saved alongside results.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def parse_subject_spec(spec: Any) -> list[int]:
    """Parse a subject specification into a sorted list of ints.

    Accepts a list of ints, or a string like ``"1-109"`` or ``"1-5,10,12-14"``.
    """
    if isinstance(spec, list):
        return sorted(int(s) for s in spec)
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        out: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-")
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
        return sorted(set(out))
    raise TypeError(f"Cannot parse subject spec: {spec!r}")


@dataclass
class Paths:
    data_dir: Path
    results_dir: Path
    reports_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.results_dir / "splits"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    def ensure(self) -> None:
        for p in [
            self.data_dir,
            self.results_dir,
            self.reports_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.splits_dir,
            self.figures_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class DataCfg:
    subjects: list[int]
    runs: list[int]
    sfreq_expected: float = 160.0
    n_channels_expected: int = 64


@dataclass
class PreprocCfg:
    l_freq: float = 7.0
    h_freq: float = 30.0
    reference: str = "average"
    tmin: float = 1.0
    tmax: float = 4.0
    baseline: Any | None = None
    reject_criteria: dict | None = None
    cache: bool = True


@dataclass
class SplitCfg:
    outer_folds: int = 5
    inner_folds: int = 3
    seed: int = 20260718
    method: str = "stratified_group_kfold"


@dataclass
class CSPCfg:
    n_components_grid: list[int] = field(default_factory=lambda: [2, 4, 6, 8])
    reg_grid: list[Any] = field(default_factory=lambda: [None, 0.1, "ledoit_wolf"])
    C_grid: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    class_weight_grid: list[Any] = field(default_factory=lambda: [None, "balanced"])


@dataclass
class NeuralCfg:
    seeds: list[int] = field(default_factory=lambda: [0])
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.25
    F1: int = 8
    D: int = 2
    kernel_length: int = 64
    patience: int = 15
    early_stopping_metric: str = "balanced_accuracy"


@dataclass
class RobustnessCfg:
    random_fractions: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5])
    n_random_masks: int = 10
    regions: list[str] = field(default_factory=lambda: ["frontal", "central", "left", "right"])
    imputation: str = "train_channel_mean"


@dataclass
class ExplainCfg:
    n_examples_per_subject: int = 8
    ig_baseline: str = "zeros"
    ig_steps: int = 50
    top_k: int = 8
    noise_std: float = 0.05
    n_bootstrap_explain: int = 50


@dataclass
class BootstrapCfg:
    n_boot: int = 1000
    ci: float = 0.95
    seed: int = 12345


@dataclass
class LatencyCfg:
    n_warmup: int = 10
    n_repeats: int = 200
    batch_size: int = 1


@dataclass
class Config:
    name: str
    paths: Paths
    data: DataCfg
    preprocessing: PreprocCfg
    splits: SplitCfg
    csp: CSPCfg
    neural: NeuralCfg
    robustness: RobustnessCfg
    explanations: ExplainCfg
    bootstrap: BootstrapCfg
    latency: LatencyCfg
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        with path.open() as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> Config:
        raw = copy.deepcopy(raw)
        p = raw.get("paths", {})
        paths = Paths(
            data_dir=Path(p.get("data_dir", "data")),
            results_dir=Path(p.get("results_dir", "results")),
            reports_dir=Path(p.get("reports_dir", "reports")),
        )
        d = raw.get("data", {})
        data = DataCfg(
            subjects=parse_subject_spec(d.get("subjects", [1])),
            runs=list(d.get("runs", [4, 8, 12])),
            sfreq_expected=float(d.get("sfreq_expected", 160.0)),
            n_channels_expected=int(d.get("n_channels_expected", 64)),
        )
        return cls(
            name=raw.get("name", "unnamed"),
            paths=paths,
            data=data,
            preprocessing=PreprocCfg(**raw.get("preprocessing", {})),
            splits=SplitCfg(**raw.get("splits", {})),
            csp=CSPCfg(**raw.get("csp", {})),
            neural=NeuralCfg(**raw.get("neural", {})),
            robustness=RobustnessCfg(**raw.get("robustness", {})),
            explanations=ExplainCfg(**raw.get("explanations", {})),
            bootstrap=BootstrapCfg(**raw.get("bootstrap", {})),
            latency=LatencyCfg(**raw.get("latency", {})),
            raw=raw,
        )

    def save(self, path: str | Path) -> None:
        """Persist the raw config dict next to results for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump(self.raw, fh, sort_keys=False)
