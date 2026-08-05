"""Command-line interface for the EEG robustness benchmark.

Each subcommand takes ``--config <path>`` and maps to a function in
:mod:`eeg_benchmark.pipeline`. Stages that depend on earlier artifacts (splits, trained
models) fail with an explicit, actionable message rather than a stack trace.
"""

from __future__ import annotations

import argparse
import sys

from . import pipeline
from .config import Config
from .utils.logging import get_logger

logger = get_logger("eeg_benchmark.cli")


def _load_cfg(args) -> Config:
    return Config.from_yaml(args.config)


def cmd_download(cfg: Config) -> None:
    from .data.download import download

    res = download(cfg)
    ok = sum(1 for v in res.values() if v)
    logger.info("Downloaded %d/%d subjects into %s", ok, len(res), cfg.paths.raw_dir)


def cmd_validate(cfg: Config) -> None:
    from .data.validation import valid_subjects, validate_data

    df = validate_data(cfg)
    good = valid_subjects(df)
    logger.info("Valid subjects (%d): %s", len(good), good)


def _guard(cfg: Config, need_splits=False, need_csp=False, need_cnn=False) -> None:
    if need_splits and not list(cfg.paths.splits_dir.glob("outer_fold_*.json")):
        sys.exit("No splits found. Run: make-splits --config <cfg>")
    if need_csp and not list((cfg.paths.results_dir / "models").glob("csp_fold*.joblib")):
        sys.exit("No CSP models found. Run: train-csp --config <cfg>")
    if need_cnn and not list((cfg.paths.results_dir / "models").glob("cnn_fold*_seed*.pt")):
        sys.exit("No CNN checkpoints found. Run: train-cnn --config <cfg>")


COMMANDS = {
    "download": lambda cfg: cmd_download(cfg),
    "validate-data": lambda cfg: cmd_validate(cfg),
    "prepare": lambda cfg: pipeline.prepare(cfg),
    "make-splits": lambda cfg: pipeline.make_splits(cfg),
    "train-csp": lambda cfg: (_guard(cfg, need_splits=True), pipeline.train_csp(cfg)),
    "train-cnn": lambda cfg: (_guard(cfg, need_splits=True), pipeline.train_cnn(cfg)),
    "evaluate": lambda cfg: pipeline.evaluate(cfg),
    "stress-test": lambda cfg: (
        _guard(cfg, need_splits=True, need_csp=True, need_cnn=True),
        pipeline.stress_test(cfg),
    ),
    "explain": lambda cfg: (
        _guard(cfg, need_splits=True, need_csp=True, need_cnn=True),
        pipeline.explain(cfg),
    ),
    "report": lambda cfg: pipeline.report(cfg),
    "run-all": lambda cfg: pipeline.run_all(cfg),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eeg-benchmark", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name, help=f"{name} stage")
        sp.add_argument("--config", required=True, help="Path to a YAML config file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_cfg(args)
    cfg.paths.ensure()
    logger.info("Running '%s' with config '%s' (%s)", args.command, args.config, cfg.name)
    COMMANDS[args.command](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
