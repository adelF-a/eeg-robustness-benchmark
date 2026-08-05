"""CLI wiring: parser accepts all commands; dependency guards fail clearly."""

import pytest
import yaml

from eeg_benchmark.cli import COMMANDS, build_parser, main


def test_parser_exposes_all_commands():
    parser = build_parser()
    # Every documented command should parse with a --config argument.
    for name in COMMANDS:
        args = parser.parse_args([name, "--config", "x.yaml"])
        assert args.command == name
        assert args.config == "x.yaml"


def _write_cfg(tmp_path, smoke_config) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(smoke_config.raw))
    return str(p)


def test_train_csp_without_splits_exits(tmp_path, smoke_config):
    cfg_path = _write_cfg(tmp_path, smoke_config)
    with pytest.raises(SystemExit):
        main(["train-csp", "--config", cfg_path])


def test_report_command_runs_without_results(tmp_path, smoke_config):
    cfg_path = _write_cfg(tmp_path, smoke_config)
    # report should tolerate missing result files and still write a report.
    assert main(["report", "--config", cfg_path]) == 0
    assert (smoke_config.paths.reports_dir / "report.md").exists()
