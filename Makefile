PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
CONFIG ?= configs/smoke.yaml

.PHONY: install lint format test smoke full report clean

install:
	$(PY) -m pip install --upgrade pip
	$(PIP) install -e ".[dev,explain]"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

test:
	$(PY) -m pytest

smoke:
	$(PY) -m eeg_benchmark.cli run-all --config $(CONFIG)

full:
	$(PY) -m eeg_benchmark.cli run-all --config configs/full.yaml

report:
	$(PY) -m eeg_benchmark.cli report --config $(CONFIG)

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
