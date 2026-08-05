"""Config identity and profile isolation.

Every run gets a ``config_hash`` derived from the scientific configuration (everything
except output *paths*). Artifacts (split files, checkpoints) are stamped with it, and a
``run_manifest.json`` records the identity of a results directory. Stages verify these
stamps and **refuse to mix** artifacts from different configurations — importantly,
without deleting anything: a mismatch raises a precise, non-destructive recovery message.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone

from ..config import Config


def config_hash(cfg: Config) -> str:
    """Stable 12-char hash of the scientific config (output paths excluded).

    Two profiles that differ only in ``results_dir``/``reports_dir`` hash the same; two
    that differ in any experimental setting (seeds, subjects, splits, model, …) differ —
    so two configs both named ``smoke`` cannot silently share a directory's artifacts.
    """
    raw = copy.deepcopy(cfg.raw) if cfg.raw else {}
    raw.pop("paths", None)
    blob = json.dumps(raw, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


# Files allowed to exist in a results_dir that is still considered "unclaimed".
_PLACEHOLDERS = {".gitkeep", "run_manifest.json", "artifacts.json"}


def _manifest_path(cfg: Config):
    return cfg.paths.results_dir / "run_manifest.json"


def _artifacts_path(cfg: Config):
    return cfg.paths.results_dir / "artifacts.json"


def _has_artifacts(cfg: Config) -> bool:
    """True if the results_dir already holds real files (empty dirs/placeholders ignored)."""
    rd = cfg.paths.results_dir
    if not rd.exists():
        return False
    return any(p.is_file() and p.name not in _PLACEHOLDERS for p in rd.rglob("*"))


def check_run_manifest(cfg: Config) -> str:
    """Return the active config hash, raising if the results_dir belongs to another config.

    A missing manifest is not resolved here (see :func:`ensure_run_manifest`). A present,
    mismatching manifest raises with a recovery instruction; no files are modified.
    """
    h = config_hash(cfg)
    p = _manifest_path(cfg)
    if p.exists():
        existing = json.loads(p.read_text()).get("config_hash")
        if existing != h:
            raise RuntimeError(
                f"results_dir '{cfg.paths.results_dir}' already belongs to config "
                f"{existing} but the active config hashes to {h}. Refusing to mix "
                f"profiles. Point this profile at a fresh results_dir/reports_dir, or "
                f"re-run make-splits into a clean directory. No files were modified."
            )
    return h


def ensure_run_manifest(cfg: Config) -> str:
    """Verify or claim the results_dir for the active config; return the config hash.

    A manifest is created only for a directory that is empty or holds nothing but allowed
    placeholders. If artifacts already exist without a manifest (a legacy/pre-identity
    run), this refuses with a migration instruction rather than silently claiming them.
    """
    h = check_run_manifest(cfg)
    p = _manifest_path(cfg)
    if p.exists():
        return h
    if _has_artifacts(cfg):
        raise RuntimeError(
            f"results_dir '{cfg.paths.results_dir}' contains artifacts but no "
            f"run_manifest.json (a legacy run predating identity stamping). Refusing to "
            f"claim it. Migrate: move or remove those files, or point this profile at a "
            f"fresh results_dir/reports_dir, then re-run. No files were modified."
        )
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {"config_hash": h, "name": cfg.name, "created": datetime.now(timezone.utc).isoformat()},
            indent=2,
        )
    )
    return h


def register_artifact(cfg: Config, name: str, stage: str) -> None:
    """Record that ``stage`` produced artifact ``name`` under the active config identity."""
    ensure_run_manifest(cfg)
    p = _artifacts_path(cfg)
    h = config_hash(cfg)
    if p.exists():
        data = json.loads(p.read_text())
        if data.get("config_hash") != h:
            raise RuntimeError(
                f"artifacts.json in '{cfg.paths.results_dir}' belongs to config "
                f"{data.get('config_hash')}, not {h}. Refusing to mix. No files were modified."
            )
    else:
        data = {"config_hash": h, "artifacts": {}}
    data["artifacts"][name] = {
        "stage": stage,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(data, indent=2))


def require_artifact(cfg: Config, name: str) -> None:
    """Raise unless artifact ``name`` was recorded for the active config (refuses legacy)."""
    p = _artifacts_path(cfg)
    h = config_hash(cfg)
    if not p.exists():
        raise RuntimeError(
            f"No artifacts manifest in '{cfg.paths.results_dir}'; '{name}' has no recorded "
            f"identity. Re-run its producing stage for this profile. No files were modified."
        )
    data = json.loads(p.read_text())
    if data.get("config_hash") != h:
        raise RuntimeError(
            f"artifacts.json belongs to config {data.get('config_hash')}, not {h}. "
            f"Refusing incompatible input '{name}'. No files were modified."
        )
    if name not in data.get("artifacts", {}):
        raise RuntimeError(
            f"Artifact '{name}' was not recorded for this config; refusing legacy/incompatible "
            f"input. Re-run its producing stage. No files were modified."
        )
