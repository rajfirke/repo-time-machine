"""Persistence layer — manages the .rtm/ directory inside an indexed repo."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RTM_DIR = ".rtm"
CONFIG_FILE = "config.json"


def rtm_dir(repo_path: str | Path) -> Path:
    """Return the .rtm/ directory path for a given repo."""
    return Path(repo_path).resolve() / RTM_DIR


def ensure_rtm_dir(repo_path: str | Path) -> Path:
    """Create .rtm/ if it doesn't exist and return its path."""
    d = rtm_dir(repo_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_config(repo_path: str | Path, config: dict) -> None:
    """Write index configuration to .rtm/config.json."""
    d = ensure_rtm_dir(repo_path)
    with open(d / CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info("Saved config to %s", d / CONFIG_FILE)


def load_config(repo_path: str | Path) -> dict | None:
    """Read index configuration. Returns None if not indexed yet."""
    cfg = rtm_dir(repo_path) / CONFIG_FILE
    if not cfg.exists():
        return None
    with open(cfg, encoding="utf-8") as f:
        return json.load(f)


def is_indexed(repo_path: str | Path) -> bool:
    """Check whether a repo has been indexed by looking for .rtm/config.json."""
    return (rtm_dir(repo_path) / CONFIG_FILE).exists()
