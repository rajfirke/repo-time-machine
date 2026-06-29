"""Persistence layer — manages the .rtm/ directory inside an indexed repo."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RTM_DIR = ".rtm"
CONFIG_FILE = "config.json"

REQUIRED_INDEX_FILES = ("code.faiss", "code_meta.json", "commits.faiss", "commits_meta.json")
OPTIONAL_INDEX_FILES = ("issues.faiss", "issues_meta.json")


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
    """Read index configuration. Returns None if corrupt or missing."""
    cfg = rtm_dir(repo_path) / CONFIG_FILE
    if not cfg.exists():
        return None
    try:
        with open(cfg, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupt config at %s: %s", cfg, exc)
        return None


def is_indexed(repo_path: str | Path) -> bool:
    """Check whether a repo has been fully indexed.

    Verifies config.json exists AND at least the core FAISS index files
    are present (code + commits). Issue indexes are optional.
    """
    d = rtm_dir(repo_path)
    if not (d / CONFIG_FILE).exists():
        return False
    return all((d / f).exists() for f in REQUIRED_INDEX_FILES)


@dataclass
class FileHealth:
    name: str
    present: bool
    size_bytes: int = 0
    required: bool = True


@dataclass
class IndexHealth:
    config_valid: bool
    config: dict = field(default_factory=dict)
    files: list[FileHealth] = field(default_factory=list)
    rtm_path: Path = field(default_factory=lambda: Path("."))

    @property
    def healthy(self) -> bool:
        return self.config_valid and all(f.present for f in self.files if f.required)

    @property
    def indexed(self) -> bool:
        return self.config_valid

    def to_dict(self) -> dict:
        """Serialise for --json output."""
        return {
            "indexed": self.indexed,
            "healthy": self.healthy,
            "rtm_path": str(self.rtm_path),
            "config": self.config,
            "files": [
                {
                    "name": f.name,
                    "present": f.present,
                    "size_bytes": f.size_bytes,
                    "required": f.required,
                }
                for f in self.files
            ],
        }


def index_health(repo_path: str | Path) -> IndexHealth:
    """Inspect the .rtm/ directory and return a structured health report."""
    d = rtm_dir(repo_path)
    cfg = load_config(repo_path)

    files: list[FileHealth] = []
    for name in REQUIRED_INDEX_FILES:
        p = d / name
        size = p.stat().st_size if p.exists() else 0
        files.append(FileHealth(name=name, present=p.exists(), size_bytes=size))
    for name in OPTIONAL_INDEX_FILES:
        p = d / name
        files.append(
            FileHealth(
                name=name,
                present=p.exists(),
                size_bytes=p.stat().st_size if p.exists() else 0,
                required=False,
            )
        )

    return IndexHealth(
        config_valid=cfg is not None and isinstance(cfg, dict),
        config=cfg if isinstance(cfg, dict) else {},
        files=files,
        rtm_path=d,
    )
