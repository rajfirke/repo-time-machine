"""Load and chunk source files from a repository for embedding."""

from pathlib import Path
from typing import Iterator


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".sh",
}


def iter_source_files(repo_path: str | Path) -> Iterator[Path]:
    """Yield all text-based source files, skipping generated/vendor dirs."""
    root = Path(repo_path)
    for path in root.rglob("*"):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.is_file() and path.suffix in TEXT_EXTENSIONS:
            yield path


def load_file(path: Path) -> str:
    """Read a file and return its contents as a string."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


# TODO: implement chunk_file() to split long files into overlapping windows
