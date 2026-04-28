"""Load and chunk source files from a repository for embedding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", ".eggs"}
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".md", ".txt", ".rst",
    ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini",
    ".sh", ".bash", ".zsh",
    ".sql", ".graphql",
    ".dockerfile",
}
ALWAYS_INCLUDE = {"Makefile", "Dockerfile", "Justfile", "Gemfile", "Rakefile"}

MAX_FILE_SIZE = 512 * 1024  # skip files larger than 512 KB


@dataclass
class FileChunk:
    """A contiguous slice of a source file, ready for embedding."""

    file: str          # relative path from repo root
    start_line: int    # 1-indexed
    end_line: int      # inclusive
    content: str
    language: str      # derived from extension

    @property
    def loc(self) -> int:
        return self.end_line - self.start_line + 1

    def header(self) -> str:
        return f"{self.file}:{self.start_line}-{self.end_line} ({self.language})"


def _language_from_ext(ext: str) -> str:
    mapping = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".jsx": "javascript", ".go": "go",
        ".rs": "rust", ".java": "java", ".c": "c", ".cpp": "cpp",
        ".h": "c", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby",
        ".php": "php", ".swift": "swift", ".sh": "shell",
        ".bash": "shell", ".zsh": "shell", ".sql": "sql",
        ".md": "markdown", ".rst": "markdown", ".txt": "text",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
        ".json": "json", ".cfg": "ini", ".ini": "ini",
    }
    return mapping.get(ext, "text")


def iter_source_files(repo_path: str | Path) -> Iterator[Path]:
    """Yield all text-based source files, skipping generated/vendor dirs."""
    root = Path(repo_path).resolve()
    for path in root.rglob("*"):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_FILE_SIZE:
            continue
        if path.name in ALWAYS_INCLUDE or path.suffix in TEXT_EXTENSIONS:
            yield path


def load_file(path: Path) -> str:
    """Read a file and return its contents as a string."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def chunk_file(
    file_path: Path,
    repo_root: Path,
    chunk_lines: int = 60,
    overlap_lines: int = 10,
) -> list[FileChunk]:
    """
    Split a source file into overlapping chunks.

    chunk_lines:   target number of lines per chunk.
    overlap_lines: lines shared between consecutive chunks.

    Small files (≤ chunk_lines) produce a single chunk.
    """
    text = load_file(file_path)
    if not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    rel_path = str(file_path.relative_to(repo_root))
    lang = _language_from_ext(file_path.suffix)

    if len(lines) <= chunk_lines:
        return [FileChunk(
            file=rel_path,
            start_line=1,
            end_line=len(lines),
            content=text,
            language=lang,
        )]

    step = max(1, chunk_lines - overlap_lines)
    chunks: list[FileChunk] = []
    for start in range(0, len(lines), step):
        end = min(start + chunk_lines, len(lines))
        chunk_text = "".join(lines[start:end])
        chunks.append(FileChunk(
            file=rel_path,
            start_line=start + 1,
            end_line=end,
            content=chunk_text,
            language=lang,
        ))
        if end >= len(lines):
            break

    return chunks


def ingest_repo(
    repo_path: str | Path,
    chunk_lines: int = 60,
    overlap_lines: int = 10,
) -> list[FileChunk]:
    """
    Walk a repo and return all file chunks, ready for embedding.

    This is the main entry point for the ingestion pipeline.
    """
    root = Path(repo_path).resolve()
    all_chunks: list[FileChunk] = []
    file_count = 0

    for file_path in iter_source_files(root):
        chunks = chunk_file(file_path, root, chunk_lines, overlap_lines)
        all_chunks.extend(chunks)
        file_count += 1

    logger.info(
        "Ingested %d files → %d chunks from %s",
        file_count, len(all_chunks), root,
    )
    return all_chunks
