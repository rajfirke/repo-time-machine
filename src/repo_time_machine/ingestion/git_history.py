"""Extract commit messages, diffs, and per-file change history from git."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CommitRecord:
    sha: str
    author: str
    date: str
    message: str
    files_changed: list[str] = field(default_factory=list)
    diff_summary: str = ""


def extract_history(repo_path: str | Path) -> list[CommitRecord]:
    """
    Return a list of CommitRecord objects for every commit in the repo.

    Uses GitPython under the hood.
    """
    # TODO: implement using git.Repo from gitpython
    raise NotImplementedError


def file_timeline(repo_path: str | Path, relative_file_path: str) -> list[CommitRecord]:
    """
    Return commits that touched a specific file, ordered newest-first.
    """
    # TODO: implement using git.Repo.iter_commits(paths=relative_file_path)
    raise NotImplementedError
