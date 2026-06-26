"""Extract commit messages, diffs, and per-file change history from git."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

import git
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

logger = logging.getLogger(__name__)


@dataclass
class CommitRecord:
    sha: str
    author: str
    date: str
    message: str
    files_changed: list[str] = field(default_factory=list)
    diff_summary: str = ""

    @property
    def short_sha(self) -> str:
        return self.sha[:8]

    def oneline(self) -> str:
        first_line = self.message.strip().split("\n", 1)[0]
        return f"{self.short_sha} {self.date} {first_line}"


def _compute_diffs(commit):
    """Compute the diff for a commit against its first parent (or empty tree for root commits)."""
    parent = commit.parents[0] if commit.parents else None
    if parent is None:
        return commit.diff(git.NULL_TREE)
    return parent.diff(commit)


def _format_diffs(diffs) -> str:
    """Build a compact diff summary (action + path per file) from a pre-computed diff list."""
    lines = []
    for d in diffs:
        action = "M"
        if d.new_file:
            action = "A"
        elif d.deleted_file:
            action = "D"
        elif d.renamed_file:
            action = "R"
        path = d.b_path or d.a_path or "?"
        lines.append(f"{action} {path}")
    return "\n".join(lines)


def _files_from_diffs(diffs) -> list[str]:
    """Extract unique file paths from a pre-computed diff list."""
    paths: list[str] = []
    for d in diffs:
        p = d.b_path or d.a_path
        if p and p not in paths:
            paths.append(p)
    return paths


def _to_record(commit) -> CommitRecord:
    try:
        diffs = _compute_diffs(commit)
    except (GitCommandError, ValueError):
        diffs = []

    return CommitRecord(
        sha=commit.hexsha,
        author=str(commit.author),
        date=commit.committed_datetime.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        message=commit.message.strip(),
        files_changed=_files_from_diffs(diffs),
        diff_summary=_format_diffs(diffs),
    )


def extract_history(
    repo_path: str | Path,
    max_commits: int = 500,
    branch: str | None = None,
) -> list[CommitRecord]:
    """
    Return CommitRecord objects for the repo, newest-first.

    max_commits: cap to avoid blowing up on huge repos.
    branch:      specific branch to walk (defaults to active branch).
    """
    repo_path = Path(repo_path).resolve()
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        logger.error("Not a git repository: %s", repo_path)
        return []

    try:
        repo.head.commit
    except ValueError:
        logger.warning("Repository has no commits: %s", repo_path)
        return []

    if repo.head.is_detached:
        ref = repo.head.commit
    elif branch:
        ref = branch
    else:
        ref = repo.active_branch

    records: list[CommitRecord] = []
    for commit in repo.iter_commits(ref, max_count=max_commits):
        records.append(_to_record(commit))

    logger.info("Extracted %d commits from %s", len(records), repo_path)
    return records


def file_timeline(
    repo_path: str | Path,
    relative_file_path: str,
    max_commits: int = 100,
) -> list[CommitRecord]:
    """
    Return commits that touched a specific file, newest-first.
    """
    repo_path = Path(repo_path).resolve()
    try:
        repo = Repo(repo_path)
    except InvalidGitRepositoryError:
        logger.error("Not a git repository: %s", repo_path)
        return []

    try:
        repo.head.commit
    except ValueError:
        logger.warning("Repository has no commits: %s", repo_path)
        return []

    records: list[CommitRecord] = []
    for commit in repo.iter_commits(paths=relative_file_path, max_count=max_commits):
        records.append(_to_record(commit))

    logger.info("Found %d commits touching '%s'", len(records), relative_file_path)
    return records
