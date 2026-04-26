"""Search commit history and diffs for context relevant to a question."""

from dataclasses import dataclass
from repo_time_machine.ingestion.git_history import CommitRecord


@dataclass
class HistoryResult:
    commit: CommitRecord
    relevance: str
    score: float = 0.0


class HistoryRetriever:
    """
    Finds commits and diffs most related to a question.

    Combines keyword search and semantic scoring over commit messages and diffs.
    """

    def __init__(self, commits: list[CommitRecord]):
        self.commits = commits

    def query(self, question: str, top_k: int = 5) -> list[HistoryResult]:
        """Return the top-k most relevant commits for the question."""
        # TODO: implement keyword + semantic scoring
        raise NotImplementedError

    def timeline_for_file(self, relative_path: str) -> list[CommitRecord]:
        """Return commits that touched a specific file."""
        return [c for c in self.commits if relative_path in c.files_changed]
