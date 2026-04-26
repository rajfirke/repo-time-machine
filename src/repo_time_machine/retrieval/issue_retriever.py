"""Optional: fetch and search GitHub issues and PRs for context."""

from dataclasses import dataclass, field


@dataclass
class IssueRecord:
    number: int
    title: str
    body: str
    url: str
    labels: list[str] = field(default_factory=list)
    is_pr: bool = False
    score: float = 0.0


class IssueRetriever:
    """
    Fetches issues and PRs from a GitHub repo and ranks them by relevance.

    Requires GITHUB_TOKEN env var (free, read-only scope is enough).
    """

    def __init__(self, repo_slug: str):
        # repo_slug: "owner/repo"
        self.repo_slug = repo_slug
        self._records: list[IssueRecord] = []

    def fetch(self, max_items: int = 200) -> None:
        """Pull issues and PRs from the GitHub API and cache locally."""
        # TODO: implement using PyGitHub
        raise NotImplementedError

    def query(self, question: str, top_k: int = 5) -> list[IssueRecord]:
        """Return the top-k most relevant issues/PRs for the question."""
        # TODO: implement keyword + semantic scoring over _records
        raise NotImplementedError
