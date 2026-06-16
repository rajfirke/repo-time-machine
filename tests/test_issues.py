"""Tests for the issue/PR retrieval layer — uses mock embedder, no GitHub calls."""

from unittest.mock import MagicMock

import numpy as np

from repo_time_machine.retrieval.issue_retriever import (
    IssueRecord,
    IssueResult,
    IssueRetriever,
    _issue_to_text,
    _keyword_score,
    _tokenize,
)

DIM = 384


def _fake_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.dim = DIM

    def _embed(texts, batch_size=64):
        rng = np.random.RandomState(99)
        vecs = rng.randn(len(texts), DIM).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    embedder.embed = _embed
    return embedder


def _sample_issues() -> list[IssueRecord]:
    return [
        IssueRecord(
            number=42,
            title="Fix validation crash on empty input",
            body="When the input is empty, validate() raises IndexError instead of ValueError.",
            url="https://github.com/test/repo/issues/42",
            labels=["bug", "good first issue"],
            is_pr=False,
            state="open",
        ),
        IssueRecord(
            number=55,
            title="Add retry logic to API client",
            body="The API client should retry on 429 and 5xx responses with exponential backoff.",
            url="https://github.com/test/repo/issues/55",
            labels=["enhancement"],
            is_pr=False,
            state="open",
        ),
        IssueRecord(
            number=60,
            title="Refactor CLI to use Typer",
            body="Replace argparse with typer for better UX and type checking.",
            url="https://github.com/test/repo/pull/60",
            labels=["refactor"],
            is_pr=True,
            state="closed",
        ),
    ]


# ---------------------------------------------------------------------------
# IssueRetriever
# ---------------------------------------------------------------------------


class TestIssueRetriever:
    def test_build_and_query(self, tmp_path):
        embedder = _fake_embedder()
        ret = IssueRetriever(tmp_path / ".rtm", embedder)
        count = ret.build_from_records(_sample_issues())
        assert count == 3

        results = ret.query("validation crash", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, IssueResult) for r in results)
        assert results[0].score >= results[1].score

    def test_persist_and_reload(self, tmp_path):
        embedder = _fake_embedder()
        rtm = tmp_path / ".rtm"
        ret = IssueRetriever(rtm, embedder)
        ret.build_from_records(_sample_issues())

        ret2 = IssueRetriever(rtm, embedder)
        assert ret2.load() is True
        assert ret2.available

        results = ret2.query("retry API", top_k=1)
        assert len(results) == 1

    def test_empty_returns_empty(self, tmp_path):
        embedder = _fake_embedder()
        ret = IssueRetriever(tmp_path / ".rtm", embedder)
        ret.build_from_records([])
        assert ret.query("anything") == []

    def test_load_missing_returns_false(self, tmp_path):
        embedder = _fake_embedder()
        ret = IssueRetriever(tmp_path / ".rtm", embedder)
        assert ret.load() is False
        assert not ret.available

    def test_top_k_capped(self, tmp_path):
        embedder = _fake_embedder()
        ret = IssueRetriever(tmp_path / ".rtm", embedder)
        ret.build_from_records(_sample_issues())
        results = ret.query("test", top_k=100)
        assert len(results) == 3

    def test_result_has_pr_tag(self, tmp_path):
        embedder = _fake_embedder()
        ret = IssueRetriever(tmp_path / ".rtm", embedder)
        ret.build_from_records(_sample_issues())
        results = ret.query("refactor CLI typer", top_k=3)
        pr_results = [r for r in results if r.issue.is_pr]
        assert len(pr_results) >= 1

    def test_no_slug_fetch_returns_zero(self, tmp_path):
        embedder = _fake_embedder()
        ret = IssueRetriever(tmp_path / ".rtm", embedder, repo_slug="")
        count = ret.fetch_and_build()
        assert count == 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestIssueToText:
    def test_formats_issue(self):
        rec = _sample_issues()[0]
        text = _issue_to_text(rec)
        assert "[Issue #42]" in text
        assert "validation" in text.lower()
        assert "bug" in text

    def test_formats_pr(self):
        rec = _sample_issues()[2]
        text = _issue_to_text(rec)
        assert "[PR #60]" in text


class TestIssueTokenize:
    def test_basic(self):
        tokens = _tokenize("Fix validation crash")
        assert "fix" in tokens
        assert "validation" in tokens
        assert "crash" in tokens

    def test_short_dropped(self):
        tokens = _tokenize("a b cd")
        assert "a" not in tokens
        assert "cd" in tokens


class TestIssueKeywordScore:
    def test_matching(self):
        q_tokens = _tokenize("validation crash input")
        rec = _sample_issues()[0]
        score = _keyword_score(q_tokens, rec)
        assert score > 0

    def test_no_match(self):
        q_tokens = _tokenize("kubernetes deployment")
        rec = _sample_issues()[0]
        score = _keyword_score(q_tokens, rec)
        assert score == 0.0

    def test_empty_question(self):
        score = _keyword_score(set(), _sample_issues()[0])
        assert score == 0.0
