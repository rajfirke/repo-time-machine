"""Tests for the retrieval layer — uses synthetic embeddings to stay fast."""

from unittest.mock import MagicMock

import numpy as np

from repo_time_machine.ingestion.code_loader import FileChunk
from repo_time_machine.ingestion.git_history import CommitRecord
from repo_time_machine.retrieval.code_retriever import CodeResult, CodeRetriever
from repo_time_machine.retrieval.history_retriever import (
    HistoryResult,
    HistoryRetriever,
    _keyword_score,
    _tokenize,
)
from repo_time_machine.retrieval.store import (
    is_indexed,
    load_config,
    save_config,
)

DIM = 384


def _fake_embedder() -> MagicMock:
    """Return a mock Embedder that produces deterministic normalized vectors."""
    embedder = MagicMock()
    embedder.dim = DIM

    def _embed(texts, batch_size=64):
        rng = np.random.RandomState(42)
        vecs = rng.randn(len(texts), DIM).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    embedder.embed = _embed
    return embedder


def _sample_chunks() -> list[FileChunk]:
    return [
        FileChunk(
            file="src/main.py",
            start_line=1,
            end_line=20,
            content="def hello():\n    print('hello world')\n",
            language="python",
        ),
        FileChunk(
            file="src/utils.py",
            start_line=1,
            end_line=15,
            content="def validate(x):\n    if x < 0:\n        raise ValueError\n",
            language="python",
        ),
        FileChunk(
            file="README.md",
            start_line=1,
            end_line=10,
            content="# My Project\nA sample project.\n",
            language="markdown",
        ),
    ]


def _sample_commits() -> list[CommitRecord]:
    return [
        CommitRecord(
            sha="aaa111",
            author="Alice",
            date="2025-01-15 10:00:00",
            message="Add input validation to utils",
            files_changed=["src/utils.py"],
            diff_summary="M src/utils.py",
        ),
        CommitRecord(
            sha="bbb222",
            author="Bob",
            date="2025-01-10 09:00:00",
            message="Initial commit with hello world",
            files_changed=["src/main.py", "README.md"],
            diff_summary="A src/main.py\nA README.md",
        ),
        CommitRecord(
            sha="ccc333",
            author="Alice",
            date="2025-01-20 14:00:00",
            message="Fix bug in validation edge case",
            files_changed=["src/utils.py"],
            diff_summary="M src/utils.py",
        ),
    ]


# ---------------------------------------------------------------------------
# CodeRetriever
# ---------------------------------------------------------------------------


class TestCodeRetriever:
    def test_build_and_query(self, tmp_path):
        embedder = _fake_embedder()
        ret = CodeRetriever(tmp_path / ".rtm", embedder)
        count = ret.build(_sample_chunks())
        assert count == 3

        results = ret.query("hello world", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, CodeResult) for r in results)
        assert results[0].score >= results[1].score

    def test_persist_and_reload(self, tmp_path):
        embedder = _fake_embedder()
        rtm = tmp_path / ".rtm"
        ret = CodeRetriever(rtm, embedder)
        ret.build(_sample_chunks())

        ret2 = CodeRetriever(rtm, embedder)
        assert ret2.load() is True

        results = ret2.query("validation", top_k=1)
        assert len(results) == 1
        assert results[0].file in ("src/main.py", "src/utils.py", "README.md")

    def test_empty_index_returns_empty(self, tmp_path):
        embedder = _fake_embedder()
        ret = CodeRetriever(tmp_path / ".rtm", embedder)
        ret.build([])
        assert ret.query("anything") == []

    def test_top_k_capped(self, tmp_path):
        embedder = _fake_embedder()
        ret = CodeRetriever(tmp_path / ".rtm", embedder)
        ret.build(_sample_chunks())
        results = ret.query("test", top_k=100)
        assert len(results) == 3

    def test_load_missing_returns_false(self, tmp_path):
        embedder = _fake_embedder()
        ret = CodeRetriever(tmp_path / ".rtm", embedder)
        assert ret.load() is False

    def test_result_header(self):
        r = CodeResult(
            file="a.py", start_line=1, end_line=10, content="x", language="python", score=0.9
        )
        assert "a.py:1-10" in r.header()


# ---------------------------------------------------------------------------
# HistoryRetriever
# ---------------------------------------------------------------------------


class TestHistoryRetriever:
    def test_build_and_query(self, tmp_path):
        embedder = _fake_embedder()
        ret = HistoryRetriever(tmp_path / ".rtm", embedder)
        count = ret.build(_sample_commits())
        assert count == 3

        results = ret.query("validation", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, HistoryResult) for r in results)
        assert results[0].score >= results[1].score

    def test_persist_and_reload(self, tmp_path):
        embedder = _fake_embedder()
        rtm = tmp_path / ".rtm"
        ret = HistoryRetriever(rtm, embedder)
        ret.build(_sample_commits())

        ret2 = HistoryRetriever(rtm, embedder)
        assert ret2.load() is True
        results = ret2.query("bug fix", top_k=1)
        assert len(results) == 1

    def test_timeline_for_file(self, tmp_path):
        embedder = _fake_embedder()
        ret = HistoryRetriever(tmp_path / ".rtm", embedder)
        ret.build(_sample_commits())
        timeline = ret.timeline_for_file("src/utils.py")
        assert len(timeline) == 2
        assert all(c.sha in ("aaa111", "ccc333") for c in timeline)

    def test_empty_commits(self, tmp_path):
        embedder = _fake_embedder()
        ret = HistoryRetriever(tmp_path / ".rtm", embedder)
        ret.build([])
        assert ret.query("anything") == []

    def test_keyword_boost(self):
        """Commits with matching keywords should score higher on the kw component."""
        q_tokens = _tokenize("validation utils")
        commit_match = _sample_commits()[0]  # "Add input validation to utils"
        commit_no = _sample_commits()[1]  # "Initial commit with hello world"
        assert _keyword_score(q_tokens, commit_match) > _keyword_score(q_tokens, commit_no)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Why was validation added?")
        assert "validation" in tokens
        assert "added" in tokens

    def test_short_tokens_dropped(self):
        tokens = _tokenize("a b cd ef")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "cd" in tokens


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStore:
    def test_ensure_and_check(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        assert not is_indexed(repo)

        save_config(repo, {"model": "test", "chunks": 5})
        assert is_indexed(repo)

        cfg = load_config(repo)
        assert cfg["model"] == "test"
        assert cfg["chunks"] == 5

    def test_load_missing(self, tmp_path):
        assert load_config(tmp_path / "nope") is None
