"""Tests for the retrieval layer — uses synthetic embeddings to stay fast."""

from unittest.mock import MagicMock

import numpy as np

from repo_time_machine.ingestion.code_loader import FileChunk
from repo_time_machine.ingestion.git_history import CommitRecord
from repo_time_machine.retrieval.code_retriever import CodeResult, CodeRetriever
from repo_time_machine.retrieval.embeddings import _FALLBACK_DIM, Embedder
from repo_time_machine.retrieval.history_retriever import (
    HistoryResult,
    HistoryRetriever,
    _keyword_score,
    _tokenize,
)
from repo_time_machine.retrieval.store import (
    clean_rtm_dir,
    index_health,
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
        # config exists but FAISS files are missing — should NOT be considered indexed
        assert not is_indexed(repo)

    def test_is_indexed_with_all_files(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        save_config(repo, {"model": "test"})
        rtm = repo / ".rtm"
        for name in ("code.faiss", "code_meta.json", "commits.faiss", "commits_meta.json"):
            (rtm / name).write_text("x")
        assert is_indexed(repo)

    def test_load_config_corrupt_json(self, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        rtm = repo / ".rtm"
        rtm.mkdir()
        (rtm / "config.json").write_text("{broken json!!!")
        assert load_config(repo) is None

    def test_load_missing(self, tmp_path):
        assert load_config(tmp_path / "nope") is None


# ---------------------------------------------------------------------------
# Index consistency validation
# ---------------------------------------------------------------------------


class TestIndexConsistency:
    def test_code_load_rejects_meta_mismatch(self, tmp_path):
        """When FAISS vector count != metadata length, load() returns False."""
        embedder = _fake_embedder()
        rtm = tmp_path / ".rtm"
        ret = CodeRetriever(rtm, embedder)
        ret.build(_sample_chunks())  # 3 chunks

        import json

        meta_path = rtm / "code_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta.pop()  # remove one entry to create mismatch
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        ret2 = CodeRetriever(rtm, embedder)
        assert ret2.load() is False

    def test_history_load_rejects_meta_mismatch(self, tmp_path):
        embedder = _fake_embedder()
        rtm = tmp_path / ".rtm"
        ret = HistoryRetriever(rtm, embedder)
        ret.build(_sample_commits())

        import json

        meta_path = rtm / "commits_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta.pop()
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        ret2 = HistoryRetriever(rtm, embedder)
        assert ret2.load() is False

    def test_code_load_handles_corrupt_json(self, tmp_path):
        embedder = _fake_embedder()
        rtm = tmp_path / ".rtm"
        ret = CodeRetriever(rtm, embedder)
        ret.build(_sample_chunks())

        (rtm / "code_meta.json").write_text("{corrupt!!!")
        ret2 = CodeRetriever(rtm, embedder)
        assert ret2.load() is False

    def test_history_load_handles_corrupt_json(self, tmp_path):
        embedder = _fake_embedder()
        rtm = tmp_path / ".rtm"
        ret = HistoryRetriever(rtm, embedder)
        ret.build(_sample_commits())

        (rtm / "commits_meta.json").write_text("{corrupt!!!")
        ret2 = HistoryRetriever(rtm, embedder)
        assert ret2.load() is False

    def test_code_load_handles_corrupt_faiss(self, tmp_path):
        embedder = _fake_embedder()
        rtm = tmp_path / ".rtm"
        ret = CodeRetriever(rtm, embedder)
        ret.build(_sample_chunks())

        (rtm / "code.faiss").write_text("not a faiss file")
        ret2 = CodeRetriever(rtm, embedder)
        assert ret2.load() is False

    def test_valid_load_still_works(self, tmp_path):
        embedder = _fake_embedder()
        rtm = tmp_path / ".rtm"
        ret = CodeRetriever(rtm, embedder)
        ret.build(_sample_chunks())

        ret2 = CodeRetriever(rtm, embedder)
        assert ret2.load() is True
        results = ret2.query("hello", top_k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# index_health
# ---------------------------------------------------------------------------


class TestIndexHealth:
    def _setup_indexed_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        save_config(repo, {"model": "test", "code_chunks": 10, "commits_indexed": 5})
        rtm = repo / ".rtm"
        for name in ("code.faiss", "code_meta.json", "commits.faiss", "commits_meta.json"):
            (rtm / name).write_bytes(b"x" * 100)
        return repo

    def test_healthy_repo(self, tmp_path):
        repo = self._setup_indexed_repo(tmp_path)
        h = index_health(repo)
        assert h.indexed
        assert h.healthy
        assert h.config["model"] == "test"
        assert len(h.files) == 6  # 4 required + 2 optional
        assert all(f.present for f in h.files if f.required)

    def test_unindexed_repo(self, tmp_path):
        repo = tmp_path / "empty"
        repo.mkdir()
        h = index_health(repo)
        assert not h.indexed
        assert not h.healthy

    def test_missing_faiss_is_unhealthy(self, tmp_path):
        repo = self._setup_indexed_repo(tmp_path)
        (repo / ".rtm" / "code.faiss").unlink()
        h = index_health(repo)
        assert h.indexed  # config exists
        assert not h.healthy  # required file missing
        missing = [f for f in h.files if not f.present and f.required]
        assert len(missing) == 1
        assert missing[0].name == "code.faiss"

    def test_optional_files_dont_affect_health(self, tmp_path):
        repo = self._setup_indexed_repo(tmp_path)
        h = index_health(repo)
        optional = [f for f in h.files if not f.required]
        assert len(optional) == 2
        assert not any(f.present for f in optional)
        assert h.healthy

    def test_to_dict_has_expected_keys(self, tmp_path):
        repo = self._setup_indexed_repo(tmp_path)
        d = index_health(repo).to_dict()
        assert "indexed" in d
        assert "healthy" in d
        assert "config" in d
        assert "files" in d
        assert isinstance(d["files"], list)

    def test_file_sizes_populated(self, tmp_path):
        repo = self._setup_indexed_repo(tmp_path)
        h = index_health(repo)
        for f in h.files:
            if f.present:
                assert f.size_bytes == 100


# ---------------------------------------------------------------------------
# Embedder dimension auto-detection
# ---------------------------------------------------------------------------


class TestEmbedderDim:
    def test_fallback_dim_before_load(self):
        embedder = Embedder("BAAI/bge-small-en-v1.5")
        assert embedder.dim == _FALLBACK_DIM

    def test_detected_dim_overrides_fallback(self):
        embedder = Embedder("BAAI/bge-small-en-v1.5")
        embedder._detected_dim = 768
        assert embedder.dim == 768

    def test_embed_empty_uses_current_dim(self):
        embedder = Embedder("BAAI/bge-small-en-v1.5")
        embedder._detected_dim = 1024
        result = embedder.embed([])
        assert result.shape == (0, 1024)


# ---------------------------------------------------------------------------
# CLI status command
# ---------------------------------------------------------------------------


class TestStatusCommand:
    def _setup_indexed_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        save_config(
            repo,
            {
                "model": "BAAI/bge-small-en-v1.5",
                "embed_dim": 384,
                "chunk_lines": 60,
                "overlap": 10,
                "code_chunks": 42,
                "commits_indexed": 15,
                "github_slug": "owner/repo",
                "issues_indexed": 8,
            },
        )
        rtm = repo / ".rtm"
        for name in ("code.faiss", "code_meta.json", "commits.faiss", "commits_meta.json"):
            (rtm / name).write_bytes(b"x" * 100)
        return repo

    def test_status_json_healthy(self, tmp_path):
        from typer.testing import CliRunner

        from repo_time_machine.cli.main import app

        repo = self._setup_indexed_repo(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["status", "--repo", str(repo), "--json"])
        assert result.exit_code == 0
        import json

        data = json.loads(result.output)
        assert data["indexed"] is True
        assert data["healthy"] is True
        assert data["config"]["code_chunks"] == 42

    def test_status_json_unindexed(self, tmp_path):
        from typer.testing import CliRunner

        from repo_time_machine.cli.main import app

        repo = tmp_path / "empty"
        repo.mkdir()
        runner = CliRunner()
        result = runner.invoke(app, ["status", "--repo", str(repo), "--json"])
        assert result.exit_code == 1
        import json

        data = json.loads(result.output)
        assert data["indexed"] is False

    def test_status_rich_output_healthy(self, tmp_path):
        from typer.testing import CliRunner

        from repo_time_machine.cli.main import app

        repo = self._setup_indexed_repo(tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["status", "--repo", str(repo)])
        assert result.exit_code == 0
        assert "healthy" in result.output.lower() or "present" in result.output.lower()

    def test_status_rich_output_unindexed(self, tmp_path):
        from typer.testing import CliRunner

        from repo_time_machine.cli.main import app

        repo = tmp_path / "empty"
        repo.mkdir()
        runner = CliRunner()
        result = runner.invoke(app, ["status", "--repo", str(repo)])
        assert result.exit_code == 1
        assert "not been indexed" in result.output


# ---------------------------------------------------------------------------
# clean_rtm_dir
# ---------------------------------------------------------------------------


class TestCleanRtmDir:
    def test_clean_removes_index_files(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        save_config(repo, {"model": "test"})
        rtm = repo / ".rtm"
        for name in ("code.faiss", "code_meta.json"):
            (rtm / name).write_bytes(b"x" * 50)
        removed, freed = clean_rtm_dir(repo)
        assert removed == 3  # config.json + 2 files
        assert freed > 0
        assert not rtm.exists()

    def test_clean_nonexistent_is_noop(self, tmp_path):
        removed, freed = clean_rtm_dir(tmp_path / "nope")
        assert removed == 0
        assert freed == 0


class TestCleanCommand:
    def test_clean_with_force(self, tmp_path):
        from typer.testing import CliRunner

        from repo_time_machine.cli.main import app

        repo = tmp_path / "repo"
        repo.mkdir()
        save_config(repo, {"model": "test"})
        (repo / ".rtm" / "code.faiss").write_bytes(b"x" * 100)

        runner = CliRunner()
        result = runner.invoke(app, ["clean", "--repo", str(repo), "--force"])
        assert result.exit_code == 0
        assert "Cleaned" in result.output

    def test_clean_nothing_to_clean(self, tmp_path):
        from typer.testing import CliRunner

        from repo_time_machine.cli.main import app

        repo = tmp_path / "empty"
        repo.mkdir()
        runner = CliRunner()
        result = runner.invoke(app, ["clean", "--repo", str(repo), "--force"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output
