"""Tests for the ingestion pipeline — runs against the repo-time-machine repo itself."""

from pathlib import Path

import pytest

from repo_time_machine.ingestion.code_loader import (
    FileChunk,
    chunk_file,
    ingest_repo,
    iter_source_files,
    load_file,
)
from repo_time_machine.ingestion.git_history import (
    CommitRecord,
    extract_history,
    file_timeline,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# code_loader tests
# ---------------------------------------------------------------------------


class TestIterSourceFiles:
    def test_finds_python_files(self):
        files = list(iter_source_files(REPO_ROOT))
        py_files = [f for f in files if f.suffix == ".py"]
        assert len(py_files) >= 3, "Should find at least the stub modules"

    def test_skips_git_dir(self):
        files = list(iter_source_files(REPO_ROOT))
        for f in files:
            assert ".git" not in f.parts

    def test_finds_toml_and_md(self):
        files = list(iter_source_files(REPO_ROOT))
        extensions = {f.suffix for f in files}
        assert ".toml" in extensions
        assert ".md" in extensions


class TestLoadFile:
    def test_reads_existing_file(self):
        readme = REPO_ROOT / "README.md"
        content = load_file(readme)
        assert "Repo Time Machine" in content

    def test_returns_empty_for_missing_file(self):
        content = load_file(Path("/nonexistent/file.py"))
        assert content == ""


class TestChunkFile:
    def test_small_file_single_chunk(self):
        init = REPO_ROOT / "src" / "repo_time_machine" / "__init__.py"
        chunks = chunk_file(init, REPO_ROOT, chunk_lines=60)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].language == "python"

    def test_large_file_multiple_chunks(self):
        readme = REPO_ROOT / "README.md"
        chunks = chunk_file(readme, REPO_ROOT, chunk_lines=10, overlap_lines=3)
        assert len(chunks) > 1, "README should produce multiple small chunks"
        assert chunks[0].start_line == 1
        assert chunks[1].start_line < chunks[0].end_line, "Chunks should overlap"

    def test_chunk_has_correct_relative_path(self):
        init = REPO_ROOT / "src" / "repo_time_machine" / "__init__.py"
        chunks = chunk_file(init, REPO_ROOT)
        assert chunks[0].file.startswith("src/")

    def test_empty_file_no_chunks(self, tmp_path):
        empty = tmp_path / "empty.py"
        empty.write_text("")
        chunks = chunk_file(empty, tmp_path)
        assert chunks == []


class TestIngestRepo:
    def test_returns_file_chunks(self):
        chunks = ingest_repo(REPO_ROOT)
        assert len(chunks) > 0
        assert all(isinstance(c, FileChunk) for c in chunks)

    def test_chunks_have_content(self):
        chunks = ingest_repo(REPO_ROOT)
        for chunk in chunks[:5]:
            assert chunk.content.strip(), f"Empty chunk at {chunk.file}"


# ---------------------------------------------------------------------------
# git_history tests
# ---------------------------------------------------------------------------


class TestExtractHistory:
    def test_returns_commit_records(self):
        records = extract_history(REPO_ROOT)
        assert len(records) >= 2, "Repo should have at least the plan + scaffold commits"
        assert all(isinstance(r, CommitRecord) for r in records)

    def test_commit_has_required_fields(self):
        records = extract_history(REPO_ROOT)
        first = records[0]
        assert first.sha
        assert first.author
        assert first.date
        assert first.message

    def test_newest_first(self):
        records = extract_history(REPO_ROOT)
        if len(records) >= 2:
            assert records[0].date >= records[1].date

    def test_files_changed_populated(self):
        records = extract_history(REPO_ROOT)
        has_files = [r for r in records if r.files_changed]
        assert len(has_files) > 0, "At least one commit should list changed files"

    def test_invalid_path_returns_empty(self, tmp_path):
        empty = tmp_path / "not_a_repo"
        empty.mkdir()
        records = extract_history(empty)
        assert records == []

    def test_max_commits_respected(self):
        records = extract_history(REPO_ROOT, max_commits=1)
        assert len(records) == 1

    def test_short_sha_and_oneline(self):
        records = extract_history(REPO_ROOT)
        first = records[0]
        assert len(first.short_sha) == 8
        assert first.short_sha in first.oneline()


class TestFileTimeline:
    def test_finds_commits_for_plan(self):
        records = file_timeline(REPO_ROOT, "plan.md")
        assert len(records) >= 1
        assert any("plan" in r.message.lower() for r in records)

    def test_nonexistent_file_returns_empty(self):
        records = file_timeline(REPO_ROOT, "does_not_exist.xyz")
        assert records == []
