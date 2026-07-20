"""Tests for the ingestion pipeline — runs against the repo-time-machine repo itself."""

from pathlib import Path
from unittest.mock import patch

from git import Repo

from repo_time_machine.ingestion.code_loader import (
    FileChunk,
    chunk_file,
    ingest_repo,
    iter_source_files,
    load_file,
)
from repo_time_machine.ingestion.git_history import (
    CommitRecord,
    _compute_diffs,
    _files_from_diffs,
    _format_diffs,
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

    def test_excludes_gitignored_files(self, tmp_path):
        """Files matched by .gitignore should not be yielded."""
        repo = Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@test.com").release()

        (tmp_path / "app.py").write_text("x = 1\n")
        (tmp_path / "debug.log").write_text("noise\n")
        (tmp_path / ".gitignore").write_text("*.log\n")

        repo.index.add(["app.py", ".gitignore"])
        repo.index.commit("init")

        files = list(iter_source_files(tmp_path))
        names = {f.name for f in files}
        assert "app.py" in names
        assert "debug.log" not in names

    def test_excludes_gitignored_directory(self, tmp_path):
        repo = Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@test.com").release()

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')\n")
        (tmp_path / "coverage").mkdir()
        (tmp_path / "coverage" / "report.json").write_text("{}\n")
        (tmp_path / ".gitignore").write_text("coverage/\n")

        repo.index.add(["src/main.py", ".gitignore"])
        repo.index.commit("init")

        files = list(iter_source_files(tmp_path))
        names = {f.name for f in files}
        assert "main.py" in names
        assert "report.json" not in names


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

    def test_chunk_lines_zero_raises(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        import pytest

        with pytest.raises(ValueError, match="chunk_lines must be >= 1"):
            chunk_file(f, tmp_path, chunk_lines=0)

    def test_chunk_lines_negative_raises(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        import pytest

        with pytest.raises(ValueError, match="chunk_lines must be >= 1"):
            chunk_file(f, tmp_path, chunk_lines=-1)

    def test_overlap_negative_raises(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        import pytest

        with pytest.raises(ValueError, match="overlap_lines must be >= 0"):
            chunk_file(f, tmp_path, chunk_lines=10, overlap_lines=-1)

    def test_overlap_ge_chunk_lines_raises(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        import pytest

        with pytest.raises(ValueError, match="overlap_lines.*must be < chunk_lines"):
            chunk_file(f, tmp_path, chunk_lines=5, overlap_lines=5)

        with pytest.raises(ValueError, match="overlap_lines.*must be < chunk_lines"):
            chunk_file(f, tmp_path, chunk_lines=5, overlap_lines=100)

    def test_ingest_repo_rejects_invalid_params(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="chunk_lines must be >= 1"):
            ingest_repo(tmp_path, chunk_lines=0)
        with pytest.raises(ValueError, match="overlap_lines must be >= 0"):
            ingest_repo(tmp_path, chunk_lines=10, overlap_lines=-5)
        with pytest.raises(ValueError, match="overlap_lines.*must be < chunk_lines"):
            ingest_repo(tmp_path, chunk_lines=5, overlap_lines=10)


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


# ---------------------------------------------------------------------------
# Initial commit diff correctness (issue #6)
# ---------------------------------------------------------------------------


class TestInitialCommitDiff:
    """Verify that initial commits diff against the empty tree, not the working tree."""

    def _make_repo_with_initial_commit(self, tmp_path: Path) -> Repo:
        """Create a repo with one commit containing exactly one file."""
        repo = Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@test.com").release()
        hello = tmp_path / "hello.txt"
        hello.write_text("hello world\n")
        repo.index.add(["hello.txt"])
        repo.index.commit("initial commit")
        return repo

    def test_files_from_initial_commit_only_lists_committed_files(self, tmp_path):
        repo = self._make_repo_with_initial_commit(tmp_path)
        (tmp_path / "untracked.txt").write_text("noise")
        initial = list(repo.iter_commits())[-1]
        diffs = _compute_diffs(initial)
        files = _files_from_diffs(diffs)
        assert files == ["hello.txt"]

    def test_diff_summary_initial_commit_only_shows_committed_files(self, tmp_path):
        repo = self._make_repo_with_initial_commit(tmp_path)
        (tmp_path / "untracked.txt").write_text("noise")
        initial = list(repo.iter_commits())[-1]
        diffs = _compute_diffs(initial)
        summary = _format_diffs(diffs)
        assert "hello.txt" in summary
        assert "untracked.txt" not in summary

    def test_initial_commit_files_not_inflated_by_later_commits(self, tmp_path):
        repo = self._make_repo_with_initial_commit(tmp_path)
        second = tmp_path / "second.txt"
        second.write_text("added later\n")
        repo.index.add(["second.txt"])
        repo.index.commit("add second file")

        initial = list(repo.iter_commits())[-1]
        diffs = _compute_diffs(initial)
        files = _files_from_diffs(diffs)
        assert files == ["hello.txt"], f"Initial commit should only touch hello.txt, got {files}"

    def test_extract_history_initial_commit_correct(self, tmp_path):
        repo = self._make_repo_with_initial_commit(tmp_path)
        (tmp_path / "extra.py").write_text("x = 1\n")
        repo.index.add(["extra.py"])
        repo.index.commit("add extra")

        records = extract_history(tmp_path)
        initial = records[-1]
        assert initial.files_changed == ["hello.txt"]
        assert "hello.txt" in initial.diff_summary
        assert "extra.py" not in initial.diff_summary


# ---------------------------------------------------------------------------
# Empty repository handling (issue #7)
# ---------------------------------------------------------------------------


class TestEmptyRepoHandling:
    """Verify that functions return empty lists instead of crashing on repos with no commits."""

    def test_extract_history_empty_repo_returns_empty(self, tmp_path):
        Repo.init(tmp_path)
        records = extract_history(tmp_path)
        assert records == []

    def test_file_timeline_empty_repo_returns_empty(self, tmp_path):
        Repo.init(tmp_path)
        records = file_timeline(tmp_path, "anything.py")
        assert records == []

    def test_extract_history_empty_repo_does_not_raise(self, tmp_path):
        Repo.init(tmp_path)
        try:
            extract_history(tmp_path)
        except Exception as exc:
            raise AssertionError(f"extract_history raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Deduplicated diff computation (issue #8)
# ---------------------------------------------------------------------------


class TestDeduplicatedDiff:
    """Verify that diff is computed once per commit and helpers produce correct output."""

    def _make_repo(self, tmp_path: Path) -> Repo:
        repo = Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@test.com").release()
        (tmp_path / "a.txt").write_text("hello\n")
        repo.index.add(["a.txt"])
        repo.index.commit("add a")
        (tmp_path / "b.txt").write_text("world\n")
        repo.index.add(["b.txt"])
        repo.index.commit("add b")
        return repo

    def test_format_diffs_produces_action_lines(self, tmp_path):
        repo = self._make_repo(tmp_path)
        commit = list(repo.iter_commits())[0]
        diffs = _compute_diffs(commit)
        summary = _format_diffs(diffs)
        assert "b.txt" in summary

    def test_files_from_diffs_extracts_paths(self, tmp_path):
        repo = self._make_repo(tmp_path)
        commit = list(repo.iter_commits())[0]
        diffs = _compute_diffs(commit)
        files = _files_from_diffs(diffs)
        assert "b.txt" in files

    def test_record_files_and_summary_consistent(self, tmp_path):
        repo = self._make_repo(tmp_path)
        commit = list(repo.iter_commits())[0]
        diffs = _compute_diffs(commit)
        files = _files_from_diffs(diffs)
        summary = _format_diffs(diffs)
        for f in files:
            assert f in summary, f"{f} in files_changed but not in diff_summary"

    def test_diff_computed_once_per_commit(self, tmp_path):
        """Patch _compute_diffs to count calls — should be exactly once per commit."""
        self._make_repo(tmp_path)
        with patch(
            "repo_time_machine.ingestion.git_history._compute_diffs",
            wraps=_compute_diffs,
        ) as mock_diff:
            records = extract_history(tmp_path)
            assert len(records) == 2
            assert mock_diff.call_count == 2, (
                f"Expected 2 diff calls for 2 commits, got {mock_diff.call_count}"
            )

    def test_empty_diffs_produce_empty_outputs(self):
        assert _format_diffs([]) == ""
        assert _files_from_diffs([]) == []


# ---------------------------------------------------------------------------
# Merge commit filtering (issue #52)
# ---------------------------------------------------------------------------


class TestMergeCommitFiltering:
    """Verify that skip_merges excludes merge commits and is_merge is set correctly."""

    def _make_repo_with_merge(self, tmp_path: Path) -> Repo:
        """Create a repo with a merge commit."""
        repo = Repo.init(tmp_path, initial_branch="main")
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@test.com").release()

        (tmp_path / "a.txt").write_text("initial\n")
        repo.index.add(["a.txt"])
        repo.index.commit("initial commit")

        repo.create_head("feature")
        repo.heads.feature.checkout()
        (tmp_path / "b.txt").write_text("feature work\n")
        repo.index.add(["b.txt"])
        repo.index.commit("add feature")

        repo.heads.main.checkout()
        (tmp_path / "c.txt").write_text("main work\n")
        repo.index.add(["c.txt"])
        repo.index.commit("main work")

        repo.index.merge_tree(repo.heads.feature)
        repo.index.commit(
            "Merge branch 'feature' into main",
            parent_commits=(repo.heads.main.commit, repo.heads.feature.commit),
        )
        return repo

    def test_is_merge_flag_set_on_merge_commits(self, tmp_path):
        self._make_repo_with_merge(tmp_path)
        records = extract_history(tmp_path)
        merge_records = [r for r in records if r.is_merge]
        non_merge = [r for r in records if not r.is_merge]
        assert len(merge_records) >= 1
        assert len(non_merge) >= 2

    def test_skip_merges_excludes_merge_commits(self, tmp_path):
        self._make_repo_with_merge(tmp_path)
        all_records = extract_history(tmp_path)
        filtered = extract_history(tmp_path, skip_merges=True)
        assert len(filtered) < len(all_records)
        assert all(not r.is_merge for r in filtered)

    def test_skip_merges_false_includes_everything(self, tmp_path):
        self._make_repo_with_merge(tmp_path)
        all_records = extract_history(tmp_path, skip_merges=False)
        assert any(r.is_merge for r in all_records)

    def test_is_merge_false_for_normal_commits(self, tmp_path):
        repo = Repo.init(tmp_path)
        repo.config_writer().set_value("user", "name", "Test").release()
        repo.config_writer().set_value("user", "email", "test@test.com").release()
        (tmp_path / "x.txt").write_text("hello\n")
        repo.index.add(["x.txt"])
        repo.index.commit("normal commit")
        records = extract_history(tmp_path)
        assert len(records) == 1
        assert not records[0].is_merge
