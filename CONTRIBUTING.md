# Contributing to Repo Time Machine

Thanks for your interest in contributing. This project is in active development and welcomes feedback, bug reports, and pull requests.

## Getting Started

```bash
git clone https://github.com/rajfirke/repo-time-machine
cd repo-time-machine
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

All tests use mock embedders and run locally — no model downloads or network calls needed.

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check src/ tests/
ruff format src/ tests/
```

Configuration is in `pyproject.toml`:
- Line length: 100
- Target: Python 3.11
- Rules: E, F, I, UP

## Pull Request Guidelines

1. Create a feature branch from `main`.
2. Keep changes minimal and focused on one thing.
3. Add or update tests for any new functionality.
4. Run `pytest` and `ruff check` before submitting.
5. Write a clear PR description explaining what and why.

## Project Layout

- `src/repo_time_machine/ingestion/` — file and git history parsing
- `src/repo_time_machine/retrieval/` — FAISS indexing and search
- `src/repo_time_machine/agent/` — question routing, LLM integration, answer building
- `src/repo_time_machine/cli/` — Typer CLI commands
- `tests/` — pytest test suite

## Reporting Issues

Open an issue on GitHub with:
- What you tried
- What you expected
- What actually happened
- Python version and OS

## Ideas for Contribution

- Support for additional embedding models
- Richer diff analysis in history retrieval
- Web UI or TUI interface
- Performance benchmarks on large repos
- Documentation improvements
