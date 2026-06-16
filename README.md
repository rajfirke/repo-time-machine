# Repo Time Machine

[![CI](https://github.com/rajfirke/repo-time-machine/actions/workflows/ci.yml/badge.svg)](https://github.com/rajfirke/repo-time-machine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

> Ask questions about your codebase — and get answers backed by code, commits, PRs, and issues together.

Most "chat with your code" tools read only the current snapshot. **Repo Time Machine** treats the entire git history as searchable memory, so you can ask:

- *Why does this validation exist?*
- *Which commit introduced this bug?*
- *What issue or PR explains this design?*
- *How should I safely refactor this function?*

---

## How It Works

```
Your question
    │
    ▼
┌──────────────────────┐
│   Question Router    │  classify → code / history / issue / mixed
└────────┬─────────────┘
         │
    ┌────┴────────────────────────────────┐
    │                │                    │
    ▼                ▼                    ▼
Code Retriever  History Retriever  Issue/PR Retriever
(FAISS semantic (commits, diffs,   (GitHub issues &
 search over     file timelines,    PRs via PyGitHub,
 source chunks)  keyword+semantic)  semantic+keyword)
    │                │                    │
    └────────────────┴────────────────────┘
                     │
                     ▼
          ┌────────────────────┐
          │  Local LLM (Ollama)│
          │  or fallback mode  │
          └────────────────────┘
                     │
                     ▼
     Cited Answer + Evidence + Timeline + Suggested Action
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/rajfirke/repo-time-machine
cd repo-time-machine

# Install
pip install -e .

# Index any git repo
rtm index /path/to/your/repo

# Ask a question
rtm ask "Why was the retry logic added?" --repo /path/to/your/repo
```

### With GitHub enrichment

```bash
# Set a GitHub token (free, read-only public scope)
export GITHUB_TOKEN=ghp_your_token_here

# Index with issues and PRs
rtm index /path/to/repo --github owner/repo

# Now answers can cite issues and PRs
rtm ask "Which issue explains this design decision?" --repo /path/to/repo
```

### With Ollama (for LLM-synthesized answers)

```bash
# Install and start Ollama
ollama pull qwen2.5-coder:7b
ollama serve

# Answers are now synthesized by the LLM with citations
rtm ask "How should I safely refactor this function?" --repo /path/to/repo
```

Without Ollama, `rtm` still works — it returns structured evidence (code snippets, commits, issues) without LLM synthesis.

---

## Output Format

Every answer includes:

| Section | What it contains |
|---|---|
| **Summary** | Direct answer to your question |
| **Evidence** | Code snippets, commit SHAs, issue/PR links |
| **Timeline** | When the relevant changes happened |
| **Suggested action** | What to do next, safely |

---

## Stack

| Layer | Tool | Cost |
|---|---|---|
| Language | Python 3.11+ | Free |
| CLI | Typer + Rich | Free |
| Embeddings | `bge-small-en-v1.5` via sentence-transformers | Free, local |
| Vector store | FAISS (CPU) | Free, local |
| LLM | Ollama + any open model | Free, local |
| Git parsing | GitPython | Free |
| GitHub enrichment | PyGitHub | Free (API rate limits apply) |

Everything runs locally. No paid API required.

---

## Project Structure

```
repo-time-machine/
├── src/repo_time_machine/
│   ├── ingestion/
│   │   ├── code_loader.py     # file walker + sliding-window chunker
│   │   └── git_history.py     # commit/diff extraction via GitPython
│   ├── retrieval/
│   │   ├── embeddings.py      # shared sentence-transformers wrapper
│   │   ├── code_retriever.py  # FAISS semantic search over code chunks
│   │   ├── history_retriever.py # semantic + keyword scoring over commits
│   │   ├── issue_retriever.py # GitHub issues/PRs via PyGitHub + FAISS
│   │   └── store.py           # .rtm/ directory and config management
│   ├── agent/
│   │   ├── router.py          # question classifier (code/history/issue/mixed)
│   │   ├── answer.py          # evidence assembly + LLM prompt + Answer schema
│   │   ├── llm.py             # Ollama HTTP client with graceful fallback
│   │   └── pipeline.py        # end-to-end orchestrator
│   └── cli/
│       └── main.py            # rtm index / rtm ask commands
├── tests/
│   ├── test_ingestion.py      # 20 tests (dogfood against this repo)
│   ├── test_retrieval.py      # 15 tests (mock embedder, FAISS persistence)
│   ├── test_issues.py         # 14 tests (issue retriever, no network)
│   └── test_agent.py          # 21 tests (router, answer builder, LLM mock)
├── plan.md                    # original design document
├── pyproject.toml             # packaging, deps, rtm entry point
└── CONTRIBUTING.md            # how to contribute
```

---

## CLI Reference

### `rtm index`

```
rtm index <repo_path> [OPTIONS]

Options:
  --github, -g TEXT       GitHub slug (owner/repo) for issue enrichment
  --max-commits, -c INT   Max commits to extract [default: 500]
  --chunk-lines INT       Lines per code chunk [default: 60]
  --overlap INT           Overlap between chunks [default: 10]
  --max-issues INT        Max issues/PRs to fetch [default: 200]
  --model, -m TEXT        Embedding model [default: BAAI/bge-small-en-v1.5]
```

### `rtm ask`

```
rtm ask <question> [OPTIONS]

Options:
  --repo, -r PATH         Path to indexed repo [default: .]
  --top-k, -k INT         Evidence pieces to retrieve [default: 5]
  --llm, -l TEXT          Ollama model [default: qwen2.5-coder:7b]
  --raw                   Show raw evidence without LLM
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

70 tests, all passing. Tests use mock embedders and run against this repo itself — no network calls, no model downloads needed.

---

## Requirements

- Python 3.11+
- Git
- Optional: [Ollama](https://ollama.ai) for LLM synthesis
- Optional: `GITHUB_TOKEN` env var for issue enrichment

---

## Why Not Just Use an IDE Plugin?

IDE plugins read open files. They don't know:

- why a line was written
- what alternatives were tried and rejected
- which PR introduced a pattern
- how a file evolved over years

Repo Time Machine answers those questions with evidence.

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## License

[MIT](./LICENSE)
