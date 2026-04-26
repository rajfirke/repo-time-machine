# Repo Time Machine

> Ask questions about your codebase — and get answers backed by code, commits, PRs, and issues together.

Most "chat with your code" tools read only the current snapshot. **Repo Time Machine** treats the entire git history as searchable memory, so you can ask:

- *Why does this validation exist?*
- *Which commit introduced this bug?*
- *What changed in this file over the last 3 months?*
- *How should I safely refactor this function?*

---

## How It Works

```
Your question
    │
    ▼
┌─────────────────────┐
│   Question Router   │  classify: code / history / issue-PR
└────────┬────────────┘
         │
    ┌────┴──────────────────────────────┐
    │                                   │
    ▼                                   ▼
Code Retriever                  History Retriever
(semantic search over           (commits, diffs,
 current source files)           file timelines)
    │                                   │
    └────────────┬──────────────────────┘
                 │
                 ▼  (optional)
         Issue/PR Retriever
         (GitHub metadata)
                 │
                 ▼
    ┌────────────────────────┐
    │  Local LLM (Ollama)    │
    │  Qwen2.5-Coder or similar│
    └────────────────────────┘
                 │
                 ▼
    Cited Answer  +  Evidence  +  Timeline  +  Suggested Action
```

---

## Stack

| Layer | Tool |
|---|---|
| CLI | `Typer` |
| Embeddings | `bge-small-en-v1.5` via `sentence-transformers` |
| Vector store | `FAISS` (local, no server needed) |
| Local LLM | `Ollama` + `Qwen2.5-Coder 7B` |
| Git parsing | `GitPython` |
| GitHub enrichment | `PyGitHub` (optional, free tier) |
| Optional API | `Groq` or `OpenRouter` (free tiers) |

Everything runs locally. No paid API required.

---

## Quickstart

> Coming in v0.1

```bash
git clone https://github.com/rajfirke/repo-time-machine
cd repo-time-machine
pip install -e .

# Index a repo
rtm index /path/to/your/repo

# Ask a question
rtm ask "Why was the retry logic added to api_client.py?"
```

---

## Output Format

Every answer comes with:

- **Summary** — the direct answer
- **Evidence** — code snippets, commit SHAs, issue links
- **Timeline** — when the relevant changes happened
- **Suggested action** — what to do next, safely

---

## Project Status

| Milestone | Status |
|---|---|
| `plan.md` — initial design | Done |
| Project scaffold | Done |
| Repo ingestion pipeline | Planned |
| Commit + diff retrieval | Planned |
| Code semantic search | Planned |
| Question router | Planned |
| CLI (`rtm`) | Planned |
| GitHub issue/PR enrichment | Planned |
| Demo on a real OSS repo | Planned |

---

## Why Not Just Use an IDE Plugin?

IDE plugins read open files. They don't know:

- why a line exists
- what alternatives were rejected
- which PR introduced a pattern
- how this file changed over 2 years

Repo Time Machine answers those questions.

---

## Contributing

This project is in early design. The plan lives in [`plan.md`](./plan.md). Feedback, ideas, and PRs welcome.

---

## License

MIT
