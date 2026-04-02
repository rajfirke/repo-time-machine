# Repo Time Machine

Historical Agentic RAG for codebases.

## Idea

`Repo Time Machine` helps developers answer questions that normal code RAG misses:

- Why does this code exist?
- What changed before this bug appeared?
- Which past PR or issue explains this logic?
- What is the safest way to modify this file now?

Instead of retrieving only source files, it retrieves across:

- current code
- git commit history
- PR titles and descriptions
- issues and discussions
- file-level change timelines

The agent then produces an evidence-backed answer and a safe change plan.

## Why This Is Interesting

Most repo assistants only read the current snapshot of code. Real engineering context lives in history. `Repo Time Machine` turns repository history into searchable memory.

This makes the project:

- useful for onboarding
- strong for debugging and regression analysis
- different from generic "chat with your code" tools

## Core Differentiators

1. Temporal retrieval, not just semantic retrieval.
2. Answers backed by code, commits, PRs, and issues together.
3. Change-planning output focused on safe edits, not just explanations.
4. Local-first design so the MVP stays cheap to run.

## MVP

Build a local-first CLI that can:

1. Ingest a git repo and chunk source files.
2. Extract commit messages, diffs, and file history.
3. Optionally fetch linked GitHub issues and PR metadata.
4. Route questions to the right retrievers:
   - code retriever
   - history retriever
   - issue/PR retriever
5. Return a cited answer with:
   - summary
   - evidence
   - timeline
   - suggested next action

## Example Queries

- Why was this validation added?
- Which commit likely introduced this behavior?
- What files usually change with this module?
- How should I safely refactor this function?
- What issue or PR explains this design choice?

## System Design

### Inputs

- local repository
- git log and diffs
- optional GitHub metadata via API

### Retrieval Layers

- semantic code search
- commit and diff search
- issue/PR search
- timeline builder for a target file or symbol

### Agent Flow

1. classify the question
2. choose retrieval sources
3. gather evidence
4. detect contradictions or uncertainty
5. generate a concise answer with citations

## Low-Cost Stack

- `Python`
- `Typer` for CLI
- `FAISS` or `LanceDB`
- `sentence-transformers` or `bge-small` embeddings
- `Ollama` with open models for local inference
- optional GitHub API for issue/PR metadata

Recommended cheap-first model path:

- embeddings: `bge-small-en-v1.5` or similar
- local generation: `Qwen2.5-Coder 7B` or similar via `Ollama`
- optional API fallback: `OpenRouter` or `Groq` only when needed

## v0.1 Deliverables

- repo ingestion pipeline
- hybrid retrieval over code + history
- local CLI for question answering
- cited markdown response format
- one demo against a real open source repo

## Non-Goals For v0.1

- IDE plugin
- autonomous code editing
- large-scale multi-repo indexing
- enterprise auth and permissions

## Success Criteria

- answers include verifiable evidence
- history-aware answers are better than plain code RAG
- setup works on a laptop with mostly open-source tooling
- the demo is compelling enough for GitHub and future extension

## First Build Order

1. define the response schema
2. build repo and git-history ingestion
3. add code and commit retrieval
4. add question router
5. add optional GitHub issue/PR enrichment
6. polish demo and docs
