"""CLI commands: index and ask."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from repo_time_machine.ingestion.code_loader import ingest_repo
from repo_time_machine.ingestion.git_history import extract_history
from repo_time_machine.retrieval.code_retriever import CodeRetriever
from repo_time_machine.retrieval.embeddings import get_embedder
from repo_time_machine.retrieval.history_retriever import HistoryRetriever
from repo_time_machine.retrieval.store import ensure_rtm_dir, is_indexed, save_config

console = Console()
app = typer.Typer(
    name="rtm",
    help="Repo Time Machine — ask questions backed by code, commits, and issues.",
    no_args_is_help=True,
)


@app.command()
def index(
    repo_path: str = typer.Argument(..., help="Path to the local git repository to index."),
    max_commits: int = typer.Option(500, "--max-commits", "-c", help="Max commits to ingest."),
    chunk_lines: int = typer.Option(60, "--chunk-lines", help="Lines per code chunk."),
    overlap: int = typer.Option(10, "--overlap", help="Overlap lines between chunks."),
    model: str = typer.Option(
        "BAAI/bge-small-en-v1.5", "--model", "-m",
        help="Sentence-transformers model for embeddings.",
    ),
    github_slug: str = typer.Option(
        None, "--github", "-g",
        help="Optional GitHub repo slug (owner/repo) to enrich with issues and PRs.",
    ),
):
    """Ingest a repository: index source files and extract git history."""
    repo = Path(repo_path).resolve()
    if not (repo / ".git").is_dir():
        console.print(f"[red]Error:[/red] {repo} is not a git repository.")
        raise typer.Exit(1)

    t0 = time.time()
    console.print(f"\n[bold]Indexing:[/bold] {repo}\n")

    rtm_path = ensure_rtm_dir(repo)
    embedder = get_embedder(model)

    console.print("[dim]1/3[/dim] Ingesting source files...")
    chunks = ingest_repo(repo, chunk_lines=chunk_lines, overlap_lines=overlap)
    console.print(f"      {len(chunks)} code chunks extracted")

    console.print("[dim]2/3[/dim] Extracting git history...")
    commits = extract_history(repo, max_commits=max_commits)
    console.print(f"      {len(commits)} commits extracted")

    console.print("[dim]3/3[/dim] Building embeddings and FAISS indexes...")
    code_ret = CodeRetriever(rtm_path, embedder)
    code_count = code_ret.build(chunks)

    hist_ret = HistoryRetriever(rtm_path, embedder)
    hist_count = hist_ret.build(commits)

    elapsed = time.time() - t0
    save_config(repo, {
        "model": model,
        "chunk_lines": chunk_lines,
        "overlap": overlap,
        "max_commits": max_commits,
        "code_chunks": code_count,
        "commits_indexed": hist_count,
        "github_slug": github_slug,
    })

    console.print(f"\n[green]Done in {elapsed:.1f}s[/green]")

    table = Table(title="Index Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Code chunks", str(code_count))
    table.add_row("Commits", str(hist_count))
    table.add_row("Index location", str(rtm_path))
    table.add_row("Embedding model", model)
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask about the repository."),
    repo_path: str = typer.Option(
        ".", "--repo", "-r",
        help="Path to the repository (must have been indexed first).",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of evidence pieces to retrieve."),
    llm_model: str = typer.Option(
        "qwen2.5-coder:7b", "--llm", "-l",
        help="Ollama model for answer synthesis.",
    ),
    raw: bool = typer.Option(
        False, "--raw",
        help="Show raw evidence without LLM synthesis.",
    ),
):
    """Ask a question about the indexed repository."""
    repo = Path(repo_path).resolve()
    if not is_indexed(repo):
        console.print(
            f"[red]Error:[/red] {repo} has not been indexed yet. "
            "Run [bold]rtm index[/bold] first."
        )
        raise typer.Exit(1)

    from repo_time_machine.agent.pipeline import Pipeline

    console.print(f"\n[bold]Question:[/bold] {question}\n")

    pipeline = Pipeline(repo_path=repo, llm_model=llm_model, top_k=top_k)
    if not pipeline.ready:
        console.print("[red]Error:[/red] Could not load indexes. Re-run [bold]rtm index[/bold].")
        raise typer.Exit(1)

    with console.status("[dim]Thinking...[/dim]"):
        answer = pipeline.ask(question)

    if answer.used_llm:
        console.print("[green]LLM-synthesized answer[/green]\n")
    else:
        console.print("[yellow]Evidence-only answer (Ollama not running)[/yellow]\n")

    md = Markdown(answer.render())
    console.print(md)


if __name__ == "__main__":
    app()
