"""CLI commands: index and ask."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import typer
from rich.console import Console
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
):
    """Ask a question about the indexed repository."""
    repo = Path(repo_path).resolve()
    if not is_indexed(repo):
        console.print(
            f"[red]Error:[/red] {repo} has not been indexed yet. "
            "Run [bold]rtm index[/bold] first."
        )
        raise typer.Exit(1)

    from repo_time_machine.retrieval.store import load_config

    cfg = load_config(repo)
    model = cfg.get("model", "BAAI/bge-small-en-v1.5") if cfg else "BAAI/bge-small-en-v1.5"
    rtm_path = ensure_rtm_dir(repo)
    embedder = get_embedder(model)

    console.print(f"\n[bold]Question:[/bold] {question}\n")

    code_ret = CodeRetriever(rtm_path, embedder)
    hist_ret = HistoryRetriever(rtm_path, embedder)

    if not code_ret.load() or not hist_ret.load():
        console.print("[red]Error:[/red] Could not load indexes. Re-run [bold]rtm index[/bold].")
        raise typer.Exit(1)

    console.print("[dim]Searching code...[/dim]")
    code_results = code_ret.query(question, top_k=top_k)

    console.print("[dim]Searching history...[/dim]")
    hist_results = hist_ret.query(question, top_k=top_k)

    if code_results:
        console.print("\n[bold]Code Evidence[/bold]\n")
        for i, cr in enumerate(code_results, 1):
            console.print(f"  [cyan]{i}.[/cyan] {cr.header()}  (score: {cr.score:.3f})")
            snippet = cr.content.strip().split("\n")
            for line in snippet[:5]:
                console.print(f"      {line}")
            if len(snippet) > 5:
                console.print(f"      [dim]... +{len(snippet) - 5} more lines[/dim]")
            console.print()

    if hist_results:
        console.print("[bold]History Evidence[/bold]\n")
        for i, hr in enumerate(hist_results, 1):
            c = hr.commit
            console.print(
                f"  [cyan]{i}.[/cyan] [{c.short_sha}] {c.date} — "
                f"{c.message.split(chr(10), 1)[0]}"
            )
            console.print(f"      relevance: {hr.relevance}  (score: {hr.score:.3f})")
            if c.files_changed:
                console.print(f"      files: {', '.join(c.files_changed[:5])}")
            console.print()

    if not code_results and not hist_results:
        console.print("[yellow]No relevant evidence found.[/yellow]")

    console.print("[dim]Tip: Day 3 will add the LLM answer synthesis layer.[/dim]")


if __name__ == "__main__":
    app()
