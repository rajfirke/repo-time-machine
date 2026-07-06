"""CLI commands: index, ask, clean, and status."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
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
from repo_time_machine.retrieval.issue_retriever import IssueRetriever
from repo_time_machine.retrieval.store import (
    clean_rtm_dir,
    ensure_rtm_dir,
    index_health,
    is_indexed,
    rtm_dir,
    save_config,
)

console = Console()


@contextmanager
def _noop_context():
    yield


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
        "BAAI/bge-small-en-v1.5",
        "--model",
        "-m",
        help="Sentence-transformers model for embeddings.",
    ),
    github_slug: str = typer.Option(
        None,
        "--github",
        "-g",
        help="GitHub repo slug (owner/repo) to fetch issues and PRs from.",
    ),
    max_issues: int = typer.Option(200, "--max-issues", help="Max issues/PRs to fetch."),
):
    """Ingest a repository: index source files, git history, and optionally GitHub issues."""
    repo = Path(repo_path).resolve()
    if not (repo / ".git").is_dir():
        console.print(f"[red]Error:[/red] {repo} is not a git repository.")
        raise typer.Exit(1)

    t0 = time.time()
    has_github = bool(github_slug)
    steps = "4" if has_github else "3"
    console.print(f"\n[bold]Indexing:[/bold] {repo}\n")

    rtm_path = ensure_rtm_dir(repo)
    embedder = get_embedder(model)

    console.print(f"[dim]1/{steps}[/dim] Ingesting source files...")
    chunks = ingest_repo(repo, chunk_lines=chunk_lines, overlap_lines=overlap)
    console.print(f"      {len(chunks)} code chunks extracted")

    console.print(f"[dim]2/{steps}[/dim] Extracting git history...")
    commits = extract_history(repo, max_commits=max_commits)
    console.print(f"      {len(commits)} commits extracted")

    console.print(f"[dim]3/{steps}[/dim] Building embeddings and FAISS indexes...")
    code_ret = CodeRetriever(rtm_path, embedder)
    code_count = code_ret.build(chunks)

    hist_ret = HistoryRetriever(rtm_path, embedder)
    hist_count = hist_ret.build(commits)

    issue_count = 0
    if has_github:
        console.print(f"[dim]4/{steps}[/dim] Fetching GitHub issues/PRs from {github_slug}...")
        issue_ret = IssueRetriever(rtm_path, embedder, repo_slug=github_slug)
        issue_count = issue_ret.fetch_and_build(max_items=max_issues)
        if issue_count:
            console.print(f"      {issue_count} issues/PRs indexed")
        else:
            console.print("      [yellow]No issues fetched (check GITHUB_TOKEN)[/yellow]")

    elapsed = time.time() - t0
    save_config(
        repo,
        {
            "model": model,
            "embed_dim": embedder.dim,
            "chunk_lines": chunk_lines,
            "overlap": overlap,
            "max_commits": max_commits,
            "code_chunks": code_count,
            "commits_indexed": hist_count,
            "github_slug": github_slug,
            "issues_indexed": issue_count,
        },
    )

    console.print(f"\n[green]Done in {elapsed:.1f}s[/green]")

    table = Table(title="Index Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Code chunks", str(code_count))
    table.add_row("Commits", str(hist_count))
    if has_github:
        table.add_row("Issues/PRs", str(issue_count))
    table.add_row("Index location", str(rtm_path))
    table.add_row("Embedding model", model)
    console.print(table)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


@app.command()
def status(
    repo_path: str = typer.Option(".", "--repo", "-r", help="Path to repository."),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON."),
):
    """Show index health, configuration, and diagnostics."""
    repo = Path(repo_path).resolve()
    health = index_health(repo)

    if as_json:
        print(json.dumps(health.to_dict(), indent=2))  # noqa: T201
        raise typer.Exit(0 if health.indexed else 1)

    if not health.indexed:
        console.print(
            f"\n[yellow]⚠ Repository has not been indexed yet.[/yellow]\n"
            f"  Run: [bold]rtm index {repo}[/bold]\n"
        )
        raise typer.Exit(1)

    cfg = health.config

    info = Table(title="Repo Time Machine Status", show_lines=True)
    info.add_column("Property", style="bold")
    info.add_column("Value")

    info.add_row("Repository", str(repo))
    info.add_row("Index location", str(health.rtm_path))
    info.add_row("Embedding model", cfg.get("model", "unknown"))
    if cfg.get("embed_dim"):
        info.add_row("Embedding dim", str(cfg["embed_dim"]))
    chunk_l, overlap_l = cfg.get("chunk_lines", "?"), cfg.get("overlap", "?")
    info.add_row("Chunk config", f"{chunk_l} lines, {overlap_l} overlap")

    info.add_row("Code chunks", str(cfg.get("code_chunks", "?")))
    info.add_row("Commits indexed", str(cfg.get("commits_indexed", "?")))
    slug = cfg.get("github_slug", "")
    if slug:
        info.add_row("GitHub slug", slug)
        info.add_row("Issues/PRs indexed", str(cfg.get("issues_indexed", 0)))
    else:
        info.add_row("GitHub enrichment", "[dim]not configured[/dim]")

    console.print()
    console.print(info)

    files_table = Table(title="Index Files", show_lines=True)
    files_table.add_column("File", style="bold")
    files_table.add_column("Status")
    files_table.add_column("Size", justify="right")
    files_table.add_column("Required")

    for fh in health.files:
        mark = "[green]✓ present[/green]" if fh.present else "[red]✗ missing[/red]"
        size = _human_size(fh.size_bytes) if fh.present else "-"
        req = "yes" if fh.required else "optional"
        files_table.add_row(fh.name, mark, size, req)

    console.print(files_table)

    if health.healthy:
        console.print("\n[green]Index is healthy — all required files present.[/green]\n")
    else:
        missing = [f.name for f in health.files if f.required and not f.present]
        console.print(
            f"\n[red]⚠ Index is incomplete:[/red] missing {', '.join(missing)}\n"
            f"  Re-run: [bold]rtm index {repo}[/bold]\n"
        )

    raise typer.Exit(0 if health.healthy else 1)


@app.command()
def clean(
    repo_path: str = typer.Option(".", "--repo", "-r", help="Path to repository."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
):
    """Remove the .rtm/ index directory to free disk space or prepare for re-indexing."""
    repo = Path(repo_path).resolve()
    d = rtm_dir(repo)

    if not d.exists():
        console.print(f"[dim]Nothing to clean — {d} does not exist.[/dim]")
        raise typer.Exit(0)

    total_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    file_count = sum(1 for f in d.rglob("*") if f.is_file())
    console.print(
        f"\n[bold]Will delete:[/bold] {d}\n"
        f"  Files: {file_count}\n"
        f"  Size:  {_human_size(total_size)}\n"
    )

    if not force:
        confirm = typer.confirm("Proceed?")
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    removed, freed = clean_rtm_dir(repo)
    console.print(
        f"[green]Cleaned:[/green] {removed} file(s) removed, {_human_size(freed)} freed.\n"
        f"  Re-run [bold]rtm index {repo}[/bold] to rebuild.\n"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask about the repository."),
    repo_path: str = typer.Option(
        ".",
        "--repo",
        "-r",
        help="Path to the repository (must have been indexed first).",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of evidence pieces to retrieve."),
    llm_model: str = typer.Option(
        "qwen2.5-coder:7b",
        "--llm",
        "-l",
        help="Ollama model for answer synthesis.",
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Show raw evidence without LLM synthesis.",
    ),
    output: str = typer.Option(
        "rich",
        "--output",
        "-o",
        help="Output format: 'rich' (default) or 'json'.",
    ),
):
    """Ask a question about the indexed repository."""
    repo = Path(repo_path).resolve()
    if not is_indexed(repo):
        if output == "json":
            print(json.dumps({"error": "Repository has not been indexed yet."}))  # noqa: T201
        else:
            console.print(
                f"[red]Error:[/red] {repo} has not been indexed yet."
                " Run [bold]rtm index[/bold] first."
            )
        raise typer.Exit(1)

    from repo_time_machine.agent.pipeline import Pipeline

    if output != "json":
        console.print(f"\n[bold]Question:[/bold] {question}\n")

    pipeline = Pipeline(repo_path=repo, llm_model=llm_model, top_k=top_k)
    if not pipeline.ready:
        if output == "json":
            print(json.dumps({"error": "Could not load indexes."}))  # noqa: T201
        else:
            console.print(
                "[red]Error:[/red] Could not load indexes. Re-run [bold]rtm index[/bold]."
            )
            for err in pipeline.load_errors:
                console.print(f"  [dim]{err}[/dim]")
        raise typer.Exit(1)

    if output != "json":
        if pipeline.partial:
            console.print(
                "[yellow]Warning:[/yellow] Only partial indexes loaded; results may be limited."
            )
            for err in pipeline.load_errors:
                console.print(f"  [dim]{err}[/dim]")
        if pipeline.has_issues:
            console.print("[dim]GitHub issues/PRs available for enrichment[/dim]")

    with console.status("[dim]Thinking...[/dim]") if output != "json" else _noop_context():
        answer = pipeline.ask(question, skip_llm=raw)

    if output == "json":
        result = answer.to_dict()
        result["question"] = question
        print(json.dumps(result, indent=2))  # noqa: T201
        return

    if raw:
        console.print("[cyan]Raw evidence mode (--raw)[/cyan]\n")
    elif answer.used_llm:
        console.print("[green]LLM-synthesized answer[/green]\n")
    else:
        console.print("[yellow]Evidence-only answer (Ollama not running)[/yellow]\n")

    md = Markdown(answer.render())
    console.print(md)


if __name__ == "__main__":
    app()
