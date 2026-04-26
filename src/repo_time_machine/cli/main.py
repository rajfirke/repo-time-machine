"""CLI commands: index and ask."""

import typer

app = typer.Typer(
    name="rtm",
    help="Repo Time Machine — ask questions backed by code, commits, and issues.",
    no_args_is_help=True,
)


@app.command()
def index(
    repo_path: str = typer.Argument(..., help="Path to the local git repository to index."),
    github_slug: str = typer.Option(
        None, "--github", "-g",
        help="Optional GitHub repo slug (owner/repo) to enrich with issues and PRs.",
    ),
):
    """Ingest a repository: index source files and extract git history."""
    typer.echo(f"Indexing repository at: {repo_path}")
    # TODO: call ingestion pipeline and persist index
    raise NotImplementedError("Ingestion not yet implemented.")


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
    typer.echo(f"Question: {question}")
    # TODO: classify question, retrieve evidence, build and render answer
    raise NotImplementedError("Answer pipeline not yet implemented.")


if __name__ == "__main__":
    app()
