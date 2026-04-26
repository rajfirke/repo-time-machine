"""Semantic search over source file chunks using FAISS + bge-small embeddings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeChunk:
    file: str
    start_line: int
    end_line: int
    content: str
    score: float = 0.0


class CodeRetriever:
    """
    Embeds source file chunks and answers semantic queries.

    index_path: directory where the FAISS index is persisted
    """

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        # TODO: load sentence-transformers model (bge-small-en-v1.5)
        # TODO: load or create FAISS index

    def index(self, chunks: list[dict]) -> None:
        """Embed and store a list of code chunks."""
        # TODO: implement
        raise NotImplementedError

    def query(self, question: str, top_k: int = 5) -> list[CodeChunk]:
        """Return the top-k most relevant code chunks for the question."""
        # TODO: implement
        raise NotImplementedError
