"""
BM25 index.

Responsible for building and exposing
a global BM25 index.
"""

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


# ---------------------------------------------------
# Global BM25 Index
# ---------------------------------------------------

BM25_INDEX = None


# ---------------------------------------------------
# Build Index
# ---------------------------------------------------

def build_index(
    documents: list[Document],
    k: int = 5
) -> None:
    """
    Build the BM25 index.

    This should be called once during ingestion.
    """

    global BM25_INDEX

    BM25_INDEX = BM25Retriever.from_documents(
        documents
    )

    BM25_INDEX.k = k


# ---------------------------------------------------
# Get Index
# ---------------------------------------------------

def get_index() -> BM25Retriever:
    """
    Return the initialized BM25 index.
    """

    if BM25_INDEX is None:

        raise RuntimeError(
            "BM25 index has not been built. "
            "Run ingest() before retrieval."
        )

    return BM25_INDEX