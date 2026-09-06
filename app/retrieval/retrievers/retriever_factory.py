"""
Retriever Factory.

Returns the configured retriever
based on the application settings.
"""

from app.config.settings import RETRIEVER_TYPE

from app.retrieval.retrievers.vector_retriever import (
    VectorRetriever
)

from app.retrieval.retrievers.hybrid_retriever import (
    HybridRetriever
)


def get_retriever():
    """
    Return the configured retriever.
    """

    if RETRIEVER_TYPE == "vector":

        return VectorRetriever()

    if RETRIEVER_TYPE == "hybrid":

        return HybridRetriever()

    raise ValueError(
        f"Unknown retriever type: {RETRIEVER_TYPE}"
    )