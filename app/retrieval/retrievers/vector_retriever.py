"""
Vector Retriever.

Retrieves documents using
vector similarity search.
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.retrieval.indexes.vector_index import (
    similarity_search
)

from app.config.settings import TOP_K


class VectorRetriever(BaseRetriever):
    """
    Dense retrieval using embeddings.
    """

    k: int = TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """
        Retrieve top-k similar documents.
        """

        return similarity_search(
            query=query,
            k=self.k
        )