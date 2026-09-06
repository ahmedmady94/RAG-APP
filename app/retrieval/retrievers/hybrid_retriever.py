"""
Hybrid Retriever.

Combines dense retrieval
(Vector Search)
with sparse retrieval
(BM25).
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.retrieval.retrievers.vector_retriever import (
    VectorRetriever
)

from app.retrieval.indexes.bm25_index import (
    get_index
)

from app.config.settings import TOP_K


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval using
    Vector Search + BM25.
    """

    k: int = TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """
        Retrieve documents using both
        dense and sparse retrieval.
        """

        # -----------------------------
        # Dense retrieval
        # -----------------------------

        vector_docs = VectorRetriever(
            k=self.k
        ).invoke(query)

        # -----------------------------
        # Sparse retrieval
        # -----------------------------

        bm25_docs = get_index().invoke(query)

        # -----------------------------
        # Merge
        # -----------------------------
                
        # Reciprocal Rank Fusion (RRF)
        # ---------------------------------------------------

        RRF_K = 60


        def document_key(doc):

            return (
                doc.metadata.get("source"),
                doc.page_content
            )


        # Build rank lookup tables
        vector_ranks = {
            document_key(doc): rank
            for rank, doc in enumerate(
                vector_docs,
                start=1
            )
        }

        bm25_ranks = {
            document_key(doc): rank
            for rank, doc in enumerate(
                bm25_docs,
                start=1
            )
        }


        # Collect all unique documents
        all_docs = {}

        for doc in vector_docs + bm25_docs:

            key = document_key(doc)

            if key not in all_docs:

                all_docs[key] = doc


        # Compute RRF scores
        rrf_scores = {}

        for key in all_docs:

            score = 0.0

            if key in vector_ranks:

                score += 1 / (
                    RRF_K + vector_ranks[key]
                )

            if key in bm25_ranks:

                score += 1 / (
                    RRF_K + bm25_ranks[key]
                )

            rrf_scores[key] = score


        # Sort by descending RRF score
        sorted_keys = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )


        # Recover Document objects
        reranked_docs = [
            all_docs[key]
            for key in sorted_keys
        ]


        # -----------------------------
        # Return top-k
        # -----------------------------

        return reranked_docs[: self.k]
