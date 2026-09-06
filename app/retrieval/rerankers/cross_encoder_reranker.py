"""
Cross-Encoder reranker.

Re-ranks retrieved documents using a
SentenceTransformers CrossEncoder model.
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str,
        top_k: int
    ):
        
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
       

    def rerank(
        self,
        query: str,
        documents: list[Document]
    ) -> list[Document]:

        if not documents:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        relevance_scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, relevance_scores),
            key=lambda x: x[1],
            reverse=True
        )

        reranked_docs = [
            doc
            for doc, score in ranked
        ]

        return reranked_docs[:self.top_k]