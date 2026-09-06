"""
Vector index.

Provides a single global interface
to the application's vector store.
"""

from app.services.vectorstores.vectorstore_factory import (
    get_vector_store
)
from app.config.settings import VECTOR_DB_BATCH_SIZE
from langchain_core.documents import Document


def index_documents(documents:list[Document]):
    """
    Add documents to the vector store.
    """
    vector_store = get_vector_store()

    for i in range(0, len(documents), VECTOR_DB_BATCH_SIZE):

        batch = documents[i:i + VECTOR_DB_BATCH_SIZE]
        vector_store.add_documents(batch)



def similarity_search(
    query: str,
    k: int = 5
):
    """
    Perform similarity search
    against the vector store.
    """
    vector_store = get_vector_store()
    return vector_store.similarity_search(
        query=query,
        k=k
    )