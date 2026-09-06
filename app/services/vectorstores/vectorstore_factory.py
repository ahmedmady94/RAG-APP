"""
Factory for vector stores.

This module returns the application's
configured vector store implementation.
"""

from app.config.settings import VECTOR_DB

from app.services.vectorstores.in_memory_store import (
    create_in_memory_store
)

from app.services.vectorstores.chroma_store import (
    create_chroma_store
)


def get_vector_store():
    """
    Return the configured vector store.
    """

    if VECTOR_DB == "in_memory":

        return create_in_memory_store()

    elif VECTOR_DB == "chroma":
    
        return create_chroma_store()

    else:

        raise ValueError(
            f"Unknown vector store: {VECTOR_DB}"
        )
    


def reset_vector_store():

    if VECTOR_DB == "chroma":

        store = create_chroma_store()

        store.delete_collection()

    elif VECTOR_DB == "in_memory":

        # nothing to do
        pass