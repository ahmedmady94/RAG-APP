from langchain_core.vectorstores import InMemoryVectorStore
from app.services.embeddings.cached_embeddings import EMBEDDINGS



def create_in_memory_store():

    return InMemoryVectorStore(
        embedding=EMBEDDINGS
    )
