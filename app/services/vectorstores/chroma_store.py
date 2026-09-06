from langchain_chroma import Chroma
from app.services.embeddings.cached_embeddings import EMBEDDINGS


def create_chroma_store():

    return Chroma(
        collection_name="hotpot",
        embedding_function=EMBEDDINGS,
        persist_directory="./data/chroma_db"
    )

