from langchain.embeddings import CacheBackedEmbeddings

from app.services.embeddings.embeddings import underlying_embeddings
from app.services.storage.cache_store import store


EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)