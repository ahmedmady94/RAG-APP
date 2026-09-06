"""
Central ingestion orchestrator.

Responsibilities:
- Load documents
- Chunk documents
- Build vector index
- Build BM25 index
"""

from typing import Literal

from app.services.vectorstores.metadata_utils import (
    sanitize_metadata
)

from app.loaders.document_loader import (
    load_document
)

from app.loaders.hotpot_loader import (
    load_hotpotqa
)

from app.retrieval.chunking import (
    split_documents
)

from app.retrieval.indexes.vector_index import (
    index_documents
)

from app.retrieval.indexes.bm25_index import (
    build_index
)

from app.config.settings import (
    TOP_K
)

from app.services.vectorstores.vectorstore_factory import (
    reset_vector_store
)




# ---------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------

def ingest(
    source: str,
    source_type: Literal["file", "hotpot"] = "file",
    limit: int | None = None
):
    """
    Unified ingestion pipeline.

    Args:
        source:
            Path to file or dataset.

        source_type:
            "file"
            "hotpot"

        limit:
            Optional dataset limit.
    """

    # ---------------------------------------------------
    # Load documents
    # ---------------------------------------------------

    if source_type == "file":

        docs = load_document(source)

    elif source_type == "hotpot":

        docs = load_hotpotqa(
            source,
            limit=limit
        )

    else:

        raise ValueError(
            f"Unsupported source_type: {source_type}"
        )





    # ---------------------------------------------------
    # Chunk documents
    # ---------------------------------------------------

    chunks = split_documents(docs)



    # ---------------------------------------------------
    # Sanitize metadata for vector stores
    # ---------------------------------------------------

    for chunk in chunks:

        chunk.metadata = sanitize_metadata(
            chunk.metadata
        )



    # ---------------------------------------------------
    # clear the collection
    # ---------------------------------------------------
    reset_vector_store()


    # ---------------------------------------------------
    # Build vector index
    # ---------------------------------------------------

    index_documents(chunks)

    # ---------------------------------------------------
    # Build BM25 index
    # ---------------------------------------------------

    build_index(
        chunks,
        k=TOP_K
    )

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------

    print("\n==============================")

    print("INGESTION COMPLETE")

    print("==============================")

    print(f"Source Type : {source_type}")

    print(f"Source      : {source}")

    print(f"Documents   : {len(docs)}")

    print(f"Chunks      : {len(chunks)}")

    print(f"Top-K       : {TOP_K}")

    print("==============================\n")