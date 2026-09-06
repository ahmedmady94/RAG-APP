"""
Factory for rerankers.
"""


from app.retrieval.rerankers.cross_encoder_reranker import (
    CrossEncoderReranker
)
from app.config.settings import RERANKER_TYPE
from app.config.settings import RERANKER_MODEL
from app.config.settings import RERANK_TOP_K

def get_reranker():

    if RERANKER_TYPE == "none":
        return None

    elif RERANKER_TYPE == "cross_encoder":
        return CrossEncoderReranker(
            model_name=RERANKER_MODEL,
            top_k=RERANK_TOP_K
        )
    
    # elif RERANKER_TYPE == "cohere":
    #     return CohereReranker()

    # elif RERANKER_TYPE == "jina":
    #     return JinaReranker()

    else:
        raise ValueError(
            f"Unknown reranker: {RERANKER_TYPE}"
        )