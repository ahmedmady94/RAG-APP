import os


# ===================================================
# API KEYS
# ===================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)


# ===================================================
# RETRIEVAL CONFIGURATION
# ===================================================

# Available options:
# - "vector"
# - "hybrid"

RETRIEVER_TYPE = "hybrid"

TOP_K = 5




# ---------------------------------------------------
# Reranker Configuration
# ---------------------------------------------------

RERANKER_TYPE = "cross_encoder"

# Available:
# "none"
# "cross_encoder"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RERANK_TOP_K = 5

# ===================================================
# LLM CONFIGURATION
# ===================================================

OPENAI_MODEL = "gpt-4o-mini"


# ===================================================
# EMBEDDING CONFIGURATION
# ===================================================

EMBEDDING_MODEL = "text-embedding-3-large"



# ---------------------------------------------------
# Vector Store Configuration
# ---------------------------------------------------

VECTOR_DB = "chroma"

# Available:
# "in_memory"
# "chroma"

VECTOR_DB_BATCH_SIZE = 5000

# ===================================================
# CHUNKING CONFIGURATION
# ===================================================

CHUNK_SIZE = 1500

CHUNK_OVERLAP = 200


# ===================================================
# EVALUATION CONFIGURATION
# ===================================================

DATASET_PATH = (
    "data/raw/hotpot_dev_distractor_v1.json"
)

EVAL_LIMIT = 5

EVAL_K = 1