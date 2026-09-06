from langchain_core.documents import Document


def recall_at_k(
    retrieved_docs: list[Document],
    supporting_facts: list[list],
    k: int = 5
) -> int:
    """
    Recall@K for HotpotQA.

    Returns 1 if at least one gold
    supporting document appears in top-k.
    """

    top_k_docs = retrieved_docs[:k]

    gold_titles = {
        fact[0]
        for fact in supporting_facts
    }

    retrieved_titles = {
        doc.metadata.get("source_title")
        for doc in top_k_docs
    }

    if gold_titles.intersection(retrieved_titles):
        return 1

    return 0


def mean_reciprocal_rank(
    retrieved_docs: list[Document],
    supporting_facts: list[list]
) -> float:
    """
    Mean Reciprocal Rank (MRR)
    for retrieval ranking quality.
    """

    gold_titles = {
        fact[0]
        for fact in supporting_facts
    }

    for idx, doc in enumerate(retrieved_docs):

        title = doc.metadata.get("source_title")

        if title in gold_titles:
            return 1 / (idx + 1)

    return 0