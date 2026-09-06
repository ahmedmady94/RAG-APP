def faithfulness_score(
    answer: str,
    retrieved_context: str
):
    """
    Naive grounding metric.
    """

    return int(
        answer.lower() in retrieved_context.lower()
    )