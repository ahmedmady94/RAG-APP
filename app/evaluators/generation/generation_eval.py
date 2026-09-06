from app.services.llm.chat_model import (
    chat_model
)


def faithfulness_score(
    answer: str,
    context: str
) -> int:

    prompt = f"""
You are evaluating whether an answer is fully supported
by the provided context.

Context:
{context}

Answer:
{answer}

Return ONLY:

1 -> if the answer is fully supported
0 -> if the answer contains hallucinations
"""

    response = chat_model.invoke(prompt)

    score = response.content.strip()

    try:
        return int(score)

    except Exception:

        return 0