import json

from langchain_core.documents import Document


def load_hotpotqa(
    path: str,
    limit: int | None = None
):
    """
    Convert HotpotQA samples into LangChain Documents.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    documents = []

    for sample in data:

        question = sample["question"]

        answer = sample["answer"]

        contexts = sample["context"]

        supporting_facts = sample["supporting_facts"]

        for title, sentences in contexts:

            content = " ".join(sentences)

            doc = Document(

                page_content=content,

                metadata={
                    "source_title": title,
                    "question": question,
                    "answer": answer,
                    "supporting_facts": supporting_facts,
                }
            )

            documents.append(doc)

    return documents