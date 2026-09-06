"""
Retrieve node.

Retrieves relevant documents
using the configured retriever.
"""

from app.state.rag_state import State

from app.retrieval.retrievers.retriever_factory import (
    get_retriever
)


def retrieve(state: State):
    """
    Retrieve relevant documents
    for the user's latest question.
    """

    question = state["messages"][-1].content

    retriever = get_retriever()

    retrieved_docs = retriever.invoke(
        question
    )

    # print("\n==============================")

    # print("RETRIEVED DOCUMENTS")

    # print("==============================")

    # for idx, doc in enumerate(retrieved_docs, start=1):

    #     print(f"\n----- Document {idx} -----")

    #     print(doc.page_content)

    # print("\n==============================")

    return {
        "context": retrieved_docs
    }