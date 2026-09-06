from langchain_core.prompts import ChatPromptTemplate

from app.services.llm.chat_model import chat_model
from app.utils.prompt_loader import load_prompt


def generate(state):
    """
    Generate answer using retrieved context + LLM.
    """

    # 1. Load system prompt from .txt file
    system_prompt = load_prompt(
        "generation/rag_system_prompt.txt"
    )

    # 2. Build LangChain prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # 3. Prepare retrieved context
    docs_content = "\n\n".join(
        doc.page_content for doc in state["context"]
    )

    # 4. Format messages for LLM
    messages = prompt.invoke({
        "question": state["messages"][-1].content,
        "context": docs_content
    })

    # 5. Call LLM
    response = chat_model.invoke(messages)

    # 6. Return updated state
    return {
        "answer": response.content
    }