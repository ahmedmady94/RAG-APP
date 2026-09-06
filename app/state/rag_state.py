from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.documents import Document


class State(TypedDict):

    # question: str
    context: list[Document]
    answer: str
    messages: Annotated[list, add_messages]