from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from app.state.rag_state import State
from app.nodes.retrieve import retrieve
from app.nodes.generate import generate



def build_rag_graph():

    builder = StateGraph(State)

    # Add nodes
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    # Define flow
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("retrieve", END)

    # Checkpointing
    memory = MemorySaver()

    graph = builder.compile(
        checkpointer=memory
    )

    return graph



def build_retrieval_graph():

    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", END)
    # Checkpointing
    memory = MemorySaver()

    graph = builder.compile(
        checkpointer=memory
    )

    return graph



def build_generation_graph():

    builder = StateGraph(State)

    # Add nodes
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    # Define flow
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("retrieve", END)

    # Checkpointing
    memory = MemorySaver()

    graph = builder.compile(
        checkpointer=memory
    )

    return graph

