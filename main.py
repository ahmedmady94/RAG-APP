from langchain_core.messages import HumanMessage

from app.graphs.rag_graph import build_rag_graph
from app.ingestion.ingest import ingest


# ---------------------------------------------
# Build graph once at startup
# ---------------------------------------------

graph = build_rag_graph()


# ---------------------------------------------
# Ingest knowledge base
# ---------------------------------------------

ingest(
        source="data/raw/hotpot_dev_distractor_v1.json",
        source_type="hotpot",
        limit=200
    )

# ---------------------------------------------
# Main application
# ---------------------------------------------

def main():

    print("\n=== RAG Assistant ===\n")

    while True:

        query = input("Ask a question (or type 'exit'): ")

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        initial_state = {
            "messages": [
                HumanMessage(content=query)
            ]
        }

        config = {
            "configurable": {
                "thread_id": "local-session"
            }
        }

        try:

            result = graph.invoke(
                initial_state,
                config=config
            )

            print("\n=== ANSWER ===\n")

            print(result["answer"])

            print("\n" + "=" * 50 + "\n")

        except Exception as e:

            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()