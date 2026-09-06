import csv
import os

from app.ingestion.ingest import ingest

from app.loaders.hotpot_loader import (
    load_hotpotqa
)

from app.graphs.rag_graph import (
    build_generation_graph
)

from app.evaluators.generation.generation_eval import (
    faithfulness_score
)

from app.evaluators.latency.latency_eval import (
    measure_latency
)

from app.config.settings import (
    DATASET_PATH,
    EVAL_LIMIT,
    RETRIEVER_TYPE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

RESULTS_PATH = (
    "data/experiments/generation/results.csv"
)


# ---------------------------------------------------
# Build graph once
# ---------------------------------------------------

graph = build_generation_graph()


# ---------------------------------------------------
# CSV Initialization
# ---------------------------------------------------

def initialize_results_csv():

    os.makedirs(
        "data/experiments/generation",
        exist_ok=True
    )

    if (
        not os.path.exists(RESULTS_PATH)
        or
        os.path.getsize(RESULTS_PATH) == 0
    ):

        with open(
            RESULTS_PATH,
            mode="w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "retriever_type",
                "chunk_size",
                "chunk_overlap",
                "top_k",
                "eval_limit",
                "faithfulness",
                "avg_latency_sec"
            ])


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

def log_experiment_result(
    faithfulness,
    latency
):

    with open(
        RESULTS_PATH,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            RETRIEVER_TYPE,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            TOP_K,
            EVAL_LIMIT,
            round(faithfulness, 4),
            round(latency, 4)
        ])


# ---------------------------------------------------
# Runner
# ---------------------------------------------------

def run_generation_experiment():

    initialize_results_csv()

    # ---------------------------------------------------
    # Ingest dataset
    # ---------------------------------------------------

    ingest(
        source=DATASET_PATH,
        source_type="hotpot",
        limit=EVAL_LIMIT
    )

    # ---------------------------------------------------
    # Load dataset
    # ---------------------------------------------------

    dataset = load_hotpotqa(DATASET_PATH)

    dataset = dataset[:EVAL_LIMIT]

    num_samples = len(dataset)

    total_faithfulness = 0

    total_latency = 0

    # ---------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------

    for idx, sample in enumerate(dataset):

        question = sample.metadata["question"]

        print("\n==============================")

        print(f"QUESTION {idx + 1}")

        print("==============================")

        print(question)

        config = {
            "configurable": {
                "thread_id": "generation-eval"
            }
        }

        response, latency = measure_latency(
            graph.invoke,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            },
            config=config
        )

        retrieved_docs = response["context"]

        generated_answer = response["answer"]

        context_text = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        # ---------------------------------------------------
        # Faithfulness
        # ---------------------------------------------------

        faithfulness = faithfulness_score(
            answer=generated_answer,
            context=context_text
        )

        total_faithfulness += faithfulness

        total_latency += latency

        print(
            f"\nFaithfulness: "
            f"{faithfulness}"
        )

        print(
            f"Latency: "
            f"{latency:.4f} sec"
        )

    # ---------------------------------------------------
    # Final metrics
    # ---------------------------------------------------

    final_faithfulness = (
        total_faithfulness / num_samples
    )

    avg_latency = total_latency / num_samples

    print("\n==============================")

    print("FINAL RESULTS")

    print("==============================")

    print(
        f"\nFaithfulness: "
        f"{final_faithfulness:.4f}"
    )

    print(
        f"Average Latency: "
        f"{avg_latency:.4f} sec"
    )

    log_experiment_result(
        faithfulness=final_faithfulness,
        latency=avg_latency
    )

    print("\nExperiment logged successfully.")