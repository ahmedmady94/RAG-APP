import csv
import os

from app.ingestion.ingest import ingest

from app.loaders.hotpot_loader import (
    load_hotpotqa
)

from app.graphs.rag_graph import (
    build_retrieval_graph
)

from app.evaluators.retrieval.retrieval_eval import (
    recall_at_k,
    mean_reciprocal_rank
)

from app.evaluators.latency.latency_eval import (
    measure_latency
)

from app.config.settings import (
    DATASET_PATH,
    EVAL_LIMIT,
    EVAL_K,
    RETRIEVER_TYPE,
    RERANKER_TYPE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

RESULTS_PATH = (
    "data/experiments/retrieval/results.csv"
)


# ---------------------------------------------------
# Build graph once
# ---------------------------------------------------

graph = build_retrieval_graph()


# ---------------------------------------------------
# CSV Initialization
# ---------------------------------------------------

def initialize_results_csv():

    os.makedirs(
        "data/experiments/retrieval",
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
                "reranker_type",
                "chunk_size",
                "chunk_overlap",
                "top_k",
                "eval_limit",
                f"recall@{EVAL_K}",
                "mrr",
                "avg_latency_sec"
            ])


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

def log_experiment_result(
    recall,
    mrr,
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
            RERANKER_TYPE,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            TOP_K,
            EVAL_LIMIT,
            round(recall, 4),
            round(mrr, 4),
            round(latency, 4)
        ])


# ---------------------------------------------------
# Runner
# ---------------------------------------------------

def run_retrieval_experiment():

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

    total_recall = 0

    total_mrr = 0

    total_latency = 0

    # ---------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------

    for idx, sample in enumerate(dataset):

        question = sample.metadata["question"]

        supporting_facts = sample.metadata[
            "supporting_facts"
        ]

        print("\n==============================")

        print(f"QUESTION {idx + 1}")

        print("==============================")

        print(question)

        config = {
            "configurable": {
                "thread_id": "retrieval-eval"
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

        # ---------------------------------------------------
        # Metrics
        # ---------------------------------------------------

        recall = recall_at_k(
            retrieved_docs=retrieved_docs,
            supporting_facts=supporting_facts,
            k=EVAL_K
        )

        mrr = mean_reciprocal_rank(
            retrieved_docs=retrieved_docs,
            supporting_facts=supporting_facts
        )

        total_recall += recall

        total_mrr += mrr

        total_latency += latency

        print(f"\nRecall@{EVAL_K}: {recall}")

        print(f"MRR: {mrr:.4f}")

        print(f"Latency: {latency:.4f} sec")

    # ---------------------------------------------------
    # Final metrics
    # ---------------------------------------------------

    final_recall = total_recall / num_samples

    final_mrr = total_mrr / num_samples

    avg_latency = total_latency / num_samples

    print("\n==============================")

    print("FINAL RESULTS")

    print("==============================")

    print(f"\nRecall@{EVAL_K}: {final_recall:.4f}")

    print(f"MRR: {final_mrr:.4f}")

    print(
        f"Average Latency: "
        f"{avg_latency:.4f} sec"
    )

    log_experiment_result(
        recall=final_recall,
        mrr=final_mrr,
        latency=avg_latency
    )

    print("\nExperiment logged successfully.")