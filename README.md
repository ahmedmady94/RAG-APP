# RAG Evaluation Framework

A modular **Retrieval-Augmented Generation (RAG) framework** built with Python, LangChain, and LangGraph.

The project is designed to experiment with and evaluate different RAG components independently, including:

* Vector retrieval
* Hybrid retrieval
* BM25 retrieval
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder reranking
* LLM-based generation
* Retrieval evaluation
* Generation evaluation
* Latency evaluation
* Persistent vector databases
* Embedding caching

The architecture separates **retrieval**, **generation**, **evaluation**, **indexing**, and **infrastructure**, making it easy to experiment with different RAG configurations without rewriting the entire application.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │       Dataset       │
                         │     HotpotQA / PDF  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Ingestion      │
                         │                     │
                         │ Load → Chunk →      │
                         │ Sanitize Metadata   │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        ┌───────────────────┐              ┌───────────────────┐
        │   Vector Index    │              │     BM25 Index    │
        │                   │              │                   │
        │ Chroma / InMemory │              │   Keyword Search  │
        └─────────┬─────────┘              └─────────┬─────────┘
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Retrieval      │
                         │                     │
                         │ Vector / BM25 /     │
                         │ Hybrid / RRF        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Reranking      │
                         │                     │
                         │ Cross-Encoder       │
                         │ (optional)          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Generation      │
                         │                     │
                         │      LLM            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       Answer        │
                         └─────────────────────┘


                    Evaluation Framework
                    ─────────────────────

          ┌────────────────┬────────────────┬────────────────┐
          │                │                │                │
          ▼                ▼                ▼                │
   Retrieval Eval   Generation Eval   Latency Eval           │
          │                │                │                │
          ▼                ▼                ▼                │
     Recall@K            Faithfulness    Avg Latency         │
     MRR                                                    │
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                   │
                                   ▼
                              CSV Results
```

---

## Key Features

### 1. Modular Retrieval

The framework supports multiple retrieval strategies:

```text
Vector Retrieval
      │
      ├── Semantic similarity
      │
      ▼

BM25 Retrieval
      │
      ├── Keyword matching
      │
      ▼

Hybrid Retrieval
      │
      ├── Vector + BM25
      │
      ▼

RRF
      │
      ├── Rank-based fusion
      │
      ▼

Reranking
      │
      └── Cross-Encoder
```

Retrieval strategies can be changed through configuration rather than rewriting the application.

---

### 2. Vector Retrieval

Semantic retrieval uses embeddings to represent documents and queries in vector space.

The project supports persistent **Chroma** storage as well as in-memory vector storage.

Example:

```python
retrieved_docs = retriever.invoke(query)
```

---

### 3. BM25 Retrieval

BM25 provides lexical retrieval based on term frequency and inverse document frequency.

This is particularly useful when exact terms, names, or keywords are important.

The implementation uses:

```text
rank_bm25
```

through LangChain's BM25 retriever.

---

### 4. Hybrid Retrieval

Hybrid retrieval combines semantic and lexical retrieval:

```text
Query
 │
 ├──────────────► Vector Search
 │
 └──────────────► BM25 Search
                       │
                       ▼
                  Result Fusion
```

The project uses **Reciprocal Rank Fusion (RRF)** to combine rankings.

RRF allows results from different retrieval systems to be combined based on their rank rather than requiring the scores from the two systems to be directly comparable.

---

### 5. Cross-Encoder Reranking

An optional reranking stage can be applied after retrieval.

```text
Query
  │
  ▼
Retriever
  │
  ▼
Top-N Documents
  │
  ▼
Cross-Encoder
  │
  ▼
Reranked Top-K
```

The Cross-Encoder evaluates the query and document together:

```text
(query, document)
        │
        ▼
   Cross-Encoder
        │
        ▼
     relevance
       score
```

This provides a more precise relevance estimate than relying solely on embedding similarity.

---

## Evaluation

The project deliberately separates **retrieval evaluation** from **generation evaluation**.

This allows retrieval quality to be measured independently from LLM generation quality.

### Retrieval Evaluation

Current retrieval metrics include:

#### Recall@K

Measures whether the relevant supporting documents appear within the top K retrieved documents.

```text
Recall@K =
relevant retrieved documents
─────────────────────────────
total relevant documents
```

#### Mean Reciprocal Rank (MRR)

Measures how high the first relevant document appears in the ranking.

```text
MRR = average(1 / rank_of_first_relevant_document)
```

---

### Generation Evaluation

Generation evaluation currently includes:

#### Faithfulness

Measures whether the generated answer is supported by the retrieved context.

This helps identify cases where the LLM produces an answer that is not grounded in the retrieved documents.

---

### Latency Evaluation

Latency is measured independently for evaluation runs.

This allows different retrieval and generation configurations to be compared based on:

* Retrieval performance
* Generation quality
* Execution time

---

## Dataset

The main benchmark used in this project is:

**HotpotQA**

Specifically:

```text
hotpot_dev_distractor_v1.json
```

HotpotQA is useful for evaluating multi-hop question answering because questions often require information from multiple supporting documents.

Example question:

```text
What government position was held by the woman
who portrayed Corliss Archer in the film Kiss and Tell?
```

The benchmark provides:

* Questions
* Context documents
* Supporting facts
* Answers

The supporting facts are used to evaluate retrieval quality.

---

## Project Structure

```text
rag_app/
│
├── app/
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── ingestion/
│   │   └── ingest.py
│   │
│   ├── loaders/
│   │   ├── document_loader.py
│   │   └── hotpot_loader.py
│   │
│   ├── retrieval/
│   │   │
│   │   ├── chunking.py
│   │   │
│   │   ├── indexes/
│   │   │   ├── vector_index.py
│   │   │   └── bm25_index.py
│   │   │
│   │   └── rerankers/
│   │       ├── cross_encoder_reranker.py
│   │       └── reranker_factory.py
│   │
│   ├── services/
│   │   │
│   │   ├── embeddings/
│   │   │   └── cached_embeddings.py
│   │   │
│   │   └── vectorstores/
│   │       ├── chroma_store.py
│   │       ├── in_memory_store.py
│   │       ├── vectorstore_factory.py
│   │       └── metadata_utils.py
│   │
│   ├── graphs/
│   │   └── rag_graph.py
│   │
│   ├── nodes/
│   │   ├── retrieve.py
│   │   └── generate.py
│   │
│   └── evaluators/
│       │
│       ├── retrieval/
│       │   ├── retrieval_eval.py
│       │   └── retrieval_runner.py
│       │
│       ├── generation/
│       │   ├── generation_eval.py
│       │   └── generation_runner.py
│       │
│       └── latency/
│           └── latency_eval.py
│
├── data/
│   ├── raw/
│   ├── experiments/
│   │   ├── retrieval/
│   │   └── generation/
│   └── chroma_db/
│
├── main_test.py
├── requirements.txt
├── .env
└── README.md
```

---

## Configuration

Configuration is centralized in:

```text
app/config/settings.py
```

Typical configuration includes:

```python
RETRIEVER_TYPE = "vector"

VECTOR_DB = "chroma"

CHUNK_SIZE = 1500

CHUNK_OVERLAP = 200

TOP_K = 5

EVAL_K = 5

EVAL_LIMIT = 1000
```

Possible retrieval configurations include:

```text
vector
hybrid
```

Possible vector stores include:

```text
in_memory
chroma
```

Reranking can be enabled/disabled independently.

---

## Embedding Cache

Embedding generation can be expensive when repeatedly running experiments.

The project therefore uses cached embeddings.

Conceptually:

```text
Document
   │
   ▼
Embedding Request
   │
   ├── Already cached? ──► Use cached embedding
   │
   └── Not cached? ──────► OpenAI Embeddings
                                  │
                                  ▼
                               Cache
```

This allows experiments to be repeated without unnecessarily regenerating embeddings for unchanged documents.

---

## Persistent Vector Storage

The project supports Chroma with a local persistent database:

```text
data/chroma_db/
```

This allows the vector index to survive application restarts.

The vector store is abstracted behind a factory so the application does not need to know which vector database implementation is being used.

---

## RAG Graph

The application uses **LangGraph** to orchestrate the RAG workflow.

Conceptually:

```text
User Question
      │
      ▼
   Retrieve
      │
      ▼
   Rerank
      │
      ▼
   Generate
      │
      ▼
    Answer
```

Retrieval and generation graphs can be evaluated independently.

---

## Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd rag_app
```

### 2. Create a virtual environment

```bash
python -m venv langchain_env
```

Activate it on Windows:

```bash
langchain_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For BM25 retrieval:

```bash
pip install rank_bm25
```

### 4. Configure environment variables

Create:

```text
.env
```

and add your API key:

```text
OPENAI_API_KEY=your_api_key
```

Do not commit `.env` to GitHub.

Add it to `.gitignore`:

```text
.env
```

---

## Running Retrieval Evaluation

Run:

```bash
python main_test.py
```

The retrieval runner evaluates the configured retrieval strategy and records results under:

```text
data/experiments/retrieval/results.csv
```

Example output:

```text
==============================
FINAL RESULTS
==============================

Recall@5: 0.9180
MRR: 0.9498
Average Latency: 4.2497 sec
```

---

## Running Generation Evaluation

Generation evaluation is separated from retrieval evaluation.

It evaluates metrics such as:

```text
Faithfulness
Latency
```

Results are stored under:

```text
data/experiments/generation/results.csv
```

---

## Experiment Tracking

Each experiment records configuration parameters alongside evaluation metrics.

Example:

| Retriever | Chunk Size | Overlap | Top-K | Recall@5 |    MRR | Latency |
| --------- | ---------: | ------: | ----: | -------: | -----: | ------: |
| vector    |       1500 |     200 |     5 |    0.918 | 0.9498 |   4.25s |
| hybrid    |       1500 |     200 |     5 |    0.917 | 0.9498 |   3.53s |

This makes it possible to compare retrieval strategies systematically instead of relying on individual examples.

---

## Experimental Goals

The project is intended as an experimentation framework rather than a single fixed RAG implementation.

Future experiments can include:

* Different chunk sizes
* Different chunk overlaps
* Different embedding models
* Different vector databases
* BM25 vs vector retrieval
* Hybrid retrieval
* Different RRF parameters
* Cross-Encoder reranking
* Different Top-K values
* Different reranking depths
* Different LLMs
* Different prompting strategies
* Retrieval vs generation latency
* Retrieval quality vs answer quality

---

## Design Principles

The project follows several software engineering principles.

### Separation of Concerns

Retrieval, indexing, generation, evaluation, and infrastructure are kept separate.

### Configuration-Driven Experiments

Experiment parameters are centralized instead of being hardcoded throughout the application.

### Dependency Inversion

Higher-level components interact with abstractions/factories rather than depending directly on a specific vector database.

### Modular Components

Retrievers, rerankers, vector stores, embeddings, and evaluators can be replaced independently.

### Reproducible Evaluation

Experiments are recorded with their configuration and metrics in CSV files.

---

## Technology Stack

| Technology           | Purpose                    |
| -------------------- | -------------------------- |
| Python               | Core language              |
| LangChain            | LLM/RAG framework          |
| LangGraph            | RAG workflow orchestration |
| OpenAI               | Embeddings / LLM           |
| Chroma               | Persistent vector database |
| SentenceTransformers | Cross-Encoder reranking    |
| BM25                 | Lexical retrieval          |
| HotpotQA             | Retrieval benchmark        |
| Pydantic             | Configuration / validation |

---

## Why This Project?

Many RAG implementations focus primarily on:

```text
Retrieve → Generate
```

This project focuses on understanding **why a RAG system performs well or poorly**.

Instead of treating RAG as a black box, each stage can be independently measured:

```text
                RAG
                 │
        ┌────────┴────────┐
        │                 │
   Retrieval          Generation
        │                 │
   Recall@K          Faithfulness
   MRR               Answer Quality
   Latency           Latency
```

This makes the framework useful for systematic experimentation and benchmarking of RAG architectures.

---

## Future Work

Potential future improvements include:

* [ ] Add NDCG evaluation
* [ ] Add Precision@K
* [ ] Add answer correctness evaluation
* [ ] Add LLM-as-a-judge evaluation
* [ ] Add multiple reranker models
* [ ] Add configurable RRF parameters
* [ ] Add Qdrant support
* [ ] Add FAISS support
* [ ] Add Elasticsearch/OpenSearch retrieval
* [ ] Add experiment visualization
* [ ] Add automated benchmark reports
* [ ] Add unit tests
* [ ] Add integration tests
* [ ] Add Docker support
* [ ] Add API layer with FastAPI
* [ ] Add production deployment configuration

---

## License

This project is intended for educational, research, and experimentation purposes.

Add your preferred license here, for example:

```text
MIT License
```

---

## Author

**Ahmed Mady**

Data Scientist → AI Engineering / RAG Systems

This project represents an ongoing exploration of production-oriented RAG architecture, retrieval evaluation, reranking, and AI engineering practices.
