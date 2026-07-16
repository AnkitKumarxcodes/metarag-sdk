# Architecture

MetaRAG separates a Retrieval-Augmented Generation (RAG) system into independent modules. Each module has a single responsibility and can be replaced without changing the rest of the workflow.

The framework can be used at two levels:

- **Framework API** — build and benchmark complete RAG systems.
- **Toolkit API** — use individual components independently.

---

## Project Structure

```text
metarag/
│
├── metarag.py          High-level framework
├── defaults.py         Shared configuration
│
├── core/
│   ├── loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_db.py
│   └── retriever.py
│
├── pipelines/
│   ├── generator.py
│   └── pipeline.py
│
├── Evaluator/
│   ├── evaluator.py
│   ├── scorer.py
│   └── metrics.py
│
└── router/
    ├── router.py
    ├── query_profiler.py
    ├── corpus_profiler.py
    └── probe_profiler.py
```

---

## Framework Layer

Most users only interact with a single class.

```python
rag = MetaRAG(
    docs="data",
    embeddings=...,
    generator=...
)

rag.fit()

rag.ask(...)

rag.benchmark(...)
```

Internally, MetaRAG builds and connects the lower-level components automatically.

---

## Internal Flow

```text
Documents
   │
   ▼
Loader ──► Chunker ──► Embeddings
                         │
                         ▼
                    Vector Database
                         │
                         ▼
                     Retriever
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   Retrieval Pipelines            Evaluation
          │                             │
          └──────────────┬──────────────┘
                         ▼
                       Router
                         │
                         ▼
                     Final Answer
```

Unlike many RAG libraries, benchmarking and evaluation are built into the framework rather than added as external scripts.

---

## Module Responsibilities

| Module | Responsibility |
|---------|----------------|
| `core/` | Loading, chunking, embeddings, vector stores and retrieval |
| `pipelines/` | Retrieval enhancement pipelines |
| `Evaluator/` | Pipeline evaluation and benchmarking |
| `router/` | Query profiling and pipeline selection |
| `metarag.py` | High-level orchestration |
| `defaults.py` | Shared configurable parameters |

---

## During `fit()`

Running

```python
rag.fit()
```

builds the retrieval system.

Typical output:

```text
Files Loaded      : 8

Documents Extracted : 101

Chunks Generated    : 333

Embeddings Generated

Vector Index Built

Pipelines Initialized : 7
```

---

## During `benchmark()`

Every configured pipeline is evaluated on the same set of queries.

```text
85 Queries

×

7 Pipelines

↓

595 Evaluations

↓

benchmark.csv
router_thresholds.json
```

The benchmark results can be inspected directly as a pandas DataFrame or exported as a CSV.

---

## Extending MetaRAG

Each major component exposes a lightweight interface.

You can replace:

- document loaders
- chunking strategies
- embedding models
- vector databases
- retrievers
- generators

without modifying the rest of the framework.

---

## Design Goals

MetaRAG aims to keep experimentation straightforward:

- independent components
- reusable pipelines
- reproducible benchmarks
- simple extension points

It is intended as a framework for experimenting with RAG systems rather than providing a single "best" pipeline.