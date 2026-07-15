# MetaRAG Documentation

> **MetaRAG** is a modular Retrieval-Augmented Generation (RAG) framework for building, experimenting with, and benchmarking retrieval pipelines.

MetaRAG breaks a RAG system into independent, interchangeable components. Replace any stage—loader, chunker, embeddings, retriever, or generator—without changing the rest of your application.

---

## Architecture at a Glance

```text
                        MetaRAG Workflow

    Documents
        │
        ▼
 ┌─────────────────┐
 │ Document Loader │
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │    Chunker      │
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │   Embeddings    │
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │   Vector Store  │
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │    Retriever    │
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │    Pipeline     │
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │    Generator    │
 └─────────────────┘
        │
        ▼
      Answer
```

---

# Features

| Feature | Description |
|----------|-------------|
| 📄 Document Loading | Load PDFs, text files and Markdown documents |
| ✂️ Multiple Chunkers | Six interchangeable chunking strategies |
| 🧠 Pluggable Embeddings | Works with local or cloud embedding models |
| 🗂 Multiple Vector Stores | InMemory, FAISS and Chroma support |
| 🔍 Multiple Retrievers | Dense, BM25, Hybrid and MMR retrieval |
| ⚙️ Retrieval Pipelines | Straight, MultiQuery, HyDE, Reranked and Full pipelines |
| 🤖 LLM Generation | Compatible with Ollama and custom generators |
| 🧩 Extensible | Implement your own modules using simple interfaces |

---

# Project Structure

```text
metarag/
│
├── core/
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vectordb.py
│   └── retriever.py
│
├── pipelines/
│   ├── pipeline.py
│   └── generator.py
│
├── examples/
│
├── tests/
│
└── docs/
```

---

# Documentation

| Guide | Purpose |
|------|---------|
| **Installation** | Install MetaRAG and optional dependencies |
| **Quick Start** | Build your first RAG pipeline in minutes |
| **Architecture** | Understand how each module works together |
| **Examples** | Run complete demonstrations of every module |
| **API Reference** | Explore the public classes and interfaces |

---

# Example Progression

MetaRAG includes runnable examples that gradually build a complete RAG pipeline.

```text
loader_demo.py
        │
        ▼
chunker_demo.py
        │
        ▼
embeddings_demo.py
        │
        ▼
vectordb_demo.py
        │
        ▼
retriever_demo.py
        │
        ▼
pipeline_demo.py
        │
        ▼
full_rag_demo.py
```

Each example introduces one new component while reusing the previous ones.

---

# Design Principles

### Modular

Every component can be used independently or replaced with a custom implementation.

### Extensible

Simple interfaces make it easy to integrate your own loaders, embedding models, retrievers, vector databases, and generators.

### Lightweight

Minimal abstractions with sensible defaults for rapid experimentation.

---

# Getting Started

If this is your first time using MetaRAG, follow the documentation in this order:

```text
Installation
      │
      ▼
Quick Start
      │
      ▼
Architecture
      │
      ▼
Examples
      │
      ▼
API Reference
```

---

## Next Step

➡ Continue with **Installation** to install MetaRAG and its optional dependencies.