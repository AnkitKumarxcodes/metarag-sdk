# Architecture

MetaRAG is built around a simple idea:

> **Every stage of a Retrieval-Augmented Generation (RAG) system should be independent, composable, and replaceable.**

Instead of hiding the entire workflow behind a single API call, MetaRAG exposes each stage as a standalone module, allowing you to customize, benchmark, or replace any component.

---

# High-Level Architecture

```text
                           MetaRAG

        ┌───────────────────────────────────────────┐
        │              Source Documents             │
        └───────────────────────────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Document Loader   │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      Chunker        │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     Embeddings      │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Vector Database   │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     Retriever       │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      Pipeline       │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     Generator       │
                └─────────────────────┘
                           │
                           ▼
                      Final Answer
```

---

# Workflow

The complete retrieval pipeline follows seven stages.

```text
Load
  │
Chunk
  │
Embed
  │
Index
  │
Retrieve
  │
Optimize
  │
Generate
```

Each stage has a single responsibility.

---

# Component Overview

## 1. Document Loader

Responsible for reading files from disk and converting them into a standard document format.

```text
PDF
TXT
MD
 │
 ▼
Document
```

Output:

- Document text
- Metadata
- Loading report

---

## 2. Chunker

Splits large documents into smaller retrieval units.

```text
Document
     │
     ▼
┌────────────┐
│ Chunk 1    │
├────────────┤
│ Chunk 2    │
├────────────┤
│ Chunk 3    │
└────────────┘
```

Supported strategies include:

- Fixed
- Recursive
- Semantic
- Sentence
- Sliding Window
- Markdown

---

## 3. Embeddings

Transforms each chunk into a numerical vector.

```text
Chunk Text
      │
      ▼
Embedding Model
      │
      ▼
Dense Vector
```

MetaRAG supports any embedding model implementing the embedding interface.

Examples:

- Sentence Transformers
- Ollama Embeddings
- OpenAI Embeddings
- Custom Models

---

## 4. Vector Database

Stores vectors for efficient similarity search.

```text
Chunk
   │
Vector
   │
   ▼
Vector Database
```

Supported implementations:

| Database | Persistence |
|----------|-------------|
| InMemory | No |
| FAISS | Local |
| Chroma | Persistent |

---

## 5. Retriever

Retrieves the most relevant chunks for a user query.

```text
User Query
      │
      ▼
Embedding
      │
      ▼
Similarity Search
      │
      ▼
Top-k Chunks
```

Available retrievers:

- Dense
- BM25
- Hybrid
- MMR

---

## 6. Pipeline

Pipelines improve retrieval before generation.

Instead of simply retrieving chunks, a pipeline may:

- Expand the query
- Generate a hypothetical answer
- Merge retrieval results
- Re-rank chunks
- Remove duplicates

```text
Query
 │
 ▼
MultiQuery
 │
 ▼
Retrieve
 │
 ▼
Rerank
 │
 ▼
Deduplicate
 │
 ▼
Top Chunks
```

Available pipelines:

| Pipeline | Purpose |
|-----------|----------|
| Straight | Basic retrieval |
| MultiQuery | Query expansion |
| HyDE | Hypothetical document retrieval |
| Reranked | Cross-encoder reranking |
| Full | Combines multiple retrieval enhancements |

---

## 7. Generator

The generator receives the optimized context and produces the final answer.

```text
Question
      │
Retrieved Context
      │
      ▼
Prompt Builder
      │
      ▼
LLM
      │
      ▼
Answer
```

Supported generators include:

- Ollama
- Custom generators
- Any implementation of `GeneratorInterface`

---

# Data Flow

```text
Documents
      │
      ▼
Documents[]
      │
      ▼
Chunks[]
      │
      ▼
Embeddings[]
      │
      ▼
Vector Index
      │
      ▼
Retrieved Chunks
      │
      ▼
Optimized Chunks
      │
      ▼
Generated Answer
```

Every module consumes the output of the previous stage.

---

# Design Principles

## Modular

Each module can be used independently.

Example:

```text
Loader

or

Chunker

or

Retriever
```

without requiring the rest of the framework.

---

## Extensible

Every major component is built around a lightweight interface.

You can implement your own:

- Loader
- Embedding Model
- Vector Database
- Retriever
- Generator

without modifying MetaRAG itself.

---

## Composable

Components are designed to work together through simple inputs and outputs.

```text
Any Loader
      │
Any Chunker
      │
Any Embedding Model
      │
Any Vector Database
      │
Any Retriever
      │
Any Generator
```

No component is tightly coupled to a specific implementation.

---

# Typical Production Pipeline

```text
                 PDFs
                  │
                  ▼
        Recursive Chunking
                  │
                  ▼
      SentenceTransformer Embeddings
                  │
                  ▼
              FAISS Index
                  │
                  ▼
          Hybrid Retrieval
                  │
                  ▼
            Full Pipeline
                  │
                  ▼
          Ollama / Cloud LLM
                  │
                  ▼
             Generated Answer
```

---

# Why This Architecture?

Traditional RAG systems often bundle loading, retrieval, and generation into a single implementation, making experimentation difficult.

MetaRAG separates these concerns into interchangeable modules, allowing developers to evaluate different strategies independently while keeping the rest of the workflow unchanged.

---

## Next Step

Continue to **Examples** to see each component in action through runnable demonstrations included with MetaRAG.