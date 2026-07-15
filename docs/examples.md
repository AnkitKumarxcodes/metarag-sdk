# Examples

MetaRAG includes a collection of runnable examples that demonstrate each module independently before combining them into a complete Retrieval-Augmented Generation (RAG) pipeline.

All examples can be found in the `examples/` directory.

```text
examples/
│
├── loader_demo.py
├── chunker_demo.py
├── embeddings_demo.py
├── vectordb_demo.py
├── retriever_demo.py
├── pipeline_demo.py
└── full_rag_demo.py
```

---

# Learning Path

The examples are designed to be explored in sequence.

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

Each example introduces one new module while building upon the previous ones.

---

# loader_demo.py

Demonstrates the document loading API.

Topics covered:

- Loading documents from a directory
- Viewing extracted documents
- Loading reports
- Loaded vs skipped files
- File type summaries

Run:

```bash
python examples/loader_demo.py
```

Example Output

```text
Documents Loaded : 101

Files Loaded : 8

Files Skipped : 0

Loaded Summary
Skipped Summary
```

---

# chunker_demo.py

Demonstrates every available chunking strategy.

Topics covered:

- Fixed Chunking
- Recursive Chunking
- Semantic Chunking
- Sentence Chunking
- Sliding Window
- Markdown Chunking

Run:

```bash
python examples/chunker_demo.py
```

Example Output

```text
Strategy : Recursive

Chunks Generated : 322

First Chunk

Metadata

Preview
```

---

# embeddings_demo.py

Demonstrates embedding generation and caching.

Topics covered:

- Single query embeddings
- Batch embeddings
- Disk caching
- Cache persistence
- Cache reuse

Run:

```bash
python examples/embeddings_demo.py
```

Example Output

```text
Embedding Dimension : 384

Documents Embedded : 322

Cache Files : 322
```

---

# vectordb_demo.py

Demonstrates vector indexing and similarity search.

Topics covered:

- Building a vector index
- Similarity search
- Top-k retrieval
- Supported vector stores

Run:

```bash
python examples/vectordb_demo.py
```

Example Output

```text
Built Index : 322 vectors

Top 3 Results

Similarity Scores
```

---

# retriever_demo.py

Compare retrieval algorithms on the same query.

Topics covered:

- Dense Retrieval
- BM25 Retrieval
- Hybrid Retrieval
- Maximum Marginal Relevance (MMR)

Run:

```bash
python examples/retriever_demo.py
```

Example Output

```text
BM25

Dense

Hybrid

MMR

Top 3 Retrieved Chunks
```

---

# pipeline_demo.py

Demonstrates retrieval optimization pipelines.

Topics covered:

- Straight Pipeline
- MultiQuery
- HyDE
- Cross-Encoder Reranking
- Full Pipeline

Run:

```bash
python examples/pipeline_demo.py
```

Example Output

```text
Straight Pipeline

↓

Retrieved Chunks

MultiQuery

↓

Expanded Queries

↓

Deduplicated Results

HyDE

↓

Generated Hypothesis

↓

Retrieved Chunks

Reranked

↓

Cross-Encoder Scores

Full

↓

Retrieve

↓

Rerank

↓

Deduplicate
```

---

# full_rag_demo.py

The complete MetaRAG workflow.

This example combines every module into an end-to-end Retrieval-Augmented Generation system.

Workflow

```text
Documents
      │
      ▼
Loader
      │
      ▼
Chunker
      │
      ▼
Embeddings
      │
      ▼
Vector Database
      │
      ▼
Retriever
      │
      ▼
Pipeline
      │
      ▼
Generator
      │
      ▼
Answer
```

Run:

```bash
python examples/full_rag_demo.py
```

Example Output

```text
Question

↓

Retrieved Context

↓

Generated Answer

↓

Latency Statistics
```

---

# Which Example Should I Run?

| If you want to... | Run |
|-------------------|-----|
| Load documents | `loader_demo.py` |
| Compare chunking strategies | `chunker_demo.py` |
| Generate embeddings | `embeddings_demo.py` |
| Build a vector database | `vectordb_demo.py` |
| Compare retrieval methods | `retriever_demo.py` |
| Compare retrieval pipelines | `pipeline_demo.py` |
| Build a complete RAG application | `full_rag_demo.py` |

---

# Recommended Order

If you're using MetaRAG for the first time, follow this sequence.

```text
1. loader_demo.py

        ↓

2. chunker_demo.py

        ↓

3. embeddings_demo.py

        ↓

4. vectordb_demo.py

        ↓

5. retriever_demo.py

        ↓

6. pipeline_demo.py

        ↓

7. full_rag_demo.py
```

Each example introduces one additional concept, making it easier to understand how the complete system is assembled.

---

## Next Step

Continue to **API Reference** for detailed documentation of MetaRAG's public classes, methods, and interfaces.