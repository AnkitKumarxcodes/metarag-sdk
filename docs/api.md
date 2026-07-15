# API Reference

This page provides a quick reference for MetaRAG's public classes and interfaces.

For detailed examples, see the **Examples** section.

---

# Package Overview

```text
metarag
│
├── DocumentLoader
├── Chunker
├── CachedEmbeddings
│
├── InMemoryVectorDB
├── FAISSVectorDB
├── ChromaVectorDB
│
├── DenseRetriever
├── BM25Retriever
├── HybridRetriever
├── MMRRetriever
│
├── StraightPipeline
├── MultiQueryPipeline
├── HyDEPipeline
├── RerankedPipeline
├── FullPipeline
│
├── GeneratorWrapper
└── OllamaGenerator
```

---

# DocumentLoader

Loads documents from a directory and extracts their content.

```python
loader = DocumentLoader("documents")

documents = loader.load()
```

## Constructor

```python
DocumentLoader(
    directory
)
```

## Methods

| Method | Description |
|----------|-------------|
| `load()` | Load all supported documents |

## Returns

```text
DocumentList
```

---

# Chunker

Splits documents into retrieval chunks.

```python
chunker = Chunker(
    strategy="recursive"
)

chunks = chunker.chunk_documents(documents)
```

## Constructor

```python
Chunker(
    strategy="recursive",
    chunk_size=500,
    overlap=50
)
```

## Methods

| Method | Description |
|----------|-------------|
| `chunk_documents()` | Chunk loaded documents |

## Strategies

```text
fixed

recursive

semantic

sentence

sliding_window

markdown
```

---

# CachedEmbeddings

Wraps any embedding model and automatically caches generated vectors.

```python
embedder = CachedEmbeddings(model)

vectors = embedder.embed_documents(texts)
```

## Methods

| Method | Description |
|----------|-------------|
| `embed()` | Embed a single text |
| `embed_query()` | Embed a query |
| `embed_documents()` | Batch embedding |

---

# Vector Databases

Stores embeddings for similarity search.

Available implementations:

```text
InMemoryVectorDB

FAISSVectorDB

ChromaVectorDB
```

Typical usage:

```python
db.build(
    chunks,
    embeddings
)
```

## Common Methods

| Method | Description |
|----------|-------------|
| `build()` | Build the vector index |
| `search()` | Similarity search |

---

# Retrievers

Retrieve the most relevant chunks for a query.

Available implementations.

```text
DenseRetriever

BM25Retriever

HybridRetriever

MMRRetriever
```

Typical usage.

```python
results = retriever.retrieve(
    query,
    k=5
)
```

## Methods

| Method | Description |
|----------|-------------|
| `retrieve()` | Retrieve top-k relevant chunks |

---

# Pipelines

Retrieval pipelines improve retrieval quality before generation.

Available pipelines.

```text
StraightPipeline

MultiQueryPipeline

HyDEPipeline

RerankedPipeline

FullPipeline
```

Typical usage.

```python
pipeline = StraightPipeline(
    retriever
)

result = pipeline.run(
    query,
    k=3
)
```

## Methods

| Method | Description |
|----------|-------------|
| `run()` | Execute the retrieval pipeline |

## Returned Object

```python
{
    "query": "...",
    "chunks": [...],
    "pipeline": "...",
    "hypothesis": ...
}
```

---

# GeneratorWrapper

Generates answers from retrieved context.

```python
generator = GeneratorWrapper(
    OllamaGenerator()
)

answer, latency = generator.generate_text(
    query,
    chunks
)
```

## Methods

| Method | Description |
|----------|-------------|
| `generate_text()` | Generate the final answer |

Returns

```python
(text, latency_ms)
```

---

# OllamaGenerator

Built-in generator for local Ollama models.

```python
generator = OllamaGenerator(
    model="mistral"
)
```

## Constructor

```python
OllamaGenerator(
    model="mistral",
    base_url="http://localhost:11434"
)
```

Requires:

```bash
ollama serve
```

---

# Core Interfaces

MetaRAG is designed around lightweight interfaces.

## EmbeddingInterface

Implement your own embedding model.

Required methods.

```python
embed_query()

embed_documents()
```

---

## GeneratorInterface

Implement your own language model.

Required method.

```python
generate(prompt)
```

---

# Typical Workflow

```text
DocumentLoader
        │
        ▼
Chunker
        │
        ▼
CachedEmbeddings
        │
        ▼
VectorDB
        │
        ▼
Retriever
        │
        ▼
Pipeline
        │
        ▼
GeneratorWrapper
        │
        ▼
Answer
```

---

# Public API Summary

| Module | Primary Class |
|---------|---------------|
| Loader | `DocumentLoader` |
| Chunking | `Chunker` |
| Embeddings | `CachedEmbeddings` |
| Vector Store | `InMemoryVectorDB`, `FAISSVectorDB`, `ChromaVectorDB` |
| Retrieval | `DenseRetriever`, `BM25Retriever`, `HybridRetriever`, `MMRRetriever` |
| Pipelines | `StraightPipeline`, `MultiQueryPipeline`, `HyDEPipeline`, `RerankedPipeline`, `FullPipeline` |
| Generation | `GeneratorWrapper`, `OllamaGenerator` |

---

# Extension Points

MetaRAG is designed to be extended.

You can implement custom:

```text
✓ Document Loaders

✓ Chunking Strategies

✓ Embedding Models

✓ Vector Databases

✓ Retrieval Algorithms

✓ Retrieval Pipelines

✓ Language Models
```

As long as the required interface is implemented, MetaRAG components remain interchangeable.

---

## Need More Help?

See:

- **Installation** — Environment setup
- **Quick Start** — Build your first RAG pipeline
- **Architecture** — Internal workflow
- **Examples** — Runnable demonstrations