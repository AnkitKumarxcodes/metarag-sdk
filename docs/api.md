# API Reference

This page documents the stable public API exposed by MetaRAG.

For lower-level building blocks (retrievers, vector databases, chunkers, etc.), see the corresponding component documentation.

---

# High-Level API

```python
from metarag import MetaRAG
```

`MetaRAG` is the primary interface for building, benchmarking and experimenting with Retrieval-Augmented Generation (RAG) systems.

---

## Constructor

```python
MetaRAG(
    docs,
    embeddings,
    generator,
    ...
)
```

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `docs` | Document directory or collection |
| `embeddings` | Embedding model or embedding interface |
| `generator` | Language model used for answer generation |

---

# Core Methods

## fit()

Builds the retrieval system.

```python
rag.fit()
```

Typical tasks include:

- loading documents
- chunk generation
- embedding generation
- vector index construction
- pipeline initialization

---

## ask()

Retrieves relevant context and generates an answer.

```python
answer = rag.ask(
    "What is Retrieval-Augmented Generation?"
)
```

Returns

```python
Answer
```

---

## benchmark()

Evaluate every configured pipeline using the same set of queries.

```python
df = rag.benchmark(
    queries,
    retrieval_only=True
)
```

Returns

```python
pandas.DataFrame
```

Also saves

```text
benchmark.csv
```

when enabled.

---

## leaderboard()

Display pipeline rankings.

```python
rag.leaderboard()
```

---

## dashboard()

Display a summary of benchmark statistics.

```python
rag.dashboard()
```

---

## report()

Generate a benchmark summary.

```python
rag.report()
```

---

## inspect()

Inspect retrieved chunks for a query.

```python
rag.inspect(
    query,
    k=3
)
```

---

## trace()

Trace a query through a specific pipeline.

```python
rag.trace(
    query,
    pipeline_name="full"
)
```

---

## explain()

Explain routing or retrieval decisions.

```python
rag.explain(query)
```

---

## analyze_query()

Compute descriptive statistics for a query.

```python
rag.analyze_query(query)
```

---

## analyze_corpus()

Analyze the indexed corpus.

```python
rag.analyze_corpus()
```

---

## save()

Save the current project.

```python
rag.save()
```

Typical output

```text
config.json

benchmark.csv

router_thresholds.json
```

---

## load()

Load a previously saved project.

```python
rag.load(path)
```

---

# Configuration

Global defaults can be modified before constructing new components.

```python
from metarag import DEFAULTS

DEFAULTS.chunk_size = 500

DEFAULTS.k = 4

DEFAULTS.hybrid_alpha = 0.5
```

These values are used when explicit parameters are not provided.

---

# Toolkit API

MetaRAG components may also be used independently.

```python
from metarag import DocumentLoader

from metarag import Chunker

from metarag import HybridRetriever

from metarag import InMemoryVectorDB

from metarag import Evaluator
```

This allows developers to build custom retrieval pipelines while reusing individual components.

---

# Public Modules

| Module | Purpose |
|---------|---------|
| `core` | Loading, chunking, embeddings, retrieval |
| `pipelines` | Retrieval pipelines and generators |
| `Evaluator` | Evaluation and scoring |
| `router` | Query profiling and routing |
| `defaults` | Global framework configuration |

---

# Version

This documentation describes the public API available in **MetaRAG v0.3.0**.

Interfaces may evolve before the first stable `1.0` release.