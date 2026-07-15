# Quick Start

This guide demonstrates how to build a complete Retrieval-Augmented Generation (RAG) pipeline using MetaRAG.

By the end, you will:

- Load documents
- Chunk them
- Create embeddings
- Build a vector database
- Retrieve relevant context
- Generate an answer using an LLM

---

# Complete Workflow

```text
Documents
    │
    ▼
DocumentLoader
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

---

# Step 1 — Load Documents

```python
from pathlib import Path
from metarag import DocumentLoader

loader = DocumentLoader(Path("documents"))

documents = loader.load()

print(len(documents))
```

Example Output

```text
101
```

---

# Step 2 — Chunk Documents

```python
from metarag import Chunker

chunker = Chunker(strategy="recursive")

chunks = chunker.chunk_documents(documents)

print(len(chunks))
```

Example Output

```text
322
```

---

# Step 3 — Generate Embeddings

```python
from metarag import CachedEmbeddings

embedder = CachedEmbeddings(...)

embeddings = embedder.embed_documents(
    [chunk.text for chunk in chunks]
)
```

Each chunk is converted into a dense numerical vector.

---

# Step 4 — Build a Vector Database

```python
from metarag import InMemoryVectorDB

vector_db = InMemoryVectorDB()

vector_db.build(
    chunks,
    embeddings
)
```

The vector database stores every chunk alongside its embedding for efficient similarity search.

---

# Step 5 — Create a Retriever

```python
from metarag import DenseRetriever

retriever = DenseRetriever(
    chunks,
    embedder,
    vector_db
)
```

Available retrievers:

| Retriever | Description |
|------------|-------------|
| Dense | Embedding similarity |
| BM25 | Keyword search |
| Hybrid | Dense + BM25 |
| MMR | Diversity-aware retrieval |

---

# Step 6 — Choose a Pipeline

```python
from metarag.pipelines import StraightPipeline

pipeline = StraightPipeline(retriever)

result = pipeline.run(
    "What is the main topic of this document?",
    k=3
)
```

Example Output

```text
Pipeline : straight

Retrieved Chunks : 3
```

Other available pipelines:

- Straight
- MultiQuery
- HyDE
- Reranked
- Full

---

# Step 7 — Generate the Final Answer

```python
from metarag.pipeline.generator import (
    OllamaGenerator,
    GeneratorWrapper,
)

generator = GeneratorWrapper(
    OllamaGenerator(model="mistral")
)

answer, latency = generator.generate_text(
    query=result["query"],
    chunks=result["chunks"]
)

print(answer)
```

Example Output

```text
The document primarily discusses the ethical implications
of Generative Artificial Intelligence across multiple
domains including healthcare, education and law.
```

---

# Complete Pipeline

```text
PDFs
 │
 ▼
Load Documents
 │
 ▼
Chunk Documents
 │
 ▼
Generate Embeddings
 │
 ▼
Build Vector Database
 │
 ▼
Retrieve Top-k Chunks
 │
 ▼
Pipeline Processing
 │
 ▼
LLM Generation
 │
 ▼
Final Answer
```

---

# Running the Example

A complete implementation is included with MetaRAG.

```bash
python examples/full_rag_demo.py
```

Individual module demonstrations are also available.

```text
loader_demo.py

chunker_demo.py

embeddings_demo.py

vectordb_demo.py

retriever_demo.py

pipeline_demo.py
```

---

# Recommended Pipeline

For most applications, the following configuration provides a good balance between retrieval quality and speed.

```text
Recursive Chunking
        │
        ▼
SentenceTransformer Embeddings
        │
        ▼
InMemoryVectorDB / FAISS
        │
        ▼
Hybrid Retriever
        │
        ▼
Full Pipeline
        │
        ▼
Ollama Generator
```

---

# What's Next?

Continue to **Architecture** to understand how every MetaRAG component interacts internally and how to customize the pipeline for your own applications.