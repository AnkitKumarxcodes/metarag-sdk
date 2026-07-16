# Examples

MetaRAG includes small, focused examples that demonstrate each component independently before combining everything into a complete RAG workflow.

Run any example with:

```bash
python examples/<example_name>.py
```

---

## Document Loader

```bash
python examples/loader_demo.py
```

Loads documents from a directory and reports what was successfully processed.

Example output

```text
DocumentLoader Report
------------------------------

Files Loaded : 8
Files Skipped: 0

Documents Extracted : 101
```

---

## Chunking

```bash
python examples/chunker_demo.py
```

Splits documents into chunks using the selected chunking strategy.

Example output

```text
Chunker Report
------------------------------

Strategy          : recursive
Chunk Size        : 500
Chunk Overlap     : 50

Chunks Generated  : 333
```

---

## Embeddings

```bash
python examples/embeddings_demo.py
```

Generates embeddings and demonstrates the caching layer.

Example output

```text
Generating embeddings...

[Embeddings] 333/333

Cached embeddings saved.
```

---

## Vector Database

```bash
python examples/vector_db_demo.py
```

Builds a vector index and performs similarity search.

Example output

```text
Building vector index...

Indexed Chunks : 333

Top-3 Results

1. ...
2. ...
3. ...
```

---

## Retrieval

```bash
python examples/retriever_demo.py
```

Compare different retrieval strategies.

Included retrievers

- BM25
- Dense
- Hybrid
- MMR

Example output

```text
Retriever : Hybrid

Retrieved Chunks : 3

Latency : 2 ms
```

---

## Pipelines

```bash
python examples/pipeline_demo.py
```

Runs every built-in retrieval pipeline.

Included pipelines

- Straight
- MultiQuery
- HyDE
- Reranked
- Full

Example output

```text
Pipeline : Full

Retrieved Chunks : 3

Pipeline completed.
```

---

# Complete Framework Demo

```bash
python examples/metarag_demo.py
```

Demonstrates the complete MetaRAG workflow.

```text
fit()
   ↓
ask()
   ↓
benchmark()
   ↓
leaderboard()
   ↓
dashboard()
   ↓
report()
   ↓
save()
```

Typical output

```text
Files Loaded        : 8

Documents Extracted : 101

Chunks Generated    : 333

Pipelines Built     : 7

Benchmark Rows      : 595

Router Thresholds Saved

Benchmark CSV Saved
```

The demo also generates project artifacts such as:

```text
benchmark.csv

config.json

router_thresholds.json
```

---

## Suggested Order

If you're new to MetaRAG, explore the examples in this order.

```text
loader
   ↓
chunker
   ↓
embeddings
   ↓
vector database
   ↓
retriever
   ↓
pipeline
   ↓
metarag
```

Each example introduces one new concept while reusing the previous components.

---

## Experiment

After running the demos, try changing:

- chunk size
- retrieval strategy
- pipeline
- embedding model
- generator

Then compare the benchmark results to see how different configurations affect retrieval performance.