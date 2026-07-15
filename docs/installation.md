# Installation

This guide walks you through installing **MetaRAG**, optional dependencies, and verifying that your environment is ready.

---

# Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| pip | Latest Recommended |
| Operating System | Windows, Linux, macOS |

---

# Install MetaRAG

Install the base package.

```bash
pip install metarag
```

The base installation contains the core RAG components:

- Document Loader
- Chunker
- Embeddings Interface
- Vector Database Interface
- Retrievers
- Pipelines
- Generator Interface

---

# Optional Dependencies

Install only the components you need.

## PDF Support

Required for loading PDF documents.

```bash
pip install metarag[pdf]
```

Equivalent package:

```bash
pip install pypdf
```

---

## FAISS Vector Store

For high-performance similarity search.

```bash
pip install metarag[faiss]
```

Equivalent package:

```bash
pip install faiss-cpu
```

---

## Chroma Vector Store

Persistent local vector database.

```bash
pip install metarag[chroma]
```

Equivalent package:

```bash
pip install chromadb
```

---

## Ollama Support

For local LLM inference.

```bash
pip install metarag[ollama]
```

Then install any Ollama model.

Example:

```bash
ollama pull mistral
```

or

```bash
ollama pull llama3
```

Start the Ollama server.

```bash
ollama serve
```

---

## Sentence Transformers

Recommended for local embeddings.

```bash
pip install sentence-transformers
```

Popular models include:

- all-MiniLM-L6-v2
- all-mpnet-base-v2
- bge-base-en-v1.5

---

# Development Installation

Clone the repository.

```bash
git clone <repository-url>

cd MetaRAG
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install MetaRAG in editable mode.

```bash
pip install -e .
```

Install development dependencies.

```bash
pip install pytest
```

---

# Verify Installation

Run the following command.

```bash
python -c "import metarag; print('MetaRAG installed successfully!')"
```

Expected output:

```text
MetaRAG installed successfully!
```

---

# Run Examples

Navigate to the project directory.

```bash
cd examples
```

Example:

```bash
python loader_demo.py
```

Other available demos:

```text
loader_demo.py

chunker_demo.py

embeddings_demo.py

vectordb_demo.py

retriever_demo.py

pipeline_demo.py

full_rag_demo.py
```

---

# Run Tests

Execute the complete test suite.

```bash
pytest
```

Or test an individual module.

```bash
pytest tests/test_loader.py

pytest tests/test_chunker.py

pytest tests/test_embeddings.py

pytest tests/test_vectordb.py

pytest tests/test_retriever.py

pytest tests/test_pipelines.py

pytest tests/test_generator.py
```

---

# Installation Overview

```text
                Install MetaRAG
                      │
                      ▼
             Optional Dependencies
      ┌──────────┬──────────┬──────────┐
      ▼          ▼          ▼          ▼
    PDF       FAISS      Chroma     Ollama
      │          │          │          │
      └──────────┴──────────┴──────────┘
                      │
                      ▼
             Verify Installation
                      │
                      ▼
                Run Examples
                      │
                      ▼
                 Run Tests
```

---

# Troubleshooting

### PDFs are skipped

Install PDF support.

```bash
pip install metarag[pdf]
```

---

### Ollama connection failed

Ensure the server is running.

```bash
ollama serve
```

Verify installed models.

```bash
ollama list
```

---

### FAISS import error

Install the FAISS dependency.

```bash
pip install faiss-cpu
```

---

### Chroma import error

Install ChromaDB.

```bash
pip install chromadb
```

---

## Next Step

➡ Continue with **Quick Start** to build your first Retrieval-Augmented Generation pipeline using MetaRAG.