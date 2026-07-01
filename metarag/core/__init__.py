"""
metarag/core/ — Core RAG Components

Low-level building blocks for document loading, chunking, embedding, and retrieval.

Exports:
- DocumentLoader: load docs from multiple formats (PDF, TXT, HTML, DOCX, CSV, JSON, URLs)
- Chunker: split documents into chunks (6 strategies)
- CachedEmbeddings: embedding with numpy caching
- VectorDB: vector database abstraction (Chroma, FAISS)
- get_retriever: retriever factory (dense, sparse, hybrid, mmr)
"""

from .loader import DocumentLoader
from .chunking import Chunker
from .embeddings import CachedEmbeddings
from .vector_db import VectorDB
from .retriever import get_retriever

__all__ = [
    "DocumentLoader",
    "Chunker",
    "CachedEmbeddings",
    "VectorDB",
    "get_retriever",
]