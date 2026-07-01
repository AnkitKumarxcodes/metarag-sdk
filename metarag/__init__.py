"""
MetaRAG v0.2 — AutoML for RAG

Intelligent Pipeline Selection Engine for domain-optimized RAG systems.

The main MetaRAG class orchestrates benchmarking, routing, and evaluation.
Users provide docs + LLM. MetaRAG learns which pipelines work best.

Usage:
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from metarag import MetaRAG

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="mistral", temperature=0)

    rag = MetaRAG(docs="./docs", embeddings=embeddings, llm=llm)
    rag.fit()
    rag.benchmark(num_queries=100)
    answer = rag.ask("What is the policy?")
    print(answer)
"""

from .metarag import MetaRAG, Answer

__version__ = "0.2.0"
__author__ = "Ankit Kumar"

__all__ = [
    "MetaRAG",
    "Answer",
]