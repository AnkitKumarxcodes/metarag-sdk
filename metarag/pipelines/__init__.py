"""
metarag/pipelines/ — RAG Pipeline Variants

7 composable pipeline variants combining retrieval, augmentation, and generation.

Exports:
- Pipeline: base composable pipeline (retriever + augmenters)
- MultiQuery: expand query into N variants
- HyDE: hypothetical document embedding
- Reranker: cross-encoder reranking
- GeneratorWrapper: LLM-based answer generation
- Answer: generated answer with metadata

Pipeline Variants:
1. straight: BM25 only
2. dense: Dense embeddings only
3. hybrid: BM25 + Dense combined
4. reranked: Dense + Cross-encoder reranking
5. mmr: Max marginal relevance (diversity)
6. multiquery: Hybrid + Query expansion
7. hyde: Dense + Hypothetical document embeddings
"""

from .pipeline import Pipeline, MultiQuery, HyDE, Reranker
from .generator import GeneratorWrapper, Answer

__all__ = [
    "Pipeline",
    "MultiQuery",
    "HyDE",
    "Reranker",
    "GeneratorWrapper",
    "Answer",
]