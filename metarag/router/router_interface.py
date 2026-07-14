# router/router_interface.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


class RouterInterface(ABC):
    """
    Contract for any router pluggable into MetaRAG.

    MetaRAG only ever calls .route(features) — it never inspects HOW a
    router makes its decision. This is what makes rag.set_router(...) work
    with the built-in Router, a hand-rolled rule engine, an sklearn model
    (via SklearnRouterAdapter), or an LLM making the call in plain English.

    features is a dict merging three signal sources, all computed by
    MetaRAG._extract_query_features():
        - CorpusProfiler output  (corpus-wide: num_docs, ocr_ratio, ...)
        - QueryProfiler output   (per-query: query_length, is_short, ...)
        - ProbeProfiler output   (per-query retrieval difficulty: max_similarity, ...)

    Minimal example:
        class MyRouter(RouterInterface):
            def route(self, features: dict) -> str:
                return "hybrid" if features.get("max_similarity", 0) > 0.7 else "multiquery"
    """

    @abstractmethod
    def route(self, features: Dict[str, Any]) -> str:
        """
        Args:
            features: dict of corpus/query/probe signals

        Returns:
            pipeline name — must match a key in MetaRAG._pipelines
        """
        raise NotImplementedError

    def explain(self, pipeline: str, features: Dict[str, Any] = None) -> str:
        """
        Optional. Human-readable justification for a routing decision.
        MetaRAG calls this opportunistically (via hasattr check) — routers
        that don't implement it just won't show an explanation, nothing breaks.
        """
        return f"{pipeline}: selected by {self.__class__.__name__}"