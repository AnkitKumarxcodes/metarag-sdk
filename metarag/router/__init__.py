"""
metarag/router/ — Routing & Profiling

Smart routing based on corpus + query + retrieval quality signals.

Exports:
- LearnedRuleRouter: learn thresholds from benchmark data (v0.2)
- Router: rule-based selector (fallback)
- CorpusProfiler: profile corpus statistics
- QueryProfiler: profile query characteristics
- ProbeProfiler: profile vector DB retrieval quality

The three profilers extract 16 features that feed into routing decisions.
LearnedRuleRouter learns optimal thresholds from your benchmark data.
"""

from .learned_rule_router import LearnedRuleRouter
from .selector import Router
from .corpus_profiler import CorpusProfiler
from .query_profiler import QueryProfiler
from .probe_profiler import ProbeProfiler

__all__ = [
    "LearnedRuleRouter",
    "Router",
    "CorpusProfiler",
    "QueryProfiler",
    "ProbeProfiler",
]