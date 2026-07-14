"""
metarag/defaults.py

Single, flat, mutable source of truth for MetaRAG's hyperparameters.
No model names, no config unrelated to tuning — just the values that
affect retrieval/chunking/scoring behavior.

Every file reads DEFAULTS.<field> at the point of use (never cached at
import time), so any change here — by the user, the router, or a
benchmark sweep — is immediately visible everywhere.

Each field accepts EITHER a single value OR a list of values:
    DEFAULTS.hybrid_alpha = 0.5          # single value — used as-is
    DEFAULTS.hybrid_alpha = [0.3, 0.5, 0.7]   # list — benchmark() sweeps over it

Use as_list() to normalize either case into a list for iteration:
    for alpha in DEFAULTS.as_list("hybrid_alpha"):
        ...
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Union, List


Number = Union[int, float]


@dataclass
class MetaRAGDefaults:
    chunk_size: Union[int, List[int]] = 500              # range: 200-1500
    chunk_overlap: Union[int, List[int]] = 50             # range: 0-150
    chunk_strategy: Union[str, List[str]] = "recursive"    # fixed|recursive|semantic|sentence|sliding_window|markdown

    k: Union[int, List[int]] = 4                          # range: 2-10
    hybrid_alpha: Union[float, List[float]] = 0.5          # range: 0.0 (BM25) - 1.0 (dense)
    mmr_lambda: Union[float, List[float]] = 0.6            # range: 0.0 (diversity) - 1.0 (relevance)

    reranker_top_k: Union[int, List[int]] = 5              # range: 2-10
    multiquery_n_variants: Union[int, List[int]] = 2       # range: 1-5

    dedup_threshold: Union[float, List[float]] = 0.85      # range: 0.5-0.95

    eval_preset: Union[str, List[str]] = "balanced"         # balanced|precision|recall

    min_win_rate_for_rule_override: Union[float, List[float]] = 0.05   # range: 0.0-0.2

    def as_list(self, field_name: str) -> list:
        """Normalize a field's current value into a list, whether it's a
        single value or already a list. Use this in benchmark()/sweep loops."""
        value = getattr(self, field_name)
        return value if isinstance(value, list) else [value]

    def as_single(self, field_name: str):
        """Get a single usable value from a field — if it's a list, returns
        the first element. Use this in normal (non-sweeping) construction code."""
        value = getattr(self, field_name)
        return value[0] if isinstance(value, list) else value


DEFAULTS = MetaRAGDefaults()