"""
metarag/evaluator/ — Evaluation Framework

Score RAG pipelines on 5 metrics: faithfulness, relevancy, precision, coverage, redundancy.

Exports:
- Evaluator: orchestrates 5 metrics, returns ScoreResult
- Scorer: composite scoring with 3 presets (balanced, precision, recall)
- ScoreResult: dataclass with all scores

Scoring Presets:
- balanced: (faith + relev + prec + cov - redund) / 5
- precision: prioritize precision + faithfulness
- recall: prioritize coverage + relevancy
"""

from .evaluator import Evaluator
from .scorer import Scorer, ScoreResult

__all__ = [
    "Evaluator",
    "Scorer",
    "ScoreResult",
]