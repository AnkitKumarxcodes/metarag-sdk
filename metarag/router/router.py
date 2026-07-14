# router/router.py

"""
MetaRAG's built-in router. Replaces the old selector.py + learned_rule_router.py
split — one class, two modes, no divergent logic to keep in sync.

Cold start (untrained):
    Rule-based selection using the raw corpus/query/probe signals directly.
    Works from the moment fit() completes, before any benchmark() has run.

Trained:
    After benchmark() -> train(), routing is win_rate-driven: the pipeline
    that actually won the most benchmark queries is the default, and a
    small set of refinement rules can only override that default toward a
    DIFFERENT pipeline if that pipeline has real supporting evidence
    (win_rate above DEFAULTS.min_win_rate_for_rule_override) AND the
    override threshold itself is LEARNED from that pipeline's own winning
    queries — never a hardcoded guess.

This class is the default. It is NOT the only option:
    rag.set_router(router)                          — any RouterInterface object
    rag.set_router_from_model(model, feature_cols)   — any sklearn-style .predict() model
    router.load(path="...")                          — load thresholds trained elsewhere
                                                         (your own project, a teammate's,
                                                         or hand-written by a user)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Any, List

import pandas as pd

from .router_interface import RouterInterface
from ..defaults import DEFAULTS


class Router(RouterInterface):
    """
    MetaRAG's reference router. See module docstring for the two modes.

    Args:
        base_path: this project's storage directory (MetaRAG._base), used
                    as the default location for save()/load(). Optional —
                    a router can be trained/used purely in-memory without
                    ever touching disk, or pointed at a different project's
                    saved thresholds via load(path=...).
    """

    # ── Cold-start rule thresholds (used only before train() has run) ──
    # These are routing-decision constants, not tunable model hyperparameters,
    # so they live here rather than in defaults.py — but they're plain class
    # attributes specifically so a subclass or an instance can override them
    # without touching this file.
    COLD_START_HIGH_CONFIDENCE_SIMILARITY = 0.85
    COLD_START_LOW_REDUNDANCY = 0.3
    COLD_START_NUMERIC_HEAVY_RATIO = 0.4
    COLD_START_SHORT_DOC_RATIO = 0.6
    COLD_START_HARD_RETRIEVAL_SIMILARITY = 0.5
    COLD_START_HARD_RETRIEVAL_AVG_SIMILARITY = 0.35
    COLD_START_HIGH_REDUNDANCY = 0.6
    COLD_START_OCR_RATIO = 0.3

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else None
        self.thresholds: Dict[str, dict] = {}
        self.rules: List[tuple] = []
        self.is_trained = False

    # ═════════════════════════════════════════════════════════
    # TRAINING
    # ═════════════════════════════════════════════════════════

    def train(self, benchmark_df: pd.DataFrame) -> None:
        """
        Learn per-pipeline win rates + winning-query feature medians from
        a benchmark_df produced by MetaRAG.benchmark(). Switches this
        router from cold-start rules into learned-threshold mode.
        """
        if len(benchmark_df) == 0:
            print("[Router] Empty benchmark, skipping training")
            return

        if "winning_pipeline" not in benchmark_df.columns:
            print("[Router] ⚠️  No 'winning_pipeline' column — "
                  "benchmark_df must include per-query winners. Skipping training.")
            return

        total_queries = benchmark_df["query"].nunique()
        feature_cols = ["max_similarity", "avg_similarity", "redundancy", "query_length", "num_docs"]
        available_cols = [c for c in feature_cols if c in benchmark_df.columns]

        for pipeline in benchmark_df["pipeline"].unique():
            pipeline_rows = benchmark_df[benchmark_df["pipeline"] == pipeline]
            wins = benchmark_df[
                (benchmark_df["pipeline"] == pipeline)
                & (benchmark_df["winning_pipeline"] == pipeline)
            ]
            win_rate = wins["query"].nunique() / total_queries if total_queries else 0.0

            stats = {
                "win_rate": round(win_rate, 4),
                "avg_composite": float(pipeline_rows["composite"].mean()) if "composite" in pipeline_rows else 0.0,
            }

            source = wins if len(wins) > 0 else pipeline_rows
            for col in available_cols:
                stats[col] = float(source[col].median()) if col in source else 0.0

            self.thresholds[pipeline] = stats

        self.rules = sorted(self.thresholds.items(), key=lambda x: x[1]["win_rate"], reverse=True)
        self.is_trained = True

        if self.base_path:
            self.save()

        print(f"[Router] Trained on {total_queries} queries, {len(benchmark_df)} rows")
        print(f"[Router] Learned {len(self.thresholds)} pipelines")
        for pipeline, stats in self.rules[:3]:
            print(f"  {pipeline}: {stats['win_rate']:.1%} win rate, avg score {stats['avg_composite']:.2f}")

    # ═════════════════════════════════════════════════════════
    # ROUTING
    # ═════════════════════════════════════════════════════════

    def route(self, features: Dict[str, Any]) -> str:
        """Dispatch to learned-mode or cold-start-mode routing."""
        if self.is_trained and self.rules:
            return self._route_learned(features)
        return self._route_cold_start(features)

    def _route_learned(self, features: Dict[str, Any]) -> str:
        """
        Win_rate-driven routing. See module docstring for the reasoning —
        the benchmark's actual winner is the default; refinement rules can
        only override it toward a pipeline with real supporting evidence,
        using thresholds learned from that pipeline's own winning queries.
        """
        default_pipeline = self.rules[0][0]

        max_sim = features.get("max_similarity", 0.0)
        avg_sim = features.get("avg_similarity", 0.0)
        redundancy = features.get("redundancy", 0.0)
        query_len = features.get("query_length", 0)

        min_win_rate = DEFAULTS.as_single("min_win_rate_for_rule_override")

        def has_evidence(pipeline_name: str) -> bool:
            return (
                pipeline_name in self.thresholds
                and self.thresholds[pipeline_name]["win_rate"] >= min_win_rate
            )

        def learned(pipeline_name: str, key: str, fallback: float) -> float:
            return self.thresholds.get(pipeline_name, {}).get(key, fallback)

        if redundancy > learned("mmr", "redundancy", self.COLD_START_HIGH_REDUNDANCY) and has_evidence("mmr"):
            return "mmr"

        mq_max_sim = learned("multiquery", "max_similarity", 1 - self.COLD_START_HARD_RETRIEVAL_SIMILARITY)
        mq_avg_sim = learned("multiquery", "avg_similarity", self.COLD_START_HARD_RETRIEVAL_AVG_SIMILARITY)
        if (max_sim < mq_max_sim or avg_sim < mq_avg_sim) and has_evidence("multiquery"):
            return "multiquery"

        if query_len <= learned("multiquery", "query_length", 3) and has_evidence("multiquery"):
            return "multiquery"

        rerank_sim = learned("reranked", "max_similarity", self.COLD_START_HIGH_CONFIDENCE_SIMILARITY)
        rerank_redund = learned("reranked", "redundancy", self.COLD_START_LOW_REDUNDANCY)
        if max_sim > rerank_sim and redundancy < rerank_redund and has_evidence("reranked"):
            return "reranked"

        return default_pipeline

    def _route_cold_start(self, f: Dict[str, Any]) -> str:
        """
        Pure rule-based selection, used before any benchmark() has been run.
        Draws on whatever of the three profiles (corpus/query/probe) are
        present in `f` — MetaRAG._extract_query_features() merges all three,
        but a router can also be called with a partial dict (e.g. a user
        only supplying probe features) and this degrades gracefully via
        the .get(key, default) fallbacks below.
        """
        max_sim = f.get("max_similarity", 0.0)
        avg_sim = f.get("avg_similarity", 0.0)
        redund = f.get("redundancy", 0.0)

        ocr_ratio = f.get("ocr_ratio", 0.0)
        numeric_ratio = f.get("numeric_ratio", 0.0)
        short_doc_ratio = f.get("short_doc_ratio", 0.0)

        is_short = f.get("is_short", False)
        is_long = f.get("is_long", False)
        has_number = f.get("contains_number", False)
        has_operator = f.get("has_operator", False)
        starts_with_wh = f.get("starts_with_wh", False)

        if max_sim > self.COLD_START_HIGH_CONFIDENCE_SIMILARITY and redund < self.COLD_START_LOW_REDUNDANCY:
            return "reranked"

        if numeric_ratio > self.COLD_START_NUMERIC_HEAVY_RATIO or short_doc_ratio > self.COLD_START_SHORT_DOC_RATIO:
            return "straight" if (has_number or not starts_with_wh) else "hybrid"

        if max_sim < self.COLD_START_HARD_RETRIEVAL_SIMILARITY or avg_sim < self.COLD_START_HARD_RETRIEVAL_AVG_SIMILARITY:
            return "multiquery"

        if redund > self.COLD_START_HIGH_REDUNDANCY:
            return "mmr"

        if ocr_ratio > self.COLD_START_OCR_RATIO:
            return "hybrid"

        if is_long or has_operator:
            return "multiquery"

        return "hybrid"

    # ═════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═════════════════════════════════════════════════════════

    def save(self, path: str = None) -> None:
        """
        Save learned thresholds to disk.

        Args:
            path: directory to save into. Defaults to self.base_path.
                  Pass an explicit path to save a copy elsewhere (e.g. to
                  share with a teammate or bundle with a shipped corpus).
        """
        target_dir = Path(path) if path else self.base_path
        if target_dir is None:
            raise ValueError("No path given and no base_path set on this Router.")

        target_path = target_dir / "router_thresholds.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w") as f:
            json.dump(self.thresholds, f, indent=2)
        print(f"[Router] Thresholds saved to {target_path}")

    def load(self, path: str = None) -> bool:
        """
        Load thresholds from disk — this router's own saved file, OR any
        other router_thresholds.json a user points it at (a teammate's
        project, a pre-trained file shipped alongside a corpus, etc.).

        Args:
            path: directory containing router_thresholds.json.
                  Defaults to self.base_path.

        Returns:
            True if loaded successfully, False otherwise (never raises —
            callers should check the return value, e.g. MetaRAG.update_router_thresholds()).
        """
        source_dir = Path(path) if path else self.base_path
        if source_dir is None:
            print("[Router] No path given and no base_path set — nothing to load.")
            return False

        source_path = source_dir / "router_thresholds.json"
        if not source_path.exists():
            print(f"[Router] No saved thresholds at {source_path}")
            return False

        try:
            with open(source_path, "r") as f:
                self.thresholds = json.load(f)
            self.rules = sorted(self.thresholds.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True)
            self.is_trained = True
            print(f"[Router] Loaded thresholds from {source_path}")
            return True
        except Exception as e:
            print(f"[Router] Failed to load: {e}")
            return False

    # ═════════════════════════════════════════════════════════
    # INTROSPECTION
    # ═════════════════════════════════════════════════════════

    def explain(self, pipeline: str, features: Optional[Dict] = None) -> str:
        """Human-readable justification for a routing decision."""
        if not self.is_trained:
            return f"{pipeline}: cold-start rule-based selection (no benchmark trained yet)"

        if pipeline not in self.thresholds:
            return f"Unknown pipeline: {pipeline}"

        t = self.thresholds[pipeline]
        is_default = self.rules and self.rules[0][0] == pipeline

        explanation = f"{pipeline}: {t['win_rate']:.1%} win rate, avg score {t['avg_composite']:.2f}"
        explanation += " (benchmark winner)" if is_default else " (selected via refinement rule)"
        return explanation

    def get_stats(self) -> Dict[str, Any]:
        """Return training statistics — used by MetaRAG.get_router_thresholds()."""
        if not self.is_trained:
            return {"status": "cold_start", "mode": "rule-based (no benchmark trained yet)"}

        return {
            "status": "trained",
            "num_pipelines": len(self.thresholds),
            "thresholds": self.thresholds,
            "top_pipeline": self.rules[0][0] if self.rules else None,
            "top_win_rate": self.rules[0][1]["win_rate"] if self.rules else 0,
        }