"""
MetaRAG v0.2 — Intelligent Pipeline Selection Engine

Core Philosophy:
- Framework, not product. Users own their router, their metrics, their optimization.
- LearnedRuleRouter learns thresholds from benchmark data (no black-box ML).
- Every method is measurable. Every output is explorable.

Usage:
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from metarag import MetaRAG

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="mistral", temperature=0)

    rag = MetaRAG(docs="./docs", embeddings=embeddings, llm=llm)
    rag.fit()
    
    # v0.2: Generate benchmark + train router
    rag.benchmark(num_queries=100)
    
    # Ask uses learned router
    answer = rag.ask("what is the policy?")
    print(answer)
"""

from __future__ import annotations
import os
import sys
import json
import time
import pandas as pd
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class Answer:
    """Answer returned by ask()"""
    text: str
    query: str
    pipeline: str
    chunks: list
    score: float
    latency_ms: float
    features: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"\n{'='*55}\n"
            f"  Query    : {self.query}\n"
            f"  Pipeline : {self.pipeline}\n"
            f"  Score    : {self.score:.2f}\n"
            f"  Latency  : {self.latency_ms:.0f}ms\n"
            f"{'─'*55}\n"
            f"  {self.text}\n"
            f"{'='*55}"
        )


@dataclass
class BenchmarkRow:
    """Single row in benchmark CSV"""
    query: str
    pipeline: str
    faithfulness: float
    relevancy: float
    precision: float
    coverage: float
    redundancy: float
    composite: float
    latency_ms: float


# ─────────────────────────────────────────────────────────────
# MetaRAG
# ─────────────────────────────────────────────────────────────

class MetaRAG:
    """
    MetaRAG v0.2 — Intelligent Pipeline Selection Engine
    
    Infrastructure for building domain-optimized RAG systems.
    Benchmarks pipelines on your data, learns routing thresholds.
    """

    VERSION = "0.2.0"

    def __init__(
        self,
        docs: Union[str, List[str]],
        embeddings,
        llm,
        project: str = "default",
        chunk_size: int = 500,
        k: int = 4,
        eval_preset: str = "balanced",
        verbose: bool = True,
    ):
        """Initialize MetaRAG project."""
        if embeddings is None:
            raise ValueError("embeddings required. See docs.")
        if llm is None:
            raise ValueError("llm required. See docs.")

        self.docs_path = docs
        self.embeddings = embeddings
        self.llm = llm
        self.project = project
        self.chunk_size = chunk_size
        self.k = k
        self.eval_preset = eval_preset
        self.verbose = verbose

        # Storage paths
        self._base = os.path.join(".metarag", project)
        self._index_dir = os.path.join(self._base, "index")
        self._cache_dir = os.path.join(self._base, "cache")
        self._logs_dir = os.path.join(self._base, "logs")
        self._profile_path = os.path.join(self._base, "corpus_profile.json")
        self._benchmark_path = os.path.join(self._base, "benchmark.csv")
        self._router_path = os.path.join(self._base, "router_thresholds.json")
        self._log_path = os.path.join(self._logs_dir, "queries.jsonl")

        for d in [self._index_dir, self._cache_dir, self._logs_dir]:
            os.makedirs(d, exist_ok=True)

        # Internal state
        self._db = None
        self._chunks = None
        self._corpus_profile = None
        self._pipelines = None
        self._router = None
        self._learned_router = None
        self._generator = None
        self._evaluator = None
        self._fitted = False

        self._log(f"MetaRAG v{self.VERSION} — project='{project}'")

    # ─────────────────────────────────────────────────────────
    # CORE: FIT
    # ─────────────────────────────────────────────────────────

    def fit(self, force: bool = False) -> "MetaRAG":
        """
        Load documents → chunk → index → profile → ready.

        Args:
            force: rebuild everything from scratch
        """
        t0 = time.time()
        self._log("fit() starting...")

        self._load_docs_and_index(force=force)
        self._load_corpus_profile(force=force)
        self._setup_pipelines()
        self._setup_router()
        self._setup_generator()
        self._setup_evaluator()

        self._fitted = True
        elapsed = round((time.time() - t0) * 1000)
        self._log(f"fit() done in {elapsed}ms — ready.")
        return self

    # ─────────────────────────────────────────────────────────
    # CORE: ASK (inference)
    # ─────────────────────────────────────────────────────────

    def ask(self, query: str) -> Answer:
        """
        Ask a question. Router picks best pipeline. Returns Answer.

        Args:
            query: your question string

        Returns:
            Answer with .text .pipeline .score .chunks .latency_ms
        """
        self._check_fitted()
        t0 = time.time()

        # Route: use learned router if trained, else fallback
        if self._learned_router and self._learned_router.is_trained:
            features = self._extract_query_features(query)
            pipeline_name = self._learned_router.route(features)
            explanation = self._learned_router.explain(pipeline_name)
            self._log(f"Router: {explanation}")
        else:
            decision = self._router.route(query)
            pipeline_name = decision["pipeline"]
            features = decision["features"]

        # Run pipeline
        result = self._pipelines[pipeline_name].run(query)
        chunks = result["chunks"]

        # Generate answer
        gen_answer = self._generator.generate(
            query=query,
            chunks=chunks,
            pipeline=pipeline_name,
        )

        # Evaluate
        score = self._evaluator.evaluate(gen_answer)

        # Create answer
        answer = Answer(
            text=gen_answer.text,
            query=query,
            pipeline=pipeline_name,
            chunks=chunks,
            score=score.composite,
            latency_ms=round((time.time() - t0) * 1000, 2),
            features=features if not self._learned_router.is_trained else {},
        )

        self._write_log(answer, score)
        return answer

    # ─────────────────────────────────────────────────────────
    # v0.2: BENCHMARK
    # ─────────────────────────────────────────────────────────

    def benchmark(
        self,
        num_queries: int = 100,
        auto_generate: bool = True,
        queries: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run all pipelines on benchmark queries. Generate labels. Train router.

        This is the core v0.2 feature: user provides queries (or we generate them),
        we run all pipelines, evaluate, and learn thresholds.

        Args:
            num_queries: if auto_generate=True, generate N queries
            auto_generate: if True, use LLM to generate queries from docs
            queries: if provided, use these queries instead

        Returns:
            benchmark_df (DataFrame with all results)
        """
        self._check_fitted()
        t0 = time.time()

        self._log(f"benchmark() starting...")

        # Step 1: Get queries
        if queries:
            queries_to_run = queries
        elif auto_generate:
            queries_to_run = self._generate_queries(num_queries)
        else:
            raise ValueError(
                "Must provide queries or set auto_generate=True"
            )

        self._log(f"Benchmarking on {len(queries_to_run)} queries...")

        # Step 2: Run all pipelines on all queries
        results = []
        for i, query in enumerate(queries_to_run):
            self._log(f"  [{i+1}/{len(queries_to_run)}] {query[:50]}...")

            for pipeline_name, pipeline in self._pipelines.items():
                # Run pipeline
                result = pipeline.run(query)
                chunks = result["chunks"]

                # Generate answer
                gen_answer = self._generator.generate(
                    query=query,
                    chunks=chunks,
                    pipeline=pipeline_name,
                )

                # Evaluate
                score = self._evaluator.evaluate(gen_answer)

                # Extract features (corpus + query profiles)
                features = self._extract_query_features(query)

                # Determine winner (which pipeline scored highest)
                winning_pipeline = max(
                    [(p, self._evaluator.evaluate(
                        self._generator.generate(
                            query, self._pipelines[p].run(query)["chunks"], p
                        )
                    ).composite) for p in self._pipelines.keys()],
                    key=lambda x: x[1]
                )[0]

                # Log row
                row = {
                    "query": query,
                    "pipeline": pipeline_name,
                    "faithfulness": score.faithfulness,
                    "relevancy": score.relevancy,
                    "precision": score.precision_avg,
                    "coverage": score.coverage,
                    "redundancy": score.redundancy,
                    "composite": score.composite,
                    "latency_ms": result.get("latency_ms", 0),
                    "winning_pipeline": winning_pipeline,
                    **features,
                }
                results.append(row)

        # Step 3: Save benchmark CSV
        benchmark_df = pd.DataFrame(results)
        benchmark_df.to_csv(self._benchmark_path, index=False)
        self._log(f"Benchmark CSV saved: {self._benchmark_path}")

        # Step 4: Train learned router
        self._train_learned_router(benchmark_df)

        elapsed = round((time.time() - t0) * 1000)
        self._log(f"benchmark() done in {elapsed}ms")

        return benchmark_df

    # ─────────────────────────────────────────────────────────
    # v0.2: ROUTER TRAINING
    # ─────────────────────────────────────────────────────────

    def _train_learned_router(self, benchmark_df: pd.DataFrame):
        """Train LearnedRuleRouter from benchmark data."""
        from .router.learned_rule_router import LearnedRuleRouter

        self._learned_router = LearnedRuleRouter(self._base)
        self._learned_router.train(benchmark_df)

    def set_custom_router(self, model):
        """
        User plugs in their own router model.
        
        Expected interface:
            model.predict(features_dict) -> pipeline_name
        
        Args:
            model: any object with .predict(features_dict) method
        """
        self._custom_router = model
        self._log("Custom router set")

    # ─────────────────────────────────────────────────────────
    # v0.2: DATA ACCESS
    # ─────────────────────────────────────────────────────────

    def get_benchmark_data(self) -> pd.DataFrame:
        """
        Return benchmark CSV as DataFrame.

        Users can use this to train their own models:
            df = rag.get_benchmark_data()
            X = df[[feature_cols]]
            y = df['winning_pipeline']
            model.fit(X, y)
        """
        if not os.path.exists(self._benchmark_path):
            raise FileNotFoundError(
                f"Benchmark CSV not found. Run benchmark() first."
            )
        return pd.read_csv(self._benchmark_path)

    def get_router_thresholds(self) -> Dict[str, Any]:
        """Return learned router thresholds (for inspection)."""
        if self._learned_router is None:
            return {}
        return self._learned_router.get_thresholds()

    # ─────────────────────────────────────────────────────────
    # OBSERVABILITY: STATUS
    # ─────────────────────────────────────────────────────────

    def status(self):
        """Print current state of this MetaRAG project."""
        logs = self._read_logs()
        n_logs = len(logs)
        avg_score = (
            round(sum(r["score"] for r in logs) / n_logs, 3)
            if logs
            else 0
        )

        router_status = "learned" if (
            self._learned_router and self._learned_router.is_trained
        ) else "rule-based"

        print(f"\n{'='*50}")
        print(f"  MetaRAG v{self.VERSION} — '{self.project}'")
        print(f"{'='*50}")
        print(f"  Fitted        : {'✅' if self._fitted else '❌'}")
        print(f"  Chunks        : {len(self._chunks) if self._chunks else 0}")
        print(f"  Chunk size    : {self.chunk_size}")
        print(f"  k             : {self.k}")
        print(f"  Eval preset   : {self.eval_preset}")
        print(f"  Queries logged: {n_logs}")
        print(f"  Avg score     : {avg_score}")
        print(f"  Router        : {router_status}")
        print(f"  Storage       : {self._base}/")
        print(f"{'='*50}\n")

    def leaderboard(self):
        """Show pipeline performance from logged queries."""
        logs = self._read_logs()

        if not logs:
            print("[MetaRAG] No queries logged yet. Run ask() first.")
            return

        from collections import defaultdict

        pipe_scores = defaultdict(list)
        for row in logs:
            pipe_scores[row["pipeline"]].append(row["score"])

        ranked = sorted(
            pipe_scores.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True,
        )

        print(f"\n{'='*50}")
        print(f"  Leaderboard — {len(logs)} queries")
        print(f"{'='*50}")
        print(f"{'Pipeline':<14} {'Queries':>8} {'Avg':>7} {'Best':>7}")
        print(f"{'─'*50}")

        for i, (name, scores) in enumerate(ranked):
            avg = round(sum(scores) / len(scores), 3)
            best = round(max(scores), 3)
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            print(
                f"{medal} {name:<12} {len(scores):>8} "
                f"{avg:>7.3f} {best:>7.3f}"
            )

        print(f"{'='*50}\n")

    # ─────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────

    def save(self) -> "MetaRAG":
        """Save config to .metarag/<project>/config.json"""
        path = os.path.join(self._base, "config.json")
        config = {
            "version": self.VERSION,
            "project": self.project,
            "chunk_size": self.chunk_size,
            "k": self.k,
            "eval_preset": self.eval_preset,
            "docs_path": (
                self.docs_path
                if isinstance(self.docs_path, str)
                else list(self.docs_path)
            ),
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        self._log(f"Config saved → {path}")
        return self

    @classmethod
    def load(cls, path: str, embeddings, llm) -> "MetaRAG":
        """Restore MetaRAG from saved config."""
        config_path = (
            os.path.join(path, "config.json")
            if os.path.isdir(path)
            else path
        )

        with open(config_path) as f:
            config = json.load(f)

        rag = cls(
            docs=config["docs_path"],
            embeddings=embeddings,
            llm=llm,
            project=config["project"],
            chunk_size=config["chunk_size"],
            k=config["k"],
            eval_preset=config["eval_preset"],
        )
        rag.fit()
        return rag

    # ─────────────────────────────────────────────────────────
    # INTERNAL SETUP
    # ─────────────────────────────────────────────────────────

    def _load_docs_and_index(self, force: bool = False):
        """Load documents, chunk, index."""
        from .core.loader import DocumentLoader
        from .core.chunking import Chunker
        from .core.vector_db import VectorDB

        self._log("Loading documents...")
        docs = DocumentLoader(self.docs_path).load_all()
        if not docs:
            raise ValueError(f"No documents found at '{self.docs_path}'")
        self._log(f"{len(docs)} documents loaded")

        self._log("Chunking...")
        self._chunks = Chunker(
            strategy="recursive",
            chunk_size=self.chunk_size,
            overlap=50,
        ).chunk_documents(
            docs,
            cache_dir=os.path.join(self._cache_dir, "chunks"),
            force=force,
        )
        self._log(f"{len(self._chunks)} chunks ready")

        self._log("Building vector index...")
        self._db = VectorDB(
            self.embeddings,
            db_type="chroma",
            persist_directory=self._index_dir,
        )
        self._db.build(self._chunks, force=force)

    def _load_corpus_profile(self, force: bool = False):
        """Profile corpus (size, stats, etc)."""
        from .router.corpus_profiler import CorpusProfiler

        if not force and os.path.exists(self._profile_path):
            with open(self._profile_path) as f:
                self._corpus_profile = json.load(f)
            self._log("Corpus profile loaded from cache")
        else:
            self._corpus_profile = CorpusProfiler().profile(self._chunks)
            with open(self._profile_path, "w") as f:
                json.dump(self._corpus_profile, f, indent=2)
            self._log("Corpus profile saved")

    def _setup_pipelines(self):
        """Initialize all 7 pipeline variants."""
        from .core.retriever import get_retriever
        from .pipelines.pipeline import (
            Pipeline,
            Reranker,
            MultiQuery,
            HyDE,
        )

        bm25_ret = get_retriever("bm25", chunks=self._chunks, k=self.k)
        dense_ret = get_retriever("dense", vectordb=self._db, k=self.k)
        hybrid_ret = get_retriever(
            "hybrid",
            vectordb=self._db,
            chunks=self._chunks,
            k=self.k,
        )
        mmr_ret = get_retriever("mmr", vectordb=self._db, k=self.k)
        reranker = Reranker()

        self._pipelines = {
            "straight": Pipeline(retriever=bm25_ret),
            "dense": Pipeline(retriever=dense_ret),
            "hybrid": Pipeline(retriever=hybrid_ret),
            "reranked": Pipeline(retriever=dense_ret, reranker=reranker),
            "mmr": Pipeline(retriever=mmr_ret),
            "multiquery": Pipeline(
                retriever=hybrid_ret,
                multiquery=MultiQuery(self.llm, n=2),
            ),
            "hyde": Pipeline(retriever=dense_ret, hyde=HyDE(self.llm)),
        }
        self._log(f"{len(self._pipelines)} pipelines ready")

    def _setup_router(self):
        """Initialize rule-based router (fallback)."""
        from .router.selector import Router

        self._router = Router(self._db, self._corpus_profile)
        self._log("Router ready")

    def _setup_generator(self):
        """Initialize answer generator."""
        from .pipelines.generator import GeneratorWrapper

        self._generator = GeneratorWrapper(self.llm)
        self._log("Generator ready")

    def _setup_evaluator(self):
        """Initialize evaluator (5 metrics)."""
        from .Evaluator.evaluator import Evaluator

        self._evaluator = Evaluator(self.embeddings, preset=self.eval_preset)
        self._log("Evaluator ready")

    # ─────────────────────────────────────────────────────────
    # INTERNAL: FEATURE EXTRACTION
    # ─────────────────────────────────────────────────────────

    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract query features for routing."""
        from .router.query_profiler import QueryProfiler
        from .router.probe_profiler import ProbeProfiler

        query_prof = QueryProfiler().profile(query)
        probe_prof = ProbeProfiler().profile(query, self._db)

        return {
            **query_prof,
            **probe_prof,
            **self._corpus_profile,
        }

    def _generate_queries(self, num_queries: int) -> List[str]:
        """Auto-generate queries from docs using LLM."""
        # For v0.2, simple heuristic: extract sentences from chunks
        # v0.3 can use LLM-based generation

        import random

        sentences = []
        for chunk in self._chunks[:20]:  # sample first 20 chunks
            parts = chunk.split(". ")
            sentences.extend(
                [s.strip() for s in parts if len(s.split()) >= 5]
            )

        selected = random.sample(
            sentences,
            min(num_queries, len(sentences)),
        )
        return selected

    # ─────────────────────────────────────────────────────────
    # INTERNAL: LOGGING
    # ─────────────────────────────────────────────────────────

    def _write_log(self, answer: Answer, score):
        """Log query result to JSONL."""
        row = {
            "query": answer.query,
            "pipeline": answer.pipeline,
            "score": answer.score,
            "faithfulness": score.faithfulness,
            "relevancy": score.relevancy,
            "precision": score.precision_avg,
            "coverage": score.coverage,
            "redundancy": score.redundancy,
            "latency_ms": answer.latency_ms,
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _read_logs(self) -> list:
        """Read all logged queries."""
        if not os.path.exists(self._log_path):
            return []
        rows = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        return rows

    def _check_fitted(self):
        """Ensure fit() was called."""
        if not self._fitted:
            raise RuntimeError("Call fit() before ask()")

    def _log(self, msg: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[MetaRAG] {msg}")

    def __repr__(self):
        return (
            f"MetaRAG(project='{self.project}', "
            f"fitted={self._fitted}, "
            f"chunks={len(self._chunks) if self._chunks else 0}, "
            f"router={'learned' if (self._learned_router and self._learned_router.is_trained) else 'rule-based'})"
        )