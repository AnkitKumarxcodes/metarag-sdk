# evaluator.py

from __future__ import annotations
import math
from typing import List, Any, Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# Evaluation Result
# ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """
    Scores for one Answer.

    faithfulness  →  is the answer grounded in the chunks?
    relevancy     →  does the answer address the question?
    precision     →  were the retrieved chunks actually useful?
    composite     →  single number combining all three
    mode          →  "fast" or "deep"
    """
    faithfulness: float
    relevancy:    float
    precision:    float
    composite:    float
    mode:         str

    def __repr__(self):
        return (
            f"EvalResult(\n"
            f"  faithfulness = {self.faithfulness:.2f}\n"
            f"  relevancy    = {self.relevancy:.2f}\n"
            f"  precision    = {self.precision:.2f}\n"
            f"  composite    = {self.composite:.2f}\n"
            f"  mode         = {self.mode}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────
# Shared utility
# ─────────────────────────────────────────────────────────────

def _cosine(a: List[float], b: List[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _get_text(chunk) -> str:
    """Works with both LangChain Documents and MetaRAG Chunks."""
    return getattr(chunk, "page_content", None) or getattr(chunk, "text", "")


# ─────────────────────────────────────────────────────────────
# Fast Evaluator — embedding based, no LLM
# Runs on every query, powers the learning loop
# ─────────────────────────────────────────────────────────────

class FastEvaluator:
    """
    Scores an answer using embeddings and word overlap.
    No LLM calls. Runs in ~50ms.

    faithfulness  →  word overlap between answer and chunks
    relevancy     →  cosine similarity between query and answer
    precision     →  average cosine similarity between query and each chunk
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def faithfulness(self, answer_text: str, chunks: List[Any]) -> float:
        """
        How much of the answer is grounded in the retrieved chunks.
        Simple word overlap — fast and dependency free.
        """
        if not chunks or not answer_text:
            return 0.0

        answer_words = set(answer_text.lower().split())
        chunk_text   = " ".join(_get_text(c) for c in chunks)
        chunk_words  = set(chunk_text.lower().split())

        overlap = answer_words & chunk_words
        return len(overlap) / len(answer_words) if answer_words else 0.0

    def relevancy(self, query: str, answer_text: str) -> float:
        """
        How well the answer addresses the question.
        Cosine similarity between query and answer embeddings.
        """
        if not query or not answer_text:
            return 0.0

        q_emb = self.embedding_model.embed_query(query)
        a_emb = self.embedding_model.embed_query(answer_text)
        return max(0.0, _cosine(q_emb, a_emb))

    def precision(self, query: str, chunks: List[Any]) -> float:
        """
        How useful the retrieved chunks were for the query.
        Average cosine similarity between query and each chunk.
        """
        if not chunks or not query:
            return 0.0

        q_emb  = self.embedding_model.embed_query(query)
        scores = []

        for chunk in chunks:
            text  = _get_text(chunk)
            c_emb = self.embedding_model.embed_query(text)
            scores.append(max(0.0, _cosine(q_emb, c_emb)))

        return sum(scores) / len(scores) if scores else 0.0

    def evaluate(self, query: str, answer_text: str, chunks: List[Any]) -> EvalResult:
        f = self.faithfulness(answer_text, chunks)
        r = self.relevancy(query, answer_text)
        p = self.precision(query, chunks)
        c = round((f + r + p) / 3, 4)

        print(f"[FastEvaluator] faithfulness={f:.2f} relevancy={r:.2f} precision={p:.2f} composite={c:.2f}")

        return EvalResult(
            faithfulness = round(f, 4),
            relevancy    = round(r, 4),
            precision    = round(p, 4),
            composite    = c,
            mode         = "fast",
        )


# ─────────────────────────────────────────────────────────────
# Deep Evaluator — local LLM as judge
# Runs on demand for accurate benchmarking
# ─────────────────────────────────────────────────────────────

class DeepEvaluator:
    """
    Scores an answer using a local LLM as judge.
    More accurate than embedding overlap.
    Slower — 2-5 seconds per query.
    Uses Ollama — free, local, private.
    """

    FAITHFULNESS_PROMPT = """You are evaluating a RAG system.

Question: {query}

Retrieved context:
{context}

Answer given:
{answer}

Score how faithful the answer is to the context on a scale of 0.0 to 1.0.
1.0 = answer is completely grounded in the context, nothing made up
0.0 = answer is completely hallucinated, nothing from context

Reply with ONLY a number between 0.0 and 1.0. Nothing else."""

    RELEVANCY_PROMPT = """You are evaluating a RAG system.

Question: {query}

Answer given:
{answer}

Score how well the answer addresses the question on a scale of 0.0 to 1.0.
1.0 = answer directly and completely answers the question
0.0 = answer is completely irrelevant to the question

Reply with ONLY a number between 0.0 and 1.0. Nothing else."""

    def __init__(self, llm):
        self.llm = llm

    def _score(self, prompt: str) -> float:
        """Call LLM and parse the score it returns."""
        try:
            result = self.llm.invoke(prompt).content.strip()
            return max(0.0, min(1.0, float(result)))
        except Exception:
            print("[DeepEvaluator] Could not parse LLM score — defaulting to 0.5")
            return 0.5

    def faithfulness(self, query: str, answer_text: str, chunks: List[Any]) -> float:
        context = "\n\n".join(_get_text(c) for c in chunks)
        prompt  = self.FAITHFULNESS_PROMPT.format(
            query=query, context=context, answer=answer_text
        )
        return self._score(prompt)

    def relevancy(self, query: str, answer_text: str) -> float:
        prompt = self.RELEVANCY_PROMPT.format(query=query, answer=answer_text)
        return self._score(prompt)

    def precision(self, query: str, chunks: List[Any], embedding_model) -> float:
        """
        Precision still uses embeddings even in deep mode —
        LLM judging chunk relevance one by one is too slow.
        """
        fast = FastEvaluator(embedding_model)
        return fast.precision(query, chunks)

    def evaluate(
        self,
        query:           str,
        answer_text:     str,
        chunks:          List[Any],
        embedding_model  = None,
    ) -> EvalResult:

        print("[DeepEvaluator] Running LLM evaluation...")

        f = self.faithfulness(query, answer_text, chunks)
        r = self.relevancy(query, answer_text)
        p = (
            self.precision(query, chunks, embedding_model)
            if embedding_model
            else 0.0
        )
        c = round((f + r + p) / 3 if embedding_model else (f + r) / 2, 4)

        print(f"[DeepEvaluator] faithfulness={f:.2f} relevancy={r:.2f} precision={p:.2f} composite={c:.2f}")

        return EvalResult(
            faithfulness = round(f, 4),
            relevancy    = round(r, 4),
            precision    = round(p, 4),
            composite    = c,
            mode         = "deep",
        )


# ─────────────────────────────────────────────────────────────
# Evaluator — unified interface
# ─────────────────────────────────────────────────────────────

class Evaluator:
    """
    Main evaluator for MetaRAG.

    Default → fast mode, embedding based, runs on every query
    deep=True → LLM as judge, runs on demand

    Usage:
        evaluator = Evaluator(embedding_model=embeddings)

        # fast — always on
        result = evaluator.evaluate(answer)

        # deep — on demand
        result = evaluator.evaluate(answer, deep=True, llm=llm)
    """

    def __init__(self, embedding_model, llm=None):
        self.embedding_model = embedding_model
        self.llm             = llm
        self._fast           = FastEvaluator(embedding_model)
        self._deep           = DeepEvaluator(llm) if llm else None

    def evaluate(self, answer, deep: bool = False) -> EvalResult:
        """
        Score an Answer object.

        answer  →  Answer from generator.py
        deep    →  use LLM as judge instead of embeddings
        """
        query       = answer.query
        answer_text = answer.text
        chunks      = answer.chunks

        if deep:
            if self._deep is None:
                print("[Evaluator] No LLM provided — falling back to fast mode")
                return self._fast.evaluate(query, answer_text, chunks)
            return self._deep.evaluate(query, answer_text, chunks, self.embedding_model)

        return self._fast.evaluate(query, answer_text, chunks)

    def set_llm(self, llm):
        """Add or swap the LLM for deep evaluation."""
        self.llm   = llm
        self._deep = DeepEvaluator(llm)