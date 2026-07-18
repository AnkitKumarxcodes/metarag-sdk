# tests/test_evaluator.py
"""
Tests for metrics.py (faithfulness/relevancy/precision/coverage/redundancy),
scorer.py (Scorer presets), and evaluator.py (Evaluator orchestration).
Uses the same deterministic FakeEmbeddings pattern as test_metarag.py.
"""

import pytest

from metarag.Evaluator.metrics import faithfulness, relevancy, precision, coverage, redundancy
from metarag.Evaluator.scorer import Scorer, WEIGHTS
from metarag.Evaluator.evaluator import Evaluator

from metarag.utils import FakeEmbeddings


class FakeAnswer:
    def __init__(self, query, text, chunks, latency_ms=100.0):
        self.query = query
        self.text = text
        self.chunks = chunks
        self.latency_ms = latency_ms


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


# ─────────────────────────────────────────────────────────
# faithfulness
# ─────────────────────────────────────────────────────────

def test_faithfulness_identical_text_high_score(embeddings):
    score = faithfulness("machine learning is powerful", ["machine learning is powerful"], embeddings)
    assert score > 0.9


def test_faithfulness_empty_answer_returns_zero(embeddings):
    assert faithfulness("", ["some context"], embeddings) == 0.0


def test_faithfulness_empty_chunks_returns_zero(embeddings):
    assert faithfulness("some answer", [], embeddings) == 0.0


# ─────────────────────────────────────────────────────────
# relevancy
# ─────────────────────────────────────────────────────────

def test_relevancy_identical_text_high_score(embeddings):
    score = relevancy("what is AI", "what is AI", embeddings)
    assert score > 0.9


def test_relevancy_empty_inputs_return_zero(embeddings):
    assert relevancy("", "answer", embeddings) == 0.0
    assert relevancy("query", "", embeddings) == 0.0


# ─────────────────────────────────────────────────────────
# precision
# ─────────────────────────────────────────────────────────

def test_precision_returns_dict_shape(embeddings):
    result = precision("machine learning", ["machine learning basics", "cooking recipes"], embeddings)
    assert set(result.keys()) == {"max", "avg", "std"}


def test_precision_empty_returns_zeros(embeddings):
    result = precision("", [], embeddings)
    assert result == {"max": 0.0, "avg": 0.0, "std": 0.0}


def test_precision_max_gte_avg(embeddings):
    result = precision("machine learning", ["machine learning", "unrelated text here"], embeddings)
    assert result["max"] >= result["avg"]


# ─────────────────────────────────────────────────────────
# coverage
# ─────────────────────────────────────────────────────────

def test_coverage_full_overlap():
    score = coverage("machine learning", ["machine learning is a field"])
    assert score == 1.0


def test_coverage_no_overlap():
    score = coverage("machine learning", ["cooking recipes are fun"])
    assert score == 0.0


def test_coverage_empty_returns_zero():
    assert coverage("", ["text"]) == 0.0
    assert coverage("query", []) == 0.0


def test_coverage_partial_overlap():
    score = coverage("machine learning basics", ["machine learning is powerful"])
    assert 0.0 < score < 1.0


# ─────────────────────────────────────────────────────────
# redundancy
# ─────────────────────────────────────────────────────────

def test_redundancy_identical_chunks_high(embeddings):
    score = redundancy(["same text here", "same text here"], embeddings)
    assert score > 0.9


def test_redundancy_single_chunk_returns_zero(embeddings):
    assert redundancy(["only one chunk"], embeddings) == 0.0


def test_redundancy_empty_returns_zero(embeddings):
    assert redundancy([], embeddings) == 0.0


# ─────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────

def test_scorer_presets_exist():
    for preset in ["balanced", "precision", "recall"]:
        scorer = Scorer(preset=preset)
        assert scorer.weights == WEIGHTS[preset]


def test_scorer_invalid_preset_raises():
    with pytest.raises(ValueError):
        Scorer(preset="nonexistent")


def test_scorer_custom_weights():
    custom = {"faithfulness": 1.0, "relevancy": 0, "precision": 0, "coverage": 0, "redundancy": 0, "latency": 0}
    scorer = Scorer(weights=custom)
    result = scorer.score(
        faithfulness=0.8, relevancy=0.0, precision={"max": 0, "avg": 0, "std": 0},
        coverage=0.0, redundancy=0.0, latency_ms=0,
    )
    assert result.composite == pytest.approx(0.8, abs=0.01)


def test_scorer_composite_never_negative():
    scorer = Scorer(preset="balanced")
    result = scorer.score(
        faithfulness=0.0, relevancy=0.0, precision={"max": 0, "avg": 0, "std": 0},
        coverage=0.0, redundancy=1.0, latency_ms=999999,
    )
    assert result.composite >= 0.0


def test_scorer_result_has_all_fields():
    scorer = Scorer(preset="balanced")
    result = scorer.score(
        faithfulness=0.5, relevancy=0.5, precision={"max": 0.6, "avg": 0.5, "std": 0.1},
        coverage=0.5, redundancy=0.2, latency_ms=200,
    )
    d = result.as_dict()
    assert set(d.keys()) == {
        "faithfulness", "relevancy", "precision_avg", "precision_max",
        "precision_std", "coverage", "redundancy", "latency_ms", "composite",
    }


# ─────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────

def test_evaluator_returns_score_result(embeddings):
    evaluator = Evaluator(embeddings, preset="balanced")
    answer = FakeAnswer("What is AI?", "AI is artificial intelligence", ["AI is artificial intelligence"])
    result = evaluator.evaluate(answer)
    assert 0.0 <= result.composite <= 1.0


def test_evaluator_no_chunks_warns_but_does_not_crash(embeddings, capsys):
    evaluator = Evaluator(embeddings, preset="balanced")
    answer = FakeAnswer("What is AI?", "some answer", [])
    result = evaluator.evaluate(answer)
    assert result.composite == 0.0


def test_evaluator_set_preset(embeddings):
    evaluator = Evaluator(embeddings, preset="balanced")
    evaluator.set_preset("precision")
    assert evaluator.scorer.weights == WEIGHTS["precision"]


# ─────────────────────────────────────────────────────────
# Preset-specific latency penalty (precision/recall vs balanced)
# ─────────────────────────────────────────────────────────

def _base_score_kwargs():
    return dict(faithfulness=0.5, relevancy=0.5, precision={"max": 0.5, "avg": 0.5, "std": 0.0},
                coverage=0.5, redundancy=0.0)


def test_precision_preset_applies_latency_penalty():
    scorer = Scorer(preset="precision")
    no_latency = scorer.score(**_base_score_kwargs(), latency_ms=0)
    max_latency = scorer.score(**_base_score_kwargs(), latency_ms=10000)   # MAX_LATENCY_MS
    assert no_latency.composite - max_latency.composite == pytest.approx(0.05)   # precision's latency weight


def test_recall_preset_ignores_latency():
    scorer = Scorer(preset="recall")
    no_latency = scorer.score(**_base_score_kwargs(), latency_ms=0)
    max_latency = scorer.score(**_base_score_kwargs(), latency_ms=10000)
    assert no_latency.composite == max_latency.composite   # recall's latency weight is 0


# ─────────────────────────────────────────────────────────
# redundancy() averages across ALL pairs, not just the first
# ─────────────────────────────────────────────────────────

def test_redundancy_averages_across_all_pairs(embeddings):
    two_identical = redundancy(["same text here", "same text here"], embeddings)
    three_with_one_outlier = redundancy(
        ["same text here", "same text here", "completely unrelated content"], embeddings
    )
    assert three_with_one_outlier < two_identical


# ─────────────────────────────────────────────────────────
# Scorer latency-penalty math (only "precision" preset has nonzero latency weight)
# ─────────────────────────────────────────────────────────

def test_precision_preset_applies_latency_penalty():
    scorer = Scorer(preset="precision")
    result = scorer.score(
        faithfulness=1.0, relevancy=0.0, precision={"max": 0, "avg": 0, "std": 0},
        coverage=0.0, redundancy=0.0, latency_ms=5000,
    )
    # 0.20*1.0 - 0.05*(5000/10000) = 0.175
    assert result.composite == pytest.approx(0.175, abs=0.001)


def test_precision_preset_latency_penalty_caps_at_max_latency():
    scorer = Scorer(preset="precision")
    kwargs = dict(faithfulness=1.0, relevancy=0.0, precision={"max": 0, "avg": 0, "std": 0},
                   coverage=0.0, redundancy=0.0)
    at_cap = scorer.score(latency_ms=10000, **kwargs)
    way_over = scorer.score(latency_ms=999999, **kwargs)
    assert at_cap.composite == pytest.approx(way_over.composite)


def test_recall_preset_ignores_latency_entirely():
    scorer = Scorer(preset="recall")
    kwargs = dict(faithfulness=0.5, relevancy=0.5, precision={"max": 0, "avg": 0.5, "std": 0},
                   coverage=0.5, redundancy=0.2)
    fast = scorer.score(latency_ms=1, **kwargs)
    slow = scorer.score(latency_ms=999999, **kwargs)
    assert fast.composite == pytest.approx(slow.composite)


# ─────────────────────────────────────────────────────────
# redundancy() — averaging across more than 2 chunks
# ─────────────────────────────────────────────────────────

def test_redundancy_averages_across_all_pairs_not_just_first(embeddings):
    all_same = redundancy(["same text here"] * 3, embeddings)
    mixed = redundancy(["same text here", "same text here", "totally unrelated content"], embeddings)
    assert all_same == pytest.approx(1.0, abs=0.05)
    assert mixed < all_same