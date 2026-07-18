from metarag.pipelines.generator import (
    build_prompt,
    GeneratorWrapper,
    GeneratorInterface,
)

import pytest
# ============================================================
# Fake Generators
# ============================================================

from metarag.utils import FakeGenerator , RetryGenerator , AlwaysFailGenerator



class RateLimitGenerator(GeneratorInterface):

    def __init__(self):

        self.calls = 0

    def generate(self, prompt):

        self.calls += 1

        if self.calls < 3:
            raise Exception("429 Rate_Limit")

        return "Recovered"



# ============================================================
# build_prompt()
# ============================================================

def test_prompt_contains_query():

    prompt = build_prompt(

        "What is AI?",

        ["Artificial Intelligence"]

    )

    assert "What is AI?" in prompt


def test_prompt_contains_context():

    prompt = build_prompt(

        "AI",

        ["Context A", "Context B"]

    )

    assert "Context A" in prompt

    assert "Context B" in prompt


def test_prompt_numbers_chunks():

    prompt = build_prompt(

        "AI",

        ["A", "B", "C"]

    )

    assert "[1]" in prompt

    assert "[2]" in prompt

    assert "[3]" in prompt


def test_prompt_with_tuple_chunks():

    prompt = build_prompt(

        "AI",

        [

            ("Chunk A", 0.9),

            ("Chunk B", 0.8),

        ]

    )

    assert "Chunk A" in prompt

    assert "Chunk B" in prompt


def test_prompt_empty_chunks():

    prompt = build_prompt(

        "AI",

        []

    )

    assert "Context:" in prompt

# ============================================================
# Wrapper
# ============================================================

def test_wrapper_creation():

    wrapper = GeneratorWrapper(

        FakeGenerator()

    )

    assert wrapper is not None


def test_wrapper_requires_generate():

    class Bad:

        pass

    try:

        GeneratorWrapper(Bad())

        assert False

    except TypeError:

        assert True


def test_generate_text():

    wrapper = GeneratorWrapper(

        FakeGenerator()

    )

    text, latency = wrapper.generate_text(

        "AI",

        ["Chunk"]

    )

    assert isinstance(text, str)

    assert latency >= 0


def test_model_name():

    wrapper = GeneratorWrapper(

        FakeGenerator()

    )

    assert wrapper.model_name == "FakeGenerator"


def test_custom_model_name():

    wrapper = GeneratorWrapper(

        FakeGenerator(),

        model_name="MyLLM",

    )

    assert wrapper.model_name == "MyLLM"

# ============================================================
# Retry Logic
# ============================================================

def test_retry_success():

    wrapper = GeneratorWrapper(

        RetryGenerator()

    )

    text, _ = wrapper.generate_text(

        "AI",

        ["Chunk"]

    )

    assert text == "Recovered"


def test_rate_limit_retry(monkeypatch):

    monkeypatch.setattr("time.sleep", lambda *_: None)   # real generator.py calls time.sleep(10*attempt) on 429s

    wrapper = GeneratorWrapper(

        RateLimitGenerator()

    )

    text, _ = wrapper.generate_text(

        "AI",

        ["Chunk"]

    )

    assert text == "Recovered"


def test_failure_after_retries():

    wrapper = GeneratorWrapper(

        AlwaysFailGenerator()

    )

    try:

        wrapper.generate_text(

            "AI",

            ["Chunk"]

        )

        assert False

    except RuntimeError:

        assert True


# ============================================================
# Chunk Handling
# ============================================================

def test_chunk_object():

    class Chunk:

        text = "Chunk Object"

    wrapper = GeneratorWrapper(

        FakeGenerator()

    )

    text, _ = wrapper.generate_text(

        "AI",

        [Chunk()]

    )

    assert text == "This is a generated answer based on the retrieved context."


def test_multiple_chunks():

    wrapper = GeneratorWrapper(

        FakeGenerator()

    )

    text, _ = wrapper.generate_text(

        "AI",

        [

            "A",

            "B",

            "C",

        ]

    )

    assert text == "This is a generated answer based on the retrieved context."


def test_latency_is_float():

    wrapper = GeneratorWrapper(

        FakeGenerator()

    )

    _, latency = wrapper.generate_text(

        "AI",

        ["Chunk"]

    )

    assert isinstance(latency, float)

# ============================================================
# OllamaGenerator (mocked HTTP — no real Ollama server needed)
# ============================================================

from metarag.pipelines.generator import OllamaGenerator


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def test_ollama_generator_connects_successfully(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse(200))
    gen = OllamaGenerator(model="mistral")
    assert gen.model == "mistral"


def test_ollama_generator_bad_status_raises_connection_error(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse(500))
    with pytest.raises(ConnectionError):
        OllamaGenerator(model="mistral")


def test_ollama_generator_unreachable_raises_connection_error(monkeypatch):
    def raise_network_error(*a, **k):
        raise OSError("network unreachable")
    monkeypatch.setattr("requests.get", raise_network_error)
    with pytest.raises(ConnectionError):
        OllamaGenerator(model="mistral")


def test_ollama_generator_generate_calls_api_and_strips_response(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse(200))
    monkeypatch.setattr("requests.post", lambda *a, **k: _FakeResponse(200, {"response": "  Paris  "}))
    gen = OllamaGenerator(model="mistral")
    assert gen.generate("What is the capital of France?") == "Paris"


# ============================================================
# Retry exhaustion — rate-limited on every attempt, never recovers
# ============================================================

class AlwaysRateLimitedGenerator(GeneratorInterface):
    def generate(self, prompt):
        raise Exception("429 rate_limit")


def test_rate_limit_on_every_attempt_exhausts_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    wrapper = GeneratorWrapper(AlwaysRateLimitedGenerator())
    with pytest.raises(RuntimeError):
        wrapper.generate_text("AI", ["Chunk"])