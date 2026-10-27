"""Tests for the Gemini client. They use a fake Gemini so they dont need an API key
and dont use any of the free tier quota."""

from types import SimpleNamespace

import pytest
from google.genai import errors

from rag.impl.llm import gemini
from rag.impl.llm.gemini import GeminiClient, MissingAPIKeyError, Throttle


def api_error(code: int) -> errors.APIError:
    error_class = errors.ServerError if code >= 500 else errors.ClientError
    return error_class(code, {"error": {"code": code, "message": "fake", "status": "FAKE"}})


class FakeClock:
    """Fake time: sleeping just moves the clock forward, so the tests run instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeModels:
    """Replaces client.models, remembers the calls and can fail for some models."""

    def __init__(self, failing: dict[str, int] | None = None) -> None:
        self.failing = failing or {}  # model name -> error code it fails with
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["model"] in self.failing:
            raise api_error(self.failing[kwargs["model"]])
        return SimpleNamespace(text="Hello there", usage_metadata={"total_token_count": 12})

    def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["model"] in self.failing:
            raise api_error(self.failing[kwargs["model"]])
        yield SimpleNamespace(text="Hel", usage_metadata=None)
        yield SimpleNamespace(text="lo", usage_metadata=None)
        yield SimpleNamespace(text=None, usage_metadata={"total_token_count": 7})


def make_client(models: FakeModels, fallback_models: list[str] | None = None) -> GeminiClient:
    return GeminiClient(
        model="main-model",
        fallback_models=fallback_models or [],
        throttle=Throttle(requests_per_minute=60_000),  # basically no waiting
        client=SimpleNamespace(models=models),
    )


def test_throttle_waits_between_calls():
    clock = FakeClock()
    throttle = Throttle(requests_per_minute=10, clock=clock.time, sleep=clock.sleep)
    throttle.wait()  # first call, no waiting
    throttle.wait()  # straight after: has to wait 60 / 10 = 6 seconds
    assert clock.sleeps == [6.0]


def test_throttle_does_not_wait_if_enough_time_passed():
    clock = FakeClock()
    throttle = Throttle(requests_per_minute=10, clock=clock.time, sleep=clock.sleep)
    throttle.wait()
    clock.now += 10
    throttle.wait()
    assert clock.sleeps == []


def test_generate_sends_prompt_and_returns_text():
    models = FakeModels()
    client = make_client(models)
    answer = client.generate("Say hello", system="You are polite")
    assert answer == "Hello there"
    call = models.calls[0]
    assert call["model"] == "main-model"
    assert call["contents"] == "Say hello"
    assert call["config"].system_instruction == "You are polite"
    assert client.last_usage == {"total_token_count": 12}


def test_generate_without_system_prompt_sends_no_config():
    models = FakeModels()
    make_client(models).generate("Say hello")
    assert models.calls[0]["config"] is None


def test_stream_yields_text_pieces():
    client = make_client(FakeModels())
    assert list(client.stream("Say hello")) == ["Hel", "lo"]
    assert client.last_usage == {"total_token_count": 7}


def test_falls_back_to_next_model_when_overloaded():
    models = FakeModels(failing={"main-model": 503})
    client = make_client(models, fallback_models=["backup-model"])
    assert client.generate("Say hello") == "Hello there"
    assert list(client.stream("Say hello")) == ["Hel", "lo"]
    assert [call["model"] for call in models.calls] == [
        "main-model",
        "backup-model",
        "main-model",
        "backup-model",
    ]


def test_no_fallback_for_other_errors():
    # a 400 means our request is wrong, another model wont fix that
    models = FakeModels(failing={"main-model": 400})
    client = make_client(models, fallback_models=["backup-model"])
    with pytest.raises(errors.ClientError):
        client.generate("Say hello")
    assert len(models.calls) == 1


def test_error_is_raised_when_all_models_fail():
    models = FakeModels(failing={"main-model": 503, "backup-model": 429})
    client = make_client(models, fallback_models=["backup-model"])
    with pytest.raises(errors.ClientError):
        client.generate("Say hello")


def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(gemini.settings, "google_api_key", "")
    with pytest.raises(MissingAPIKeyError):
        GeminiClient()
