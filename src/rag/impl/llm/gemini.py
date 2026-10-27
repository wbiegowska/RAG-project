"""Gemini client (Google AI Studio), the only LLM backend of the project.

Everything that talks to the LLM goes through here: text-to-SQL, the router, the final
answer and later the RAGAS judge. This way the API key, the model name and the rate
limiting are all handled in one place.

Use `get_llm()` instead of creating a GeminiClient yourself, it gives back the same client
every time so all the calls share one throttle. If every part of the pipeline had its own
client we would go over the free tier limit without noticing.
"""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Any

from google import genai
from google.genai import errors, types

from rag.base.llm import LLMClient
from rag.settings import settings

logger = logging.getLogger(__name__)

# Retries are done by the google-genai SDK itself, it retries on 429 (too many requests)
# and 5xx errors with exponential backoff (waits of about 5s then 10s). If the model still
# fails after that we switch to the next model in settings.llm_fallback_models.
RETRY_OPTIONS = types.HttpRetryOptions(attempts=3, initial_delay=5.0, max_delay=60.0)

# errors where trying another model makes sense: model overloaded (503) or out of quota (429).
# We saw the 503 "high demand" one quite often while testing with the free tier
FALLBACK_CODES = {429, 503}


class MissingAPIKeyError(RuntimeError):
    """Raised when GOOGLE_API_KEY is not set in the .env file"""


class Throttle:
    """Waits between calls so we never send more than `requests_per_minute` per minute.

    Before each call it checks how long ago the last one was and sleeps for the rest of
    the interval if needed. There is a lock so it also works when several threads call
    Gemini at the same time (RAGAS does that), they just wait for their turn.
    """

    def __init__(
        self,
        requests_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = 60.0 / requests_per_minute
        self._clock = clock  # clock and sleep can be replaced in the tests
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last_call: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last_call is not None:
                remaining = self.min_interval - (now - self._last_call)
                if remaining > 0:
                    self._sleep(remaining)
                    now = self._clock()
            self._last_call = now


class GeminiClient(LLMClient):
    """LLMClient implementation for Gemini, with throttling, retries and fallback models."""

    def __init__(
        self,
        model: str | None = None,
        fallback_models: list[str] | None = None,
        throttle: Throttle | None = None,
        client: Any = None,
    ) -> None:
        self.model = model or settings.llm_model
        if fallback_models is None:
            fallback_models = settings.llm_fallback_models
        self.fallback_models = [m for m in fallback_models if m != self.model]
        self.throttle = throttle or Throttle(settings.llm_requests_per_minute)
        # `client` is only passed in the tests (a fake one), normally we create the real one
        self._client = client or self._create_client()
        # token counts of the last call, used later by the monitoring
        self.last_usage: Any = None

    @staticmethod
    def _create_client() -> genai.Client:
        if not settings.google_api_key:
            raise MissingAPIKeyError(
                "GOOGLE_API_KEY is missing. Copy .env.example to .env and paste your key "
                "from https://aistudio.google.com/apikey"
            )
        return genai.Client(
            api_key=settings.google_api_key,
            http_options=types.HttpOptions(retry_options=RETRY_OPTIONS),
        )

    def generate(self, prompt: str, system: str | None = None) -> str:
        models = [self.model, *self.fallback_models]
        for i, model in enumerate(models):
            self.throttle.wait()
            try:
                response = self._client.models.generate_content(
                    model=model, contents=prompt, config=self._config(system)
                )
            except errors.APIError as error:
                if self._should_fall_back(error, model, models[i + 1 :]):
                    continue
                raise
            self.last_usage = response.usage_metadata
            return response.text or ""
        raise RuntimeError("No Gemini model to try")  # cant happen, the list is never empty

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        models = [self.model, *self.fallback_models]
        for i, model in enumerate(models):
            self.throttle.wait()
            started = False
            try:
                chunks = self._client.models.generate_content_stream(
                    model=model, contents=prompt, config=self._config(system)
                )
                for chunk in chunks:
                    if chunk.usage_metadata:
                        self.last_usage = chunk.usage_metadata
                    # some chunks have no text (e.g the last one with only the token counts)
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except errors.APIError as error:
                # once part of the answer is shown we cant switch model anymore
                if not started and self._should_fall_back(error, model, models[i + 1 :]):
                    continue
                raise

    @staticmethod
    def _should_fall_back(error: errors.APIError, model: str, next_models: list[str]) -> bool:
        if error.code not in FALLBACK_CODES or not next_models:
            return False
        logger.warning(
            "Gemini model %s failed (%s %s), trying %s instead",
            model,
            error.code,
            error.status,
            next_models[0],
        )
        return True

    @staticmethod
    def _config(system: str | None) -> types.GenerateContentConfig | None:
        # we dont set the temperature on purpose, Google recommends keeping the default
        # for the Gemini 3 models
        if system is None:
            return None
        return types.GenerateContentConfig(system_instruction=system)


def get_llm(model: str | None = None) -> GeminiClient:
    """Returns the shared client for a model (one per model, since limits are per model)"""
    return _shared_client(model or settings.llm_model)


@lru_cache
def _shared_client(model: str) -> GeminiClient:
    return GeminiClient(model=model)
