"""Judge abstraction.

A `Judge` is a wrapper around an LLM that scores RAG outputs. The base class
handles disk caching and retry; subclasses implement the actual API call.

Caching is keyed on (model, prompt, temperature) — change any of those and you
get a fresh call. This is critical: you will re-run experiments many times during
development, and judge calls cost real money. Cache hits should be the default.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import diskcache
from pydantic import BaseModel
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ragval" / "judge"


class _SharedRateLimiter:
    """A rate limiter shared by every Judge instance using the same model.

    This must be shared, not per-instance. Providers enforce rate limits per
    model at the ORGANIZATION level, but a benchmark run typically builds two
    judges on the same model — one as the answer generator, one as the scorer.
    With per-instance throttles the two interleave and the COMBINED request
    rate is roughly 4/3 of the configured rate, which quietly exceeds the cap
    and earns a stream of 429s. Keying the limiter on model_id fixes that.
    """

    _instances: dict[str, _SharedRateLimiter] = {}
    _registry_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_call = 0.0

    @classmethod
    def for_model(cls, model_id: str) -> _SharedRateLimiter:
        with cls._registry_lock:
            if model_id not in cls._instances:
                cls._instances[model_id] = cls()
            return cls._instances[model_id]

    def acquire(self, min_gap: float) -> None:
        """Block until at least `min_gap` seconds have passed since the last call."""
        if min_gap <= 0:
            return
        with self._lock:
            wait = min_gap - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()


class _DailyQuotaError(RuntimeError):
    """Base class for daily-quota exhaustion. Retrying these is pointless."""


def _is_daily_quota_error(exc: Exception) -> bool:
    """Distinguish a DAILY cap from a transient per-minute 429.

    Providers signal both with HTTP 429; only the message distinguishes them.
    Retrying a per-minute limit is correct. Retrying a per-day limit just
    burns backoff for hours and then fails anyway.
    """
    if getattr(exc, "status_code", None) != 429:
        return False
    msg = str(exc).lower()
    return (
        "per day" in msg
        or "requests per day" in msg
        or "tokens per day" in msg
        or "rpd" in msg
        or "tpd" in msg
    )


class JudgeResponse(BaseModel):
    """What every judge returns for one call."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


class Judge(ABC):
    """Abstract judge. Subclasses implement `_call_api`."""

    # Subclasses set these
    model_id: str = ""
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    min_seconds_between_calls: float = 0.0  # rate limit; 0 = no throttling

    use_cache: bool = True  # MockJudge disables this — see its docstring

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        temperature: float = 0.0,
        min_seconds_between_calls: float | None = None,
    ):
        """
        Args:
            cache_dir: where to store the judge-call disk cache.
            temperature: sampling temperature (part of the cache key).
            min_seconds_between_calls: override the class default throttle.
                Free tiers need the conservative class default; on a paid tier
                (Groq Developer allows ~1000 RPM) pass a much smaller value —
                e.g. 0.1 — or 0 to disable throttling entirely.
        """
        self.cache = None
        if self.use_cache:
            cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
            cache_path.mkdir(parents=True, exist_ok=True)
            self.cache = diskcache.Cache(str(cache_path))
        self.temperature = temperature
        if min_seconds_between_calls is not None:
            self.min_seconds_between_calls = min_seconds_between_calls
        self._limiter = _SharedRateLimiter.for_model(self.model_id)

    def _cache_key(self, prompt: str) -> str:
        payload = json.dumps(
            {"model": self.model_id, "prompt": prompt, "temp": self.temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def call(self, prompt: str) -> JudgeResponse:
        """Call the judge with caching and rate limiting."""
        key = self._cache_key(prompt)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                cached["cached"] = True
                cached["cost_usd"] = 0.0
                return JudgeResponse(**cached)

        # Rate limit: only applies to actual API calls, not cache hits.
        # Shared across all judges on this model — see _SharedRateLimiter.
        self._limiter.acquire(self.min_seconds_between_calls)

        response = self._call_api_with_retry(prompt)
        if self.cache is not None:
            self.cache.set(key, response.model_dump())
        return response

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_not_exception_type(_DailyQuotaError),
        reraise=True,
    )
    def _call_api_with_retry(self, prompt: str) -> JudgeResponse:
        return self._call_api(prompt)

    @abstractmethod
    def _call_api(self, prompt: str) -> JudgeResponse: ...

    def _compute_cost(self, in_tokens: int, out_tokens: int) -> float:
        return (
            in_tokens / 1_000_000 * self.cost_per_1m_input
            + out_tokens / 1_000_000 * self.cost_per_1m_output
        )


class GeminiJudge(Judge):
    """Google Gemini Flash judge. Free tier: 1500 RPD, 15 RPM (as of mid-2026).

    Uses gemini-2.5-flash. Set GEMINI_API_KEY env var.
    """

    model_id = "gemini-2.5-flash"
    cost_per_1m_input = 0.30
    cost_per_1m_output = 2.50
    min_seconds_between_calls = 13.0

    def __init__(self, cache_dir: Path | str | None = None, temperature: float = 0.0):
        super().__init__(cache_dir=cache_dir, temperature=temperature)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY (free at https://aistudio.google.com/apikey)")
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def _call_api(self, prompt: str) -> JudgeResponse:
        from google.genai import types

        result = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        text = result.text or ""
        usage = getattr(result, "usage_metadata", None)
        in_toks = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_toks = getattr(usage, "candidates_token_count", 0) if usage else 0
        return JudgeResponse(
            text=text,
            model=self.model_id,
            input_tokens=in_toks,
            output_tokens=out_toks,
            cost_usd=self._compute_cost(in_toks, out_toks),
        )


class GroqQuotaExhaustedError(_DailyQuotaError):
    """Raised when a Groq DAILY quota (RPD/TPD) is exhausted.

    Distinct from a per-minute 429, which is transient and worth retrying.
    A daily cap will not clear for hours, so retrying is pointless — callers
    should checkpoint and exit so the run can resume tomorrow.
    """


class GroqJudge(Judge):
    """Groq judge. Defaults to Llama 3.3 70B; pass `model` for another.

    Set GROQ_API_KEY env var.

    On the free tier the binding constraint is usually TOKENS PER DAY, not
    requests per minute. A common pattern is to run the answer *generator* on
    a high-quota model (llama-3.1-8b-instant) and reserve the 70B budget for
    *judging*, where model quality actually changes the numbers.
    """

    model_id = "llama-3.3-70b-versatile"
    cost_per_1m_input = 0.59
    cost_per_1m_output = 0.79
    min_seconds_between_calls = 2.5  # ~24 RPM, safely under 30 RPM Groq free-tier limit

    # Per-1M pricing for models we support explicitly; used for cost reporting.
    _PRICING = {
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant": (0.05, 0.08),
    }

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        temperature: float = 0.0,
        min_seconds_between_calls: float | None = None,
        model: str | None = None,
    ):
        if model:
            # Instance-level override; must be set before super().__init__ so the
            # shared rate limiter is keyed on the right model.
            self.model_id = model
            if model in self._PRICING:
                self.cost_per_1m_input, self.cost_per_1m_output = self._PRICING[model]
        super().__init__(
            cache_dir=cache_dir,
            temperature=temperature,
            min_seconds_between_calls=min_seconds_between_calls,
        )
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY (free at https://console.groq.com)")
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self.last_rate_limit_headers: dict[str, str] = {}

    def _call_api(self, prompt: str) -> JudgeResponse:
        try:
            raw = self._client.chat.completions.with_raw_response.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            self.last_rate_limit_headers = {
                k: v for k, v in raw.headers.items() if k.startswith("x-ratelimit")
            }
            response = raw.parse()
        except Exception as e:
            if _is_daily_quota_error(e):
                raise GroqQuotaExhaustedError(str(e)) from e
            raise
        text = response.choices[0].message.content or ""
        in_toks = response.usage.prompt_tokens if response.usage else 0
        out_toks = response.usage.completion_tokens if response.usage else 0
        return JudgeResponse(
            text=text,
            model=self.model_id,
            input_tokens=in_toks,
            output_tokens=out_toks,
            cost_usd=self._compute_cost(in_toks, out_toks),
        )


class MockJudge(Judge):
    """Deterministic judge for tests. Returns a canned response.

    Caching is DISABLED for mocks: all MockJudge instances share model_id
    "mock", so a shared disk cache would leak canned responses between
    tests configured with different response_text.
    """

    model_id = "mock"
    use_cache = False
    cost_per_1m_input = 0.0
    cost_per_1m_output = 0.0

    def __init__(self, response_text: str = '{"score": 4, "reasoning": "mock"}', **kw: Any):
        super().__init__(**kw)
        self.response_text = response_text

    def _call_api(self, prompt: str) -> JudgeResponse:
        return JudgeResponse(
            text=self.response_text,
            model=self.model_id,
            input_tokens=len(prompt) // 4,
            output_tokens=len(self.response_text) // 4,
            cost_usd=0.0,
        )
