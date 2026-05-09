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
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ragval" / "judge"


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

    def __init__(self, cache_dir: Path | str | None = None, temperature: float = 0.0):
        cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(cache_path))
        self.temperature = temperature
        self._last_call_time = 0.0
        self._call_lock = threading.Lock()

    def _cache_key(self, prompt: str) -> str:
        payload = json.dumps(
            {"model": self.model_id, "prompt": prompt, "temp": self.temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def call(self, prompt: str) -> JudgeResponse:
        """Call the judge with caching and rate limiting."""
        key = self._cache_key(prompt)
        cached = self.cache.get(key)
        if cached is not None:
            cached["cached"] = True
            cached["cost_usd"] = 0.0
            return JudgeResponse(**cached)

        # Rate limit: only applies to actual API calls, not cache hits
        if self.min_seconds_between_calls > 0:
            with self._call_lock:
                elapsed = time.time() - self._last_call_time
                wait = self.min_seconds_between_calls - elapsed
                if wait > 0:
                    time.sleep(wait)
                self._last_call_time = time.time()

        response = self._call_api_with_retry(prompt)
        self.cache.set(key, response.model_dump())
        return response

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
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


class GroqJudge(Judge):
    """Groq Llama 3.3 70B judge. Free tier with generous rate limits.

    Set GROQ_API_KEY env var. Useful as a second judge for calibration.
    """

    model_id = "llama-3.3-70b-versatile"
    cost_per_1m_input = 0.59
    cost_per_1m_output = 0.79
    min_seconds_between_calls = 2.5  # ~24 RPM, safely under 30 RPM Groq free-tier limit

    def __init__(self, cache_dir: Path | str | None = None, temperature: float = 0.0):
        super().__init__(cache_dir=cache_dir, temperature=temperature)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY (free at https://console.groq.com)")
        from groq import Groq

        self._client = Groq(api_key=api_key)

    def _call_api(self, prompt: str) -> JudgeResponse:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
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
    """Deterministic judge for tests. Returns a canned response."""

    model_id = "mock"
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
