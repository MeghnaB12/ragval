import json
import tempfile

from ragval.judges import MockJudge


def test_mock_judge_returns_response():
    j = MockJudge(cache_dir=tempfile.mkdtemp())
    r = j.call("anything")
    assert r.text
    assert r.model == "mock"


class _CachedMockJudge(MockJudge):
    """Caching-enabled mock, used only to test the Judge cache path."""

    use_cache = True


def test_judge_caches():
    cache_dir = tempfile.mkdtemp()
    j = _CachedMockJudge(cache_dir=cache_dir)
    r1 = j.call("hello")
    r2 = j.call("hello")
    assert r1.cached is False
    assert r2.cached is True
    assert r2.cost_usd == 0.0


def test_judge_cache_key_differs_by_prompt():
    cache_dir = tempfile.mkdtemp()
    j = _CachedMockJudge(cache_dir=cache_dir)
    j.call("a")
    r = j.call("b")
    assert r.cached is False


def test_mock_judge_returns_canned_json():
    payload = '{"score": 5, "reasoning": "great"}'
    j = MockJudge(cache_dir=tempfile.mkdtemp(), response_text=payload)
    r = j.call("anything")
    parsed = json.loads(r.text)
    assert parsed["score"] == 5


def test_mock_judge_does_not_cache():
    """Regression: two mocks with the same prompt but different canned
    responses must not leak responses through a shared cache."""
    j1 = MockJudge(response_text='{"score": 4, "reasoning": "a"}')
    j2 = MockJudge(response_text='{"score": 1, "reasoning": "b"}')
    j1.call("same prompt")
    r = j2.call("same prompt")
    assert '"score": 1' in r.text


def test_rate_limiter_shared_across_instances_of_same_model():
    """Regression: generator + judge on the same model must share a throttle,
    or their combined request rate exceeds the provider's per-model cap."""
    import ragval.judges as judges_mod

    a = judges_mod._SharedRateLimiter.for_model("shared-test-model")
    b = judges_mod._SharedRateLimiter.for_model("shared-test-model")
    c = judges_mod._SharedRateLimiter.for_model("other-model")
    assert a is b
    assert a is not c


def test_rate_limiter_enforces_gap():
    import time

    import ragval.judges as judges_mod

    limiter = judges_mod._SharedRateLimiter()
    start = time.time()
    limiter.acquire(0.0)  # first call, no wait
    limiter.acquire(0.15)
    limiter.acquire(0.15)
    assert time.time() - start >= 0.15


def test_min_seconds_override():
    """Paid tiers need a smaller throttle than the free-tier class default."""
    j = MockJudge(min_seconds_between_calls=0.0)
    assert j.min_seconds_between_calls == 0.0
