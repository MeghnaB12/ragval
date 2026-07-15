"""Report YOUR account's actual Groq rate limits, straight from the API.

Usage:
    export GROQ_API_KEY=...
    python scripts/check_quota.py
    python scripts/check_quota.py --model llama-3.1-8b-instant

Published rate-limit tables go stale and vary by account. Groq returns the
real numbers in `x-ratelimit-*` response headers on every call. This makes
one tiny call per model and prints them, so benchmark planning starts from
ground truth instead of a blog post.

Costs ~20 tokens per model checked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragval.judges import GroqJudge  # noqa: E402

DEFAULT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def humanize(key: str) -> str:
    return {
        "x-ratelimit-limit-requests": "requests / day (RPD)",
        "x-ratelimit-remaining-requests": "  ...remaining today",
        "x-ratelimit-limit-tokens": "tokens / minute (TPM)",
        "x-ratelimit-remaining-tokens": "  ...remaining this min",
        "x-ratelimit-reset-requests": "requests reset in",
        "x-ratelimit-reset-tokens": "tokens reset in",
    }.get(key, key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=None)
    args = parser.parse_args()
    models = args.model or DEFAULT_MODELS

    for model in models:
        print(f"\n=== {model} ===")
        try:
            judge = GroqJudge(model=model, min_seconds_between_calls=0.0)
            # Bypass the cache: we want live headers, not a stored response.
            judge._call_api("hi")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}")
            continue

        headers = judge.last_rate_limit_headers
        if not headers:
            print("  (no x-ratelimit headers returned)")
            continue
        for k in sorted(headers):
            print(f"  {humanize(k):<28} {headers[k]}")

    print(
        "\nNote: Groq documents x-ratelimit-limit-requests as the DAILY cap and\n"
        "x-ratelimit-limit-tokens as the PER-MINUTE cap. A tokens-per-day cap may\n"
        "not appear in headers at all — watch for 429s mentioning 'per day'.\n"
        "\nPlan the benchmark against whichever limit is smallest:\n"
        "  python scripts/run_benchmark.py --estimate-only --n 150 --configs closed_book bm25_k3 oracle"
    )


if __name__ == "__main__":
    main()
