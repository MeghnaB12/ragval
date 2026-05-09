"""Metric implementations.

Each metric is a callable that takes (EvalSample, RagOutput, Judge) and returns
a MetricResult. Metrics use explicit rubrics in their judge prompts and return
JSON, which we parse defensively.

Week 1: faithfulness, answer_relevance.
Week 2: context_precision, context_recall.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from ragval.judges import Judge
from ragval.types import EvalSample, MetricResult, RagOutput


def _parse_json_score(text: str) -> tuple[float | None, str]:
    """Extract {score, reasoning} from judge text. Tolerant of code fences and noise.

    Returns (score, reasoning). Score is None if parsing fails.
    """
    # Strip code fences if present
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    # Find the first JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None, text
    try:
        data = json.loads(match.group(0))
        score = data.get("score")
        reasoning = data.get("reasoning", "")
        if isinstance(score, (int, float)):
            return float(score), str(reasoning)
    except json.JSONDecodeError:
        pass
    return None, text


class Metric(ABC):
    """Base class for metrics."""

    name: str = ""

    @abstractmethod
    def score(self, sample: EvalSample, output: RagOutput, judge: Judge) -> MetricResult: ...


class Faithfulness(Metric):
    """Is the answer grounded in the retrieved contexts?

    Score 1-5:
        5 = every claim in the answer is directly supported by the contexts
        3 = mixed; some claims supported, some not
        1 = answer contradicts contexts or is entirely made up
    """

    name = "faithfulness"

    PROMPT = """You are an expert evaluator scoring whether an answer is faithful to its supporting contexts.

QUESTION: {question}

CONTEXTS:
{contexts}

ANSWER: {answer}

Rate faithfulness on a scale of 1-5:
5 = Every factual claim in the answer is directly supported by the contexts.
4 = Almost all claims supported; minor unsupported details.
3 = Mixed. Some claims supported, others not directly stated in contexts.
2 = Most claims are not supported by the contexts.
1 = The answer contradicts the contexts or fabricates facts not present.

Respond ONLY with valid JSON in this exact format:
{{"score": <integer 1-5>, "reasoning": "<one or two sentence explanation>"}}"""

    def score(self, sample: EvalSample, output: RagOutput, judge: Judge) -> MetricResult:
        contexts_str = "\n\n".join(
            f"[{i + 1}] {c}" for i, c in enumerate(output.retrieved_contexts)
        )
        prompt = self.PROMPT.format(
            question=sample.question,
            contexts=contexts_str if contexts_str else "(none)",
            answer=output.answer,
        )
        response = judge.call(prompt)
        raw, reasoning = _parse_json_score(response.text)
        if raw is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                raw_score=None,
                reasoning=f"PARSE_FAILURE: {response.text[:200]}",
                judge_model=response.model,
                cost_usd=response.cost_usd,
            )
        # Normalize 1-5 to 0-1
        normalized = (raw - 1) / 4
        return MetricResult(
            metric_name=self.name,
            score=max(0.0, min(1.0, normalized)),
            raw_score=raw,
            reasoning=reasoning,
            judge_model=response.model,
            cost_usd=response.cost_usd,
        )


class AnswerRelevance(Metric):
    """Does the answer actually address the question?

    Score 1-5:
        5 = directly and completely answers the question
        3 = partially relevant, drifts or hedges
        1 = does not answer the question
    """

    name = "answer_relevance"

    PROMPT = """You are an expert evaluator scoring how well an answer addresses a question.

QUESTION: {question}

ANSWER: {answer}

Rate answer relevance on a scale of 1-5:
5 = Directly and completely answers the question.
4 = Answers the question with minor gaps or extra information.
3 = Partially relevant. Drifts off-topic or hedges significantly.
2 = Mostly irrelevant or evasive.
1 = Does not answer the question at all.

This metric only judges relevance to the question — NOT factual correctness or grounding.

Respond ONLY with valid JSON in this exact format:
{{"score": <integer 1-5>, "reasoning": "<one or two sentence explanation>"}}"""

    def score(self, sample: EvalSample, output: RagOutput, judge: Judge) -> MetricResult:
        prompt = self.PROMPT.format(
            question=sample.question,
            answer=output.answer,
        )
        response = judge.call(prompt)
        raw, reasoning = _parse_json_score(response.text)
        if raw is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                raw_score=None,
                reasoning=f"PARSE_FAILURE: {response.text[:200]}",
                judge_model=response.model,
                cost_usd=response.cost_usd,
            )
        normalized = (raw - 1) / 4
        return MetricResult(
            metric_name=self.name,
            score=max(0.0, min(1.0, normalized)),
            raw_score=raw,
            reasoning=reasoning,
            judge_model=response.model,
            cost_usd=response.cost_usd,
        )


class AnswerCorrectness(Metric):
    """Does the answer match the ground truth?

    Unlike faithfulness (vs. context) and answer_relevance (vs. question),
    this metric compares the system's answer to a known correct answer.
    It tolerates:
      - Different verbosity (truth: "Mt. Everest"; answer: "Mount Everest is...")
      - Different forms of the same entity ("Jorge Amado" vs "Jorge Leal Amado de Faria")
      - Equivalent phrasings ("yes" vs "they are the same")

    It does NOT tolerate:
      - Wrong entity / wrong fact
      - Refusal to answer ("the context doesn't say")
      - Partial answers that miss the key fact

    Score 1-5:
        5 = answer fully and correctly contains the ground truth fact
        4 = correct but with minor missing detail or extra unsupported info
        3 = partially correct (e.g. one of two required facts present)
        2 = mostly wrong but with some overlap
        1 = wrong, refused, or unrelated to the ground truth
    """

    name = "answer_correctness"

    PROMPT = """You are an expert evaluator scoring the correctness of an answer against a known ground truth.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

SYSTEM'S ANSWER: {answer}

Rate correctness on a scale of 1-5:
5 = The system's answer fully and correctly contains the ground truth fact. Verbose answers and equivalent phrasings (e.g. "Jorge Amado" vs "Jorge Leal Amado de Faria") still score 5.
4 = Correct, but with a minor missing detail or extra unsupported information.
3 = Partially correct. For example, one of two required facts is present, or the entity is right but a date is wrong.
2 = Mostly wrong, but with some genuine overlap with the ground truth.
1 = Wrong, refused to answer, or unrelated to the ground truth. "The context does not contain..." also scores 1 — refusal is not correctness.

Be strict about factual content. Be lenient about phrasing, verbosity, and equivalent forms.

Respond ONLY with valid JSON in this exact format:
{{"score": <integer 1-5>, "reasoning": "<one or two sentence explanation>"}}"""

    def score(self, sample: EvalSample, output: RagOutput, judge: Judge) -> MetricResult:
        prompt = self.PROMPT.format(
            question=sample.question,
            ground_truth=sample.ground_truth_answer,
            answer=output.answer,
        )
        response = judge.call(prompt)
        raw, reasoning = _parse_json_score(response.text)
        if raw is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                raw_score=None,
                reasoning=f"PARSE_FAILURE: {response.text[:200]}",
                judge_model=response.model,
                cost_usd=response.cost_usd,
            )
        normalized = (raw - 1) / 4
        return MetricResult(
            metric_name=self.name,
            score=max(0.0, min(1.0, normalized)),
            raw_score=raw,
            reasoning=reasoning,
            judge_model=response.model,
            cost_usd=response.cost_usd,
        )


# Registry — used by CLI and runner to look up metrics by name.
METRIC_REGISTRY: dict[str, type[Metric]] = {
    Faithfulness.name: Faithfulness,
    AnswerRelevance.name: AnswerRelevance,
    AnswerCorrectness.name: AnswerCorrectness,
}
