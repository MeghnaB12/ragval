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


class ContextPrecision(Metric):
    """What fraction of retrieved contexts are actually relevant to the question?

    Judge-based, per-context: the judge labels each retrieved context as
    relevant (contains information needed to answer) or not. The score is
    the fraction relevant. One judge call per output, batched into a single
    prompt to keep costs sane.
    """

    name = "context_precision"

    PROMPT = """You are an expert evaluator judging retrieval quality.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

RETRIEVED CONTEXTS:
{contexts}

For EACH context, decide whether it is RELEVANT: does it contain information \
needed to answer the question (fully or partially)? A context about the right \
entity but the wrong fact is NOT relevant.

Respond ONLY with valid JSON in this exact format:
{{"relevant": [<true/false for context 1>, <true/false for context 2>, ...], \
"reasoning": "<one sentence>"}}
The "relevant" list must have exactly {n} entries."""

    def score(self, sample: EvalSample, output: RagOutput, judge: Judge) -> MetricResult:
        contexts = output.retrieved_contexts
        if not contexts:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                raw_score=0.0,
                reasoning="No contexts retrieved.",
                judge_model=judge.model_id,
            )
        contexts_str = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        prompt = self.PROMPT.format(
            question=sample.question,
            ground_truth=sample.ground_truth_answer,
            contexts=contexts_str,
            n=len(contexts),
        )
        response = judge.call(prompt)
        flags, reasoning = _parse_json_bool_list(response.text, "relevant")
        if flags is None or len(flags) != len(contexts):
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                raw_score=None,
                reasoning=f"PARSE_FAILURE: {response.text[:200]}",
                judge_model=response.model,
                cost_usd=response.cost_usd,
            )
        precision = sum(flags) / len(flags)
        return MetricResult(
            metric_name=self.name,
            score=precision,
            raw_score=precision,
            reasoning=reasoning,
            judge_model=response.model,
            cost_usd=response.cost_usd,
            metadata={"relevant_flags": flags},
        )


class ContextRecall(Metric):
    """Do the retrieved contexts contain everything needed to produce the
    ground truth answer?

    Judge-based: the judge decomposes the ground truth answer into its
    required facts and checks whether each is attributable to the retrieved
    contexts. Score = fraction of required facts covered.
    """

    name = "context_recall"

    PROMPT = """You are an expert evaluator judging retrieval completeness.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

RETRIEVED CONTEXTS:
{contexts}

Step 1: Identify the fact(s) required to state the ground truth answer \
(often 1-2 facts; multi-hop questions require 2+).
Step 2: For each required fact, decide whether it is present in the \
retrieved contexts.

Respond ONLY with valid JSON in this exact format:
{{"covered": [<true/false for fact 1>, ...], "reasoning": "<the facts and which are covered>"}}"""

    def score(self, sample: EvalSample, output: RagOutput, judge: Judge) -> MetricResult:
        contexts = output.retrieved_contexts
        contexts_str = (
            "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) if contexts else "(none)"
        )
        prompt = self.PROMPT.format(
            question=sample.question,
            ground_truth=sample.ground_truth_answer,
            contexts=contexts_str,
        )
        response = judge.call(prompt)
        flags, reasoning = _parse_json_bool_list(response.text, "covered")
        if flags is None or not flags:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                raw_score=None,
                reasoning=f"PARSE_FAILURE: {response.text[:200]}",
                judge_model=response.model,
                cost_usd=response.cost_usd,
            )
        recall = sum(flags) / len(flags)
        return MetricResult(
            metric_name=self.name,
            score=recall,
            raw_score=recall,
            reasoning=reasoning,
            judge_model=response.model,
            cost_usd=response.cost_usd,
            metadata={"covered_flags": flags},
        )


def _matches_gold_title(context: str, gold_titles: set[str]) -> bool:
    """Retrieved contexts are formatted 'Title: text'. A context matches a
    gold title if it starts with that title followed by a colon.

    Prefix matching (rather than splitting on ':') is deliberate: Wikipedia
    titles can themselves contain colons ("Star Trek: First Contact").
    """
    return any(context.startswith(f"{t}:") for t in gold_titles)


class RetrievalRecall(Metric):
    """Deterministic (judge-free) retrieval recall for datasets with gold
    supporting documents, like HotpotQA.

    Compares the *titles* of retrieved contexts against the sample's known
    supporting titles (`metadata.supporting_titles`, falling back to
    `ground_truth_contexts`). Free, exact, and reproducible — use this
    alongside the judge-based ContextRecall to sanity-check the judge.
    """

    name = "retrieval_recall"

    def score(self, sample: EvalSample, output: RagOutput, judge: Judge) -> MetricResult:
        gold = set(sample.metadata.get("supporting_titles") or sample.ground_truth_contexts)
        if not gold:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                raw_score=None,
                reasoning="No gold supporting titles on sample.",
            )
        hit = sum(1 for t in gold if any(c.startswith(f"{t}:") for c in output.retrieved_contexts))
        recall = hit / len(gold)
        return MetricResult(
            metric_name=self.name,
            score=recall,
            raw_score=recall,
            reasoning=f"{hit}/{len(gold)} gold titles retrieved.",
            judge_model="deterministic",
        )


class RetrievalPrecision(Metric):
    """Deterministic retrieval precision against gold supporting titles."""

    name = "retrieval_precision"

    def score(self, sample: EvalSample, output: RagOutput, judge: Judge) -> MetricResult:
        gold = set(sample.metadata.get("supporting_titles") or sample.ground_truth_contexts)
        if not output.retrieved_contexts:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                raw_score=0.0,
                reasoning="No contexts retrieved.",
                judge_model="deterministic",
            )
        hit = sum(1 for c in output.retrieved_contexts if _matches_gold_title(c, gold))
        precision = hit / len(output.retrieved_contexts)
        return MetricResult(
            metric_name=self.name,
            score=precision,
            raw_score=precision,
            reasoning=f"{hit}/{len(output.retrieved_contexts)} retrieved contexts are gold.",
            judge_model="deterministic",
        )


def _parse_json_bool_list(text: str, key: str) -> tuple[list[bool] | None, str]:
    """Extract {key: [bools], reasoning} from judge text. Tolerant of fences."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None, text
    try:
        data = json.loads(match.group(0))
        flags = data.get(key)
        reasoning = str(data.get("reasoning", ""))
        if isinstance(flags, list) and all(isinstance(f, bool) for f in flags):
            return flags, reasoning
    except json.JSONDecodeError:
        pass
    return None, text


# Registry — used by CLI and runner to look up metrics by name.
METRIC_REGISTRY: dict[str, type[Metric]] = {
    Faithfulness.name: Faithfulness,
    AnswerRelevance.name: AnswerRelevance,
    AnswerCorrectness.name: AnswerCorrectness,
    ContextPrecision.name: ContextPrecision,
    ContextRecall.name: ContextRecall,
    RetrievalRecall.name: RetrievalRecall,
    RetrievalPrecision.name: RetrievalPrecision,
}
