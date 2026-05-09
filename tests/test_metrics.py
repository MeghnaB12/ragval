import tempfile

from ragval.judges import MockJudge
from ragval.metrics import AnswerRelevance, Faithfulness, _parse_json_score
from ragval.types import EvalSample, RagOutput


def _sample():
    return EvalSample(
        id="t1",
        question="What color is the sky?",
        ground_truth_answer="Blue",
        ground_truth_contexts=["The sky appears blue due to Rayleigh scattering."],
    )


def _output():
    return RagOutput(
        answer="The sky is blue because of Rayleigh scattering.",
        retrieved_contexts=["The sky appears blue due to Rayleigh scattering."],
    )


def test_parse_json_score_plain():
    s, r = _parse_json_score('{"score": 4, "reasoning": "good"}')
    assert s == 4.0
    assert r == "good"


def test_parse_json_score_with_code_fence():
    s, r = _parse_json_score('```json\n{"score": 5, "reasoning": "x"}\n```')
    assert s == 5.0


def test_parse_json_score_with_preamble():
    s, _ = _parse_json_score('Sure! Here is my judgement:\n{"score": 2, "reasoning": "meh"}')
    assert s == 2.0


def test_parse_json_score_garbage():
    s, _ = _parse_json_score("totally not json")
    assert s is None


def test_faithfulness_normalizes():
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text='{"score": 5, "reasoning": "supported"}',
    )
    result = Faithfulness().score(_sample(), _output(), judge)
    assert result.metric_name == "faithfulness"
    assert result.score == 1.0  # raw 5 -> 1.0
    assert result.raw_score == 5.0


def test_faithfulness_low_score():
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text='{"score": 1, "reasoning": "fabricated"}',
    )
    result = Faithfulness().score(_sample(), _output(), judge)
    assert result.score == 0.0


def test_answer_relevance():
    judge = MockJudge(
        cache_dir=tempfile.mkdtemp(),
        response_text='{"score": 3, "reasoning": "partial"}',
    )
    result = AnswerRelevance().score(_sample(), _output(), judge)
    assert result.metric_name == "answer_relevance"
    assert result.score == 0.5  # raw 3 -> 0.5


def test_metric_handles_parse_failure():
    judge = MockJudge(cache_dir=tempfile.mkdtemp(), response_text="garbage")
    result = Faithfulness().score(_sample(), _output(), judge)
    assert result.score == 0.0
    assert "PARSE_FAILURE" in result.reasoning
