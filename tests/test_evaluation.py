from deployment_and_evaluation.evaluation import (
    EVALUATION_CASES,
    evaluate_case,
)


def test_evaluation_has_30_cases():
    """Evaluation dataset must contain exactly 30 cases."""

    assert len(EVALUATION_CASES) == 30


def test_supported_evaluation_case():
    """A known handbook question should pass evaluation."""

    case = {
        "question": (
            "How many paid annual leave days "
            "do full-time employees receive?"
        ),
        "expected": "18",
        "should_refuse": False,
    }

    result = evaluate_case(case)

    assert result["passed"] is True
    assert result["refused"] is False
    assert "18" in result["actual"]


def test_refusal_evaluation_case():
    """An unsupported handbook question should pass by refusing."""

    case = {
        "question": "Does the company provide free gym membership?",
        "expected": None,
        "should_refuse": True,
    }

    result = evaluate_case(case)

    assert result["passed"] is True
    assert result["refused"] is True


def test_dataset_contains_supported_and_refusal_cases():
    """Evaluation should cover both grounded answers and refusals."""

    supported = [
        case
        for case in EVALUATION_CASES
        if case["should_refuse"] is False
    ]

    refusals = [
        case
        for case in EVALUATION_CASES
        if case["should_refuse"] is True
    ]

    assert len(supported) > 0
    assert len(refusals) > 0