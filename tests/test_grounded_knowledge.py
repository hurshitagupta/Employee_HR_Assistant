import pytest

from grounded_knowledge.grounded_knowledge import GroundedResponse, answer_question, retrieve_context


def test_grounded_answer():
    """Known handbook question should return a grounded answer with citation."""

    response = answer_question(
        "How many annual leave days do eligible full-time employees receive?"
    )

    assert isinstance(response, GroundedResponse)
    assert "18" in response.answer
    assert "[Page 5]" in response.citations
    assert response.refused is False


def test_unsupported_question_refusal():
    """Unsupported policy question should be refused."""

    response = answer_question(
        "Does the company provide employees with a free gym membership?"
    )

    assert isinstance(response, GroundedResponse)
    assert response.refused is True
    assert response.citations == []
    assert "could not be found" in response.answer.lower()


def test_empty_question_failure():
    """Empty questions should be rejected."""

    with pytest.raises(ValueError,match="Question cannot be empty"):
        retrieve_context("")