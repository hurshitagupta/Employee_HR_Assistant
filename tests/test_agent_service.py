from pydantic import ValidationError
import pytest

from agent_service.agent_service import (
    AgentResponse,
    calculate_monthly_salary,
    run_agent,
)


def test_agent_success():
    """Agent should answer the request and use the required tools."""

    response = run_agent(
        "What is the employee referral policy? "
        "Also, my annual salary is INR 1200000 and I took "
        "2 unpaid leave days this month. "
        "What will be my estimated salary?"
    )

    assert isinstance(response, AgentResponse)

    assert "search_company_docs" in response.tools_used
    assert "calculate_monthly_salary" in response.tools_used

    assert response.answer.strip()


def test_salary_tool():
    """Salary tool should correctly calculate salary after unpaid leave."""

    result = calculate_monthly_salary.invoke(
        {
            "annual_salary": 1200000,
            "unpaid_leave_days": 2,
        }
    )

    assert "93,333.33" in result


def test_empty_question_failure():
    """Agent should reject an empty question."""

    with pytest.raises(ValueError, match="Question cannot be empty"):
        run_agent("")


def test_invalid_output_validation():
    """Invalid service output should fail Pydantic validation."""

    with pytest.raises(ValidationError):
        AgentResponse(
            answer="Example answer",
            tools_used="search_company_docs",
        )