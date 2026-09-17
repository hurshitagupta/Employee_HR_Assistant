import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter


load_dotenv()


class AgentResponse(BaseModel):
    answer: str = Field(description="Final answer for the employee.")
    tools_used: list[str] = Field(description="Tools used to answer the question.")


# created a dummy retrieval tool
@tool
def search_company_docs(query: str) -> str:
    """Search company policies for employee-related questions."""

    policies = {
        "referral": (
            "The standard employee referral bonus is INR 20,000. "
            "It becomes payable after the referred candidate completes "
            "90 calendar days of continuous employment."
        ),
        "annual leave": (
            "Eligible full-time employees receive 18 days of paid "
            "annual leave for each completed year of service."
        ),
        "notice period": (
            "The standard notice period for confirmed full-time employees "
            "is 30 calendar days. Employees on probation normally have "
            "a 15-calendar-day notice period."
        ),
    }

    query_lower = query.lower()

    for topic, policy in policies.items():
        if topic in query_lower:
            return policy

    return "No relevant company policy was found."


# tool that calculates monthly salary after unpaid leave
@tool
def calculate_monthly_salary(annual_salary: float,unpaid_leave_days: int,) -> str:
    """
    Estimate gross monthly salary after unpaid leave.
    The company uses a 30-day payroll month for this calculation.
    """

    if annual_salary <= 0:
        raise ValueError("Annual salary must be greater than zero.")

    if unpaid_leave_days < 0 or unpaid_leave_days > 30:
        raise ValueError(
            "Unpaid leave days must be between 0 and 30."
        )

    monthly_salary = annual_salary / 12
    daily_salary = monthly_salary / 30
    deduction = daily_salary * unpaid_leave_days
    payable_salary = monthly_salary - deduction

    return (
        f"Gross monthly salary: INR {monthly_salary:,.2f}. "
        f"LOP deduction for {unpaid_leave_days} unpaid leave day(s): "
        f"INR {deduction:,.2f}. "
        f"Estimated payable gross salary: INR {payable_salary:,.2f}."
    )

model = ChatOpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0,
    timeout=30_000,
    max_retries=2,
)

agent = create_agent(
    model=model,
    tools=[search_company_docs,calculate_monthly_salary,],
    system_prompt=(
        "You are an internal Employee HR Assistant. "
        "Use search_company_docs for company-policy questions. "
        "Use calculate_monthly_salary for salary calculations involving annual salary and unpaid leave. "
        "Use both tools when a question requires both."
        "Do not invent company policies. "
        "Keep answers clear and concise."
    )
)

def run_agent(question: str) -> AgentResponse:

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    result = agent.invoke({
            "messages": [
                {"role": "user","content": question}
            ]
        },
        config={ "recursion_limit": 6}
    )

    final_answer = result["messages"][-1].content

    tools_used = []

    for message in result["messages"]:
        if hasattr(message, "name") and message.name:
            tools_used.append(message.name)

    return AgentResponse(
        answer=final_answer,
        tools_used=tools_used,
    )


if __name__ == "__main__":

    question = (
        "What is the employee referral policy? "
        "Also, my annual salary is INR 1200000 and I took 2 unpaid leave days this month."
        "What will be my estimated salary?"
    )

    response = run_agent(question)

    print("\n=== EMPLOYEE HR ASSISTANT ===")
    print(f"Answer: {response.answer}")
    print(f"Tool used: {response.tools_used}")