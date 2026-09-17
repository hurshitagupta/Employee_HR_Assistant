# Employee HR Assistant

An AI-powered employee support assistant built with LangChain that helps employees retrieve company policy information and perform salary estimations.

The project combines tool-calling agents, grounded document retrieval, structured validation, streaming, observability, evaluation, guardrails, and deployment into a single employee-facing service.

## Project Overview

The Employee HR Assistant is designed to provide employees with a single interface for common HR-related queries.

The assistant can determine which capability is required based on the employee's question. It currently supports:

- Searching company policy information through the `search_company_docs` tool.
- Estimating monthly gross salary after unpaid leave through the `calculate_monthly_salary` tool.
- Using multiple tools when a question requires more than one capability.
- Returning responses through a validated Pydantic output schema.
- Limiting agent execution to prevent uncontrolled execution loops.
- Model timeout and retry handling for improved reliability.

The project uses a fictional Employee Handbook for **Northstar Digital Solutions Pvt. Ltd.** as its company knowledge corpus. The handbook is included only as synthetic project data and does not represent the policies of a real organization.

---

## Agent Service

The agent service is the decision-making layer of the Employee HR Assistant.

Instead of manually routing each employee query through `if/else` conditions, a LangChain tool-calling agent determines which available tool is appropriate for the request.

### Available Tools

#### `search_company_docs`

Handles employee questions related to company policies.

Example:

```text
What is the employee referral policy?
```

The agent recognizes this as a policy-related request and calls `search_company_docs`.

The initial agent implementation uses a small policy lookup to verify tool selection. This tool is designed so that its internal implementation can be replaced by the document retrieval pipeline without changing the agent interface.

#### `calculate_monthly_salary`

Calculates estimated gross monthly salary after unpaid leave.

The calculation follows:

```text
Monthly Salary = Annual Salary / 12

Daily Salary = Monthly Salary / 30

LOP Deduction = Daily Salary × Unpaid Leave Days

Estimated Payable Salary = Monthly Salary - LOP Deduction
```

This is an estimate of gross salary and does not include tax, provident fund, benefits, bonuses, or other payroll adjustments.

---

## Agent Flow

```text
Employee Question
        |
        v
Tool-Calling Agent
        |
        +----------------------------+
        |                            |
        v                            v
search_company_docs       calculate_monthly_salary
        |                            |
        v                            v
Company Policy             Salary Calculation
        |                            |
        +-------------+--------------+
                      |
                      v
                 Final Answer
                      |
                      v
              Pydantic Validation
                      |
                      v
               Employee Response
```

For a combined question such as:

```text
What is the employee referral policy?
My annual salary is INR 1,200,000 and I took
2 unpaid leave days. What will be my estimated salary?
```

the agent can use both available tools before producing the final response.

---

## Output Validation

The service uses a Pydantic model to define its response contract:

```python
class AgentResponse(BaseModel):
    answer: str
    tools_used: list[str]
```

The final service response is constructed through this schema before being returned to the caller.

This ensures that the application always returns the expected fields and that invalid output structures raise a validation error rather than being silently accepted.

---

## Reliability Controls

The agent service includes basic reliability controls:

- **Step limit** — restricts agent execution to prevent uncontrolled loops.
- **Timeout** — prevents a model request from waiting indefinitely.
- **Retry** — retries transient model failures with a capped number of attempts.
- **Input validation** — rejects empty employee questions.
- **Output validation** — validates the final response through Pydantic.
- **Secret hygiene** — API credentials are loaded from environment variables rather than stored in source code.

---

## Run the Agent

From the project root:

```bash
uv run python -m agent_service.agent_service
```

The example request requires both company-policy lookup and salary calculation, allowing the agent's tool-selection behaviour to be observed.

---

## Automated Tests

Run the agent-service tests with:

```bash
uv run pytest tests/test_agent_service.py -v
```

The tests verify:

- Successful agent execution.
- Selection of both required tools for a combined query.
- Correct deterministic salary calculation.
- Rejection of an empty employee question.
- Rejection of output that does not satisfy the Pydantic response schema.

---