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

## Grounded Knowledge

The Employee HR Assistant uses Retrieval-Augmented Generation (RAG) to answer questions from the Employee Handbook rather than relying on the model's general knowledge.

The knowledge pipeline uses the fictional Employee Handbook stored at:

```text
data/employee_handbook.pdf
```

### Retrieval Pipeline

```text
Employee Handbook PDF
        |
        v
    PyPDFLoader
        |
        v
     Chunking
        |
        v
Local Hugging Face Embeddings
        |
        v
      FAISS
        |
        v
     Retriever
        |
        v
Relevant Handbook Chunks
        |
        v
Grounded LLM Response
```

The PDF is loaded using `PyPDFLoader` and divided into smaller chunks using `RecursiveCharacterTextSplitter`.

The current chunk configuration is:

```text
Chunk size: 1000 characters
Chunk overlap: 150 characters
```

The chunks are converted into vector embeddings using the local `sentence-transformers/all-MiniLM-L6-v2` embedding model and indexed using FAISS.

For each employee question, the retriever returns the most semantically relevant handbook chunks instead of sending the complete handbook to the language model.

### Grounded Answer Generation

The language model receives the employee question together with the retrieved handbook context.

It is explicitly instructed to:

- Answer only from the supplied handbook context.
- Avoid using outside knowledge for company policies.
- Avoid inventing policies that are not documented.
- Include the supporting handbook page citation.
- Refuse to answer when the retrieved context does not contain sufficient information.

Example supported question:

```text
How many annual leave days do eligible full-time employees receive?
```

Example response:

```text
Eligible full-time employees receive 18 days of paid annual
leave for each completed year of service. (Page 5)
```

The page information originates from the metadata attached to the PDF documents during retrieval.

### Grounded Refusal

Vector similarity search returns the closest available chunks even when the handbook does not contain an answer to the employee's question.

For this reason, retrieval alone is not treated as proof that an answer exists.

The retrieved context is checked during grounded answer generation. If the context does not provide enough information, the service returns a refusal instead of generating an unsupported company policy.

Example unsupported question:

```text
Does the company provide employees with a free gym membership?
```

Response:

```text
The requested information could not be found in the Employee Handbook.
```

The structured response records this state as:

```text
Citations: []
Refused: True
```

This prevents the assistant from presenting unsupported information as an official company policy.

### Grounded Response Schema

Grounded responses are validated using Pydantic:

```python
class GroundedResponse(BaseModel):
    answer: str
    citations: list[str]
    refused: bool
```

A successful grounded response contains the answer and its supporting page citation.

An unsupported request returns an empty citation list and sets `refused` to `True`.

### Run Grounded Knowledge

From the project root:

```bash
uv run python -m grounded_knowledge.grounded_knowledge
```

### Automated Tests

Run:

```bash
uv run pytest tests/test_grounded_knowledge.py -v
```

The automated tests verify:

- Successful retrieval of a known handbook policy.
- Correct policy information from the retrieved context.
- Presence of the supporting page citation.
- Refusal for information that is not present in the handbook.
- Absence of fabricated citations on refused responses.
- Rejection of empty questions.

---

## Streaming UX

The Employee HR Assistant supports real-time response streaming using Server-Sent Events (SSE).

Instead of waiting for retrieval and answer generation to fully complete before returning a response, the service progressively sends intermediate execution steps and generated model chunks to the client.


### Server-Sent Events

The `/stream` FastAPI route returns a `StreamingResponse` with:

```text
Content-Type: text/event-stream
```

Events are formatted using the SSE protocol.

Example intermediate event:

```text
event: step
data: {"name": "retrieval", "status": "started"}
```

Example streamed model event:

```text
event: token
data: {"content": "Eligible"}
```

Example completion event:

```text
event: done
data: {"message": "Stream completed."}
```

Each SSE event is separated by a blank line so that connected clients can process the events individually.

### Intermediate Steps

The service exposes important execution stages instead of streaming only the final answer.

The current stream includes:

```text
retrieval - started
retrieval - finished
model - started
token events...
model - finished
done
```

This allows a client to observe the progress of the request while the final answer is still being generated.

### Model Streaming

Model output is generated using asynchronous streaming rather than a standard blocking model invocation.

```python
async for chunk in model.astream(prompt):
```

Each available model chunk is immediately converted into an SSE `token` event and sent to the connected client.

This reduces the perceived waiting time because the employee can begin receiving the response before the complete answer has been generated.

### Streaming Endpoint

Start the FastAPI service from the project root:

```bash
uv run uvicorn streaming_ux.streaming_ux:app
```

The streaming endpoint is:

```text
GET /stream
```

It accepts an employee question through the `question` query parameter.

FastAPI also exposes automatically generated API documentation at:

```text
http://127.0.0.1:8000/docs
```

### Streaming Error Handling

Invalid requests and runtime failures are also represented as SSE events.

For example, an empty question produces:

```text
event: error
data: {"message": "Question cannot be empty."}
```

If retrieval or model streaming fails, an error event is emitted and the stream is stopped rather than continuing with an incomplete response.

### Automated Tests

Run the streaming tests with:

```bash
uv run pytest tests/test_streaming_ux.py -v
```

The tests verify:

- Correct SSE event formatting.
- Successful intermediate-step streaming.
- Presence of retrieval and model execution events.
- Streaming of generated model chunks.
- Successful stream completion.
- Error handling for an empty question.