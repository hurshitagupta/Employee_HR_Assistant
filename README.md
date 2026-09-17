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

---

## Observability

The Employee HR Assistant includes an observability layer for tracking individual requests, application behavior, performance, refusals, and failures.

The implementation provides:

- Correlation IDs
- Structured JSON logs
- Prometheus-compatible metrics
- Request latency measurement
- Grounded refusal tracking
- Error tracking
- High-latency alerting


### Correlation IDs

Every request receives a unique UUID correlation ID.

Example:

```text
a17dc38c-7c76-4a87-a76d-84ad719bf690
```

The same correlation ID is included in the structured events generated during that request.

This makes it possible to associate request start, completion, refusal, failure, and alert events with the same execution.

### Structured Logging

Application events are recorded as JSON rather than unstructured print statements.

Example request event:

```json
{
    "event": "request_started",
    "correlation_id": "a17dc38c-7c76-4a87-a76d-84ad719bf690",
    "question": "How many annual leave days do eligible full-time employees receive?"
}
```

Example completion event:

```json
{
    "event": "request_completed",
    "correlation_id": "a17dc38c-7c76-4a87-a76d-84ad719bf690",
    "latency_seconds": 2.831,
    "refused": false,
    "citations": ["[Page 5]"]
}
```

Structured logs make application events easier to search, filter, and process using monitoring systems.

### Application Metrics

Prometheus-compatible metrics are collected using `prometheus-client`.

The service currently records:

| Metric | Purpose |
| --- | --- |
| `hr_assistant_requests_total` | Total HR Assistant requests |
| `hr_assistant_errors_total` | Total failed requests |
| `hr_assistant_refusals_total` | Questions refused because the handbook did not contain sufficient information |
| `hr_assistant_request_latency_seconds` | Distribution of request processing latency |

A request increments:

```text
hr_assistant_requests_total
```

A failed request increments:

```text
hr_assistant_errors_total
```

An unsupported handbook question increments:

```text
hr_assistant_refusals_total
```

Request duration is recorded in the latency histogram.

### Refusal Monitoring

Grounded refusals are monitored separately from application errors.

For example:

```text
Does the company provide employees with a free gym membership?
```

is not considered a system failure.

The application successfully processes the request, but the Employee Handbook does not contain sufficient information to answer it.

The refusal counter allows these cases to be monitored independently:

```text
hr_assistant_refusals_total
```

This distinction separates unsupported knowledge requests from actual application failures.

### High-Latency Alert

The observability layer includes a high-latency alert.

The current threshold is:

```text
5 seconds
```

If a request exceeds the threshold, the application emits a structured alert event:

```json
{
    "event": "high_latency_alert",
    "correlation_id": "example-id",
    "latency_seconds": 6.0,
    "threshold_seconds": 5.0
}
```

The alert logic is tested directly with both high and normal latency values.

This allows the alert behavior to be verified without intentionally slowing down real model requests.

### Run Observability

From the project root:

```bash
uv run python -m observability.observability
```

The output includes:

- Structured request logs
- Correlation ID
- Grounded answer
- Request latency
- Prometheus metrics
- Alert event when the configured condition is exceeded

### Automated Tests

Run:

```bash
uv run pytest tests/test_observability.py -v
```

The tests verify:

- Successful observable request processing
- Correlation ID generation
- Grounded refusal handling
- Prometheus metric exposure
- High-latency alert triggering
- Normal-latency behavior
- Application failure handling

---

## Deployment

The Employee HR Assistant is containerized using Docker so that the application and its dependencies can run consistently outside the local development environment.

The deployed container exposes the existing FastAPI streaming service rather than introducing a separate deployment-specific application.

### Docker Architecture

```text
Employee HR Assistant
        |
        v
    Dockerfile
        |
        v
   Docker Image
        |
        v
 Docker Container
        |
        v
 FastAPI + Uvicorn
        |
        v
   /stream API
```

### Docker Image

The Docker image uses Python 3.12 and installs the project's pinned dependencies from `requirements.txt`.

The application is started using Uvicorn:

```text
uvicorn streaming_ux.streaming_ux:app --host 0.0.0.0 --port 8000
```

Binding the service to `0.0.0.0` makes the FastAPI application accessible outside the container.

### Build the Image

From the project root:

```powershell
docker build -f deployment/Dockerfile -t employee-hr-assistant .
```

Verify that the image was created:

```powershell
docker images
```

### Run the Container

Environment variables are supplied at runtime rather than being stored inside the Docker image.

```powershell
docker run -d --env-file .env -p 8000:8000 --name employee-hr-container employee-hr-assistant
```

The application is then available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Verify the Containerized Service

The SSE endpoint can be tested using:

```powershell
curl.exe -N "http://127.0.0.1:8000/stream?question=How%20many%20annual%20leave%20days%20do%20employees%20receive"
```

A successful containerized request produces retrieval events, streamed model chunks, and a final completion event.

Example flow:

```text
event: step
retrieval started

event: step
retrieval finished

event: step
model started

event: token
...

event: step
model finished

event: done
```

This verifies that document retrieval, model access, FastAPI, and SSE streaming operate successfully from the Docker container.

### Secret Hygiene

The `.env` file is excluded from the Docker image using `.dockerignore`.

Runtime configuration such as the OpenRouter API key, base URL, and model name is passed using:

```text
--env-file .env
```

This keeps credentials outside the built image.
---


### Evaluation Dataset

The evaluation contains exactly 30 cases covering areas such as:

- Annual leave
- Sick leave
- Working hours
- Probation
- Employee referrals
- Internal transfers
- Performance reviews
- Payroll
- Expense claims
- Notice periods
- Credential security
- AI usage policies
- Unsupported employee questions

The dataset includes both supported handbook questions and intentionally unsupported questions.

This measures not only whether the assistant can retrieve correct information, but also whether it avoids answering questions for which the handbook provides no evidence.

### Evaluation Criteria

For a supported handbook question, a case passes when:

```text
refused == False
```

and the expected information is present in the generated answer.

For example:

```text
Question:
How many paid annual leave days do full-time employees receive?

Expected:
18
```

For an unsupported question, a case passes when:

```text
refused == True
```

For example:

```text
Question:
Does the company provide free gym membership?

Expected:
Refusal
```

This provides a simple and reproducible evaluation instead of using another language model to judge the generated answer.

### Run the 30-Case Evaluation

From the project root:

```powershell
uv run python -m evaluation.evaluation
```

The evaluation reports each case individually:

```text
01. [PASS] ...
02. [PASS] ...
03. [FAIL] ...
```

and produces an aggregate summary:


The complete evaluation result is stored in:

```text
outputs/evaluation.txt
```

Failed evaluation cases are retained as useful evidence for identifying retrieval or answer-generation limitations rather than modifying the expected values simply to produce a perfect score.

### Evaluation Tests

The evaluation harness also has automated tests that verify:

- The dataset contains exactly 30 cases.
- Supported handbook questions can pass the evaluation criteria.
- Unsupported questions correctly pass through refusal.
- Both supported and refusal cases are represented in the evaluation dataset.

Run:

```powershell
uv run pytest tests/test_evaluation.py -v
```

---

## Guardrails and Evidence

The project includes shared guardrails to prevent uncontrolled execution and oversized inputs during the assessment.

The main guardrail logic is stored in:

```text
guardrails.py
```

Evidence for the implemented guardrails is generated using:

```text
guardrails_evidence.py
```

This evidence file intentionally triggers the guardrails so their behaviour can be demonstrated and saved as part of the 

### Shared Guardrail Design

The guardrails are kept in a shared module rather than rewriting the same checks separately for every task.

The implemented guardrail evidence demonstrates:

* Hard step limit
* Per-operation timeout
* Capped retry attempts
* Input/token budget enforcement

### Run Guardrail Evidence

```bash
uv run python guardrails_evidence.py
```

### Save Guardrail Evidence

```bash
uv run python guardrails_evidence.py > outputs/guardrails_evidence_output.txt
```

The saved output provides reproducible evidence that each guardrail fires when its configured limit or failure condition is reached.

---

## Environment Configuration

The project uses environment variables to keep API credentials and model configuration outside the source code.

An `.env.example` file is included to show the required environment variables without exposing any actual secrets:

Create a local `.env` file using the same variables and provide your own values.

The actual `.env` file is excluded from GitHub using `.gitignore` to ensure that API keys are not committed to the repository.

## Requirements

The `requirements.txt` file contains the Python dependencies required to run the assessment.

Install the dependencies using:

```bash
uv pip install -r requirements.txt
```

The main dependencies used in this assessment include LangChain, OpenRouter integration, FastAPI, Uvicorn, python-dotenv, HTTPX, and pytest.

After installing the dependencies and configuring the `.env` file, the individual assessment tasks can be executed using the commands provided in their respective sections above.