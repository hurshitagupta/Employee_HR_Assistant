import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_openrouter import ChatOpenRouter

from grounded_knowledge.grounded_knowledge import retrieve_context


load_dotenv()


app = FastAPI(
    title="Employee HR Assistant - Streaming UX"
)


model = ChatOpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0,
    timeout=30_000,
    max_retries=2,
)


def sse_event(event: str, data: dict) -> str:
    """Format an event using the SSE protocol."""

    return (
        f"event: {event}\n"
        f"data: {json.dumps(data)}\n\n"
    )


async def stream_answer(question: str):
    """Stream retrieval steps and model tokens."""

    if not question.strip():
        yield sse_event(
            "error",
            {"message": "Question cannot be empty."},
        )
        return
    
    yield sse_event(
        "step",
        {
            "name": "retrieval",
            "status": "started",
        },
    )

    try:
        context, pages = retrieve_context(question)

    except Exception as exc:
        yield sse_event(
            "error",
            {
                "message": f"Retrieval failed: {exc}",
            },
        )
        return


    yield sse_event(
        "step",
        {
            "name": "retrieval",
            "status": "finished",
            "pages": pages,
        },
    )

    prompt = f"""
You are an Employee HR Assistant.

Answer the employee's question using ONLY the Employee Handbook
context provided below.

Rules:
- Do not use outside knowledge.
- Do not invent company policies.
- Include the supporting page citation.
- If the context does not contain enough information, say:
  The requested information could not be found in the Employee Handbook.

Employee question:
{question}

Employee Handbook context:
{context}
"""

    yield sse_event(
        "step",
        {
            "name": "model",
            "status": "started",
        },
    )

    try:

        async for chunk in model.astream(prompt):

            if chunk.content:

                yield sse_event("token",{"content": chunk.content},)

    except Exception as exc:

        yield sse_event("error",{"message": f"Model streaming failed: {exc}"},)
        return

    yield sse_event("step",{"name": "model","status": "finished",})

    yield sse_event("done",{"message": "Stream completed."})


@app.get("/stream")
async def stream(
    question: str = Query(...),
):

    return StreamingResponse(
        stream_answer(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


