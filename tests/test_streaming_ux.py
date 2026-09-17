import pytest

from streaming_ux.streaming_ux import sse_event, stream_answer


def test_sse_event_format():
    """SSE events should contain an event type and JSON data."""

    event = sse_event(
        "step",
        {
            "name": "retrieval",
            "status": "started",
        },
    )

    assert "event: step" in event
    assert '"name": "retrieval"' in event
    assert '"status": "started"' in event
    assert event.endswith("\n\n")


@pytest.mark.anyio
async def test_streaming_success():
    """A valid question should stream steps, tokens and completion."""

    events = []

    async for event in stream_answer(
        "How many annual leave days do eligible full-time employees receive?"
    ):
        events.append(event)

    full_stream = "".join(events)

    assert "event: step" in full_stream
    assert '"name": "retrieval"' in full_stream
    assert '"name": "model"' in full_stream

    assert "event: token" in full_stream
    assert "event: done" in full_stream

    assert "18" in full_stream


@pytest.mark.anyio
async def test_empty_question_failure():
    """An empty question should return an SSE error event."""

    events = []

    async for event in stream_answer(""):
        events.append(event)

    full_stream = "".join(events)

    assert "event: error" in full_stream
    assert "Question cannot be empty" in full_stream

    assert "event: token" not in full_stream
    assert "event: done" not in full_stream