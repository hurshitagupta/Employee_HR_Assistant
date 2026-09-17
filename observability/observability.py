import json
import logging
import time
import uuid

from prometheus_client import Counter, Histogram, generate_latest

from grounded_knowledge.grounded_knowledge import answer_question


logger = logging.getLogger("employee_hr_assistant")
logger.setLevel(logging.INFO)

if not logger.handlers:

    handler = logging.StreamHandler()

    handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    logger.addHandler(handler)


def log_event(event: str, correlation_id: str, **details):
    """Write one structured JSON log event."""

    log_data = {
        "event": event,
        "correlation_id": correlation_id,
        **details,
    }

    logger.info(
        json.dumps(log_data)
    )


REQUEST_COUNT = Counter(
    "hr_assistant_requests_total",
    "Total number of HR Assistant requests.",
)

ERROR_COUNT = Counter(
    "hr_assistant_errors_total",
    "Total number of failed HR Assistant requests.",
)

REFUSAL_COUNT = Counter(
    "hr_assistant_refusals_total",
    "Total number of grounded refusals.",
)

REQUEST_LATENCY = Histogram(
    "hr_assistant_request_latency_seconds",
    "HR Assistant request latency in seconds.",
)

ALERT_LATENCY_SECONDS = 5.0


def check_latency_alert(
    latency: float,
    correlation_id: str,
):
    """Emit an alert when request latency exceeds the threshold."""

    if latency > ALERT_LATENCY_SECONDS:

        log_event(
            event="high_latency_alert",
            correlation_id=correlation_id,
            latency_seconds=round(latency, 3),
            threshold_seconds=ALERT_LATENCY_SECONDS,
        )

        return True

    return False


def process_request(question: str):
    """Process one HR request with logs and metrics."""

    correlation_id = str(uuid.uuid4())

    start_time = time.perf_counter()

    REQUEST_COUNT.inc()

    log_event(
        event="request_started",
        correlation_id=correlation_id,
        question=question,
    )

    try:

        response = answer_question(question)

        if response.refused:

            REFUSAL_COUNT.inc()

            log_event(
                event="request_refused",
                correlation_id=correlation_id,
                reason="Information not found in Employee Handbook.",
            )

        latency = time.perf_counter() - start_time

        REQUEST_LATENCY.observe(latency)

        log_event(
            event="request_completed",
            correlation_id=correlation_id,
            latency_seconds=round(latency, 3),
            refused=response.refused,
            citations=response.citations,
        )

        check_latency_alert(latency,correlation_id)

        return {
            "correlation_id": correlation_id,
            "response": response,
            "latency_seconds": latency,
        }

    except Exception as exc:

        ERROR_COUNT.inc()

        latency = time.perf_counter() - start_time

        REQUEST_LATENCY.observe(latency)

        log_event(
            event="request_failed",
            correlation_id=correlation_id,
            error=str(exc),
            latency_seconds=round(latency, 3),
        )

        raise


def get_metrics() -> str:
    """Return Prometheus metrics as text."""

    return generate_latest().decode("utf-8")


if __name__ == "__main__":

    question = (
        "How many annual leave days do eligible full-time employees receive?"
    )

    result = process_request(question)

    print("\n=== OBSERVABILITY RESULT ===")

    print(
        f"Correlation ID: "
        f"{result['correlation_id']}"
    )

    print(
        f"Answer: "
        f"{result['response'].answer}"
    )

    print(
        f"Latency: "
        f"{result['latency_seconds']:.3f} seconds"
    )

    print("\n=== APPLICATION METRICS ===")

    metrics = get_metrics()

    for line in metrics.splitlines():

        if line.startswith("hr_assistant_"):
            print(line)