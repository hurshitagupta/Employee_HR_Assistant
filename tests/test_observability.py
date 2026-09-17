import pytest

from observability.observability import check_latency_alert, get_metrics, process_request


def test_successful_observable_request():
    """A valid request should return a response and correlation ID."""

    result = process_request(
        "How many annual leave days do eligible full-time employees receive?"
    )

    assert result["correlation_id"]
    assert result["response"].refused is False
    assert "18" in result["response"].answer
    assert result["latency_seconds"] >= 0


def test_refusal_is_observed():
    """Unsupported handbook questions should be recorded as refusals."""

    result = process_request(
        "Does the company provide employees with a free gym membership?"
    )

    assert result["correlation_id"]
    assert result["response"].refused is True
    assert result["response"].citations == []


def test_metrics_are_exposed():
    """Application metrics should be available in Prometheus format."""

    metrics = get_metrics()

    assert "hr_assistant_requests_total" in metrics
    assert "hr_assistant_errors_total" in metrics
    assert "hr_assistant_refusals_total" in metrics
    assert "hr_assistant_request_latency_seconds" in metrics


def test_high_latency_alert():
    """Latency above the threshold should trigger the alert."""

    triggered = check_latency_alert(
        latency=6.0,
        correlation_id="test-correlation-id",
    )

    assert triggered is True


def test_normal_latency_no_alert():
    """Normal latency should not trigger the alert."""

    triggered = check_latency_alert(
        latency=1.0,
        correlation_id="test-correlation-id",
    )

    assert triggered is False


def test_failure_path():
    """Invalid input should propagate as a failed observable request."""

    with pytest.raises(ValueError):
        process_request("")