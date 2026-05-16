import numpy as np
import pytest

from Topyfic.benchmarking import (
    resolve_parity_reference_device,
    resolve_requested_devices,
    summarize_durations,
    topic_alignment_cost,
)


def test_summarize_durations_separates_warmup_from_measured_runs():
    summary = summarize_durations([10.0, 4.0, 6.0], warmup_runs=1)

    assert summary["warmup_seconds"] == [10.0]
    assert summary["measured_seconds"] == [4.0, 6.0]
    assert summary["mean_seconds"] == 5.0
    assert summary["median_seconds"] == 5.0


def test_summarize_durations_requires_measured_run():
    with pytest.raises(ValueError):
        summarize_durations([1.0], warmup_runs=1)


def test_topic_alignment_cost_returns_reference_order_alignment():
    left = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )
    right = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
        ]
    )

    rows, cols, costs = topic_alignment_cost(left, right)

    assert rows.tolist() == [0, 1]
    assert cols.tolist() == [1, 0]
    np.testing.assert_allclose(costs, [0.0, 0.0], atol=1e-8)


def test_resolve_requested_devices_rejects_unknown_device():
    with pytest.raises(ValueError):
        resolve_requested_devices(["tpu"])


def test_resolve_parity_reference_device_prefers_cpu():
    reference_device, candidate_devices = resolve_parity_reference_device(["mps", "cpu", "cuda"])

    assert reference_device == "cpu"
    assert candidate_devices == ["mps", "cuda"]


def test_resolve_parity_reference_device_falls_back_to_first_available_device():
    reference_device, candidate_devices = resolve_parity_reference_device(["mps", "cuda"])

    assert reference_device == "mps"
    assert candidate_devices == ["cuda"]