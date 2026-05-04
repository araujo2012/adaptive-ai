import numpy as np
import pytest

from adaptive_ai.math import (
    evaluate_matrices,
    evaluate_matrices_batches,
    train_matrices_batches,
)


def test_streaming_evaluation_matches_full_array_metrics():
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    outputs = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
    matrices = [np.array([[0.25], [0.0]], dtype=np.float64)]
    tolerances = [0.3]

    batches = [
        (inputs[:2], outputs[:2]),
        (inputs[2:], outputs[2:]),
    ]

    full = evaluate_matrices(inputs, outputs, matrices, tolerances)
    streamed = evaluate_matrices_batches(batches, matrices, tolerances)

    assert streamed["accepted_count"] == full["accepted_count"]
    assert streamed["accepted_rate"] == full["accepted_rate"]
    assert streamed["mse"] == full["mse"]


def test_batch_training_reduces_mse_while_reusing_batch_factory():
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    outputs = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
    matrices = [np.array([[0.01], [0.0]], dtype=np.float64)]

    def batch_factory():
        yield inputs[:2], outputs[:2]
        yield inputs[2:], outputs[2:]

    before = evaluate_matrices(inputs, outputs, matrices, tolerances=[0.25])["mse"]
    trained = train_matrices_batches(
        batch_factory,
        matrices,
        steps=300,
        learning_rate=0.5,
    )
    after = evaluate_matrices(inputs, outputs, trained, tolerances=[0.25])["mse"]

    assert after < before


def test_batch_training_rejects_empty_batch_factory():
    matrices = [np.array([[0.01], [0.0]], dtype=np.float64)]

    def batch_factory():
        if False:
            yield None

    with pytest.raises(ValueError, match="training requires at least one sample"):
        train_matrices_batches(batch_factory, matrices, steps=1, learning_rate=0.1)


def test_batch_training_checks_stop_before_fetching_first_batch():
    matrices = [np.array([[0.01], [0.0]], dtype=np.float64)]
    factory_calls = 0

    def batch_factory():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("batch_factory should not be called")
        yield

    trained = train_matrices_batches(
        batch_factory,
        matrices,
        steps=10,
        learning_rate=0.1,
        stop_checker=lambda: True,
    )

    assert factory_calls == 0
    assert trained is not matrices
    np.testing.assert_allclose(trained[0], matrices[0])


def test_batch_training_checks_stop_after_fetch_before_update():
    inputs = np.array([[1.0]], dtype=np.float64)
    outputs = np.array([[1.0]], dtype=np.float64)
    matrices = [np.array([[0.01], [0.0]], dtype=np.float64)]
    stop_now = False

    def batch_factory():
        nonlocal stop_now
        stop_now = True
        yield inputs, outputs

    trained = train_matrices_batches(
        batch_factory,
        matrices,
        steps=1,
        learning_rate=0.5,
        stop_checker=lambda: stop_now,
    )

    np.testing.assert_allclose(trained[0], matrices[0])
