import time

import numpy as np
import pytest

from adaptive_ai import AdaptiveAI


def wait_for_job(ai, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = ai.get_training_job(job_id)
        if job["status"] in {"completed", "failed", "canceled", "paused"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish before timeout")


def test_workspace_is_created_with_sqlite_and_array_directories(tmp_path):
    ai = AdaptiveAI(path=tmp_path)

    assert ai.workspace_path == tmp_path
    assert (tmp_path / ".adaptive_ai" / "adaptive_ai.sqlite3").exists()
    assert (tmp_path / ".adaptive_ai" / "arrays").is_dir()
    assert (tmp_path / ".adaptive_ai" / "models").is_dir()


def test_set_and_put_input_output_persist_float64_streaming_dataset(tmp_path):
    ai = AdaptiveAI(path=tmp_path)

    ai.set_input_output([[0, 0], [1, 1]], [[0], [1]])
    ai.put_input_output([[1, 0]], [[1]])

    dataset = ai.get_dataset()
    assert dataset.sample_count == 3
    assert dataset.input_size == 2
    assert dataset.output_size == 1

    batches = list(dataset.iter_batches(batch_size=10))
    assert len(batches) == 1
    assert batches[0].inputs.dtype == np.float64
    assert batches[0].outputs.dtype == np.float64
    np.testing.assert_allclose(batches[0].inputs, [[0, 0], [1, 1], [1, 0]])
    np.testing.assert_allclose(batches[0].outputs, [[0], [1], [1]])


def test_rejects_incompatible_dimensions_and_outputs_outside_sigmoid_range(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0, 0]], [[0.5]])

    with pytest.raises(ValueError, match="input dimension"):
        ai.put_input_output([[0, 0, 0]], [[0.5]])

    with pytest.raises(ValueError, match="output dimension"):
        ai.put_input_output([[0, 0]], [[0.5, 0.2]])

    with pytest.raises(ValueError, match="between 0 and 1"):
        AdaptiveAI(path=tmp_path / "other").set_input_output([[0]], [[1.5]])


def test_forward_pass_uses_bias_and_sigmoid_for_each_layer(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    matrices = [
        np.array(
            [
                [1.0, -1.0],
                [0.5, 0.25],
                [0.1, -0.2],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [0.75],
                [-0.5],
                [0.3],
            ],
            dtype=np.float64,
        ),
    ]

    predictions = ai.predict_with_matrices([[2.0, -1.0]], matrices)

    first_input = np.array([[2.0, -1.0, 1.0]])
    hidden = 1.0 / (1.0 + np.exp(-(first_input @ matrices[0])))
    second_input = np.concatenate([hidden, np.ones((1, 1))], axis=1)
    expected = 1.0 / (1.0 + np.exp(-(second_input @ matrices[1])))
    np.testing.assert_allclose(predictions, expected)


def test_gradient_descent_reduces_mse_on_simple_dataset(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    inputs = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    outputs = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
    matrices = [np.array([[0.01], [0.0]], dtype=np.float64)]

    before = ai.evaluate_matrices(inputs, outputs, matrices, tolerances=[0.25])["mse"]
    trained = ai.train_matrices(inputs, outputs, matrices, steps=300, learning_rate=0.5)
    after = ai.evaluate_matrices(inputs, outputs, trained, tolerances=[0.25])["mse"]

    assert after < before


def test_acceptance_uses_per_output_tolerances(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    expected = np.array([[0.5, 0.5], [0.9, 0.1]], dtype=np.float64)
    predicted = np.array([[0.55, 0.69], [0.7, 0.2]], dtype=np.float64)

    metrics = ai.evaluate_predictions(predicted, expected, tolerances=[0.1, 0.2])

    assert metrics["accepted_count"] == 1
    assert metrics["accepted_rate"] == 0.5


def test_fixed_training_job_completes_and_logs_progress(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0], [1], [2], [3]], [[0], [0], [1], [1]])

    job = ai.start_training(
        max_seconds=0.2,
        tolerances=[0.35],
        amount_strategy="fixed",
        fixed_steps=2,
        learning_rate=0.2,
        seed=1,
    )

    finished = wait_for_job(ai, job["job_id"])
    logs = ai.get_training_logs(job["job_id"])
    models = ai.get_models()

    assert finished["status"] == "completed"
    assert finished["rounds_completed"] >= 1
    assert logs
    assert models
    assert "matrices" not in models[0]


def test_sample_square_strategy_records_validation_count_and_steps(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0], [1], [2], [3], [4]], [[0], [0], [1], [1], [1]])

    job = ai.start_training(
        max_seconds=0.2,
        tolerances=[0.99],
        amount_strategy="sample_square",
        learning_rate=0.1,
        seed=2,
    )

    wait_for_job(ai, job["job_id"])
    logs = ai.get_training_logs(job["job_id"], limit=1)

    assert logs[0]["validation_count"] >= 1
    assert logs[0]["steps"] == logs[0]["validation_count"] ** 2


def test_training_job_can_be_paused_and_canceled(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0], [1], [2], [3]], [[0], [0], [1], [1]])

    paused = ai.start_training(
        max_seconds=5.0,
        tolerances=[0.2],
        amount_strategy="fixed",
        fixed_steps=500,
        learning_rate=0.1,
        seed=3,
    )
    ai.pause_training(paused["job_id"])
    paused_finished = wait_for_job(ai, paused["job_id"])
    assert paused_finished["status"] == "paused"

    canceled = ai.start_training(
        max_seconds=5.0,
        tolerances=[0.2],
        amount_strategy="fixed",
        fixed_steps=500,
        learning_rate=0.1,
        seed=4,
    )
    ai.cancel_training(canceled["job_id"])
    canceled_finished = wait_for_job(ai, canceled["job_id"])
    assert canceled_finished["status"] == "canceled"


def test_get_model_returns_matrices_and_mutation_prunes_pool_to_sqrt_dataset(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0], [1], [2], [3], [4], [5], [6], [7], [8]],
        [[0], [0], [0], [1], [1], [1], [1], [1], [1]],
    )

    job = ai.start_training(
        max_seconds=0.5,
        tolerances=[0.95],
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.05,
        seed=5,
    )
    wait_for_job(ai, job["job_id"])

    models = ai.get_models()
    assert 1 <= len(models) <= 3
    model = ai.get_model(models[0]["model_id"])
    assert model["matrices"]
    assert all(matrix.dtype == np.float64 for matrix in model["matrices"])


def test_training_job_streams_batches_without_loading_full_dataset(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0], [1], [2], [3], [4]],
        [[0], [0], [1], [1], [1]],
        sample_ids=[f"sample-{index}" for index in range(5)],
    )

    job = ai.start_training(
        max_seconds=0.2,
        tolerances=[0.95],
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.05,
        seed=5,
        train_ratio=0.8,
        batch_size=2,
    )
    finished = wait_for_job(ai, job["job_id"])

    assert finished["status"] == "completed"
    assert finished["rounds_completed"] >= 1
    assert finished["train_ratio"] == 0.8
    assert finished["batch_size"] == 2


def test_start_training_normalizes_train_ratio_before_persisting(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0], [1], [2], [3]],
        [[0], [0], [1], [1]],
        sample_ids=[f"ratio-{index}" for index in range(4)],
    )

    job = ai.start_training(
        max_seconds=0.2,
        tolerances=[0.95],
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.05,
        seed=6,
        train_ratio="0.5",
        batch_size=2,
    )
    finished = wait_for_job(ai, job["job_id"])

    assert finished["status"] == "completed"
    assert finished["train_ratio"] == 0.5


def test_start_training_rejects_invalid_train_ratio_with_value_error(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0], [1]], [[0], [1]], sample_ids=["left", "right"])

    with pytest.raises(
        ValueError,
        match="train_ratio must be finite and greater than 0 and less than 1",
    ):
        ai.start_training(
            max_seconds=0.2,
            tolerances=[0.95],
            amount_strategy="fixed",
            fixed_steps=1,
            learning_rate=0.05,
            train_ratio="half",
            batch_size=2,
        )


def test_streaming_dataset_public_usage_smoke(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0], [1]], [[0], [1]], sample_ids=["left", "right"])

    dataset = ai.get_dataset()
    first_batch = next(dataset.iter_batches(batch_size=1))
    selected = ai.get_samples(["right"])

    assert dataset.sample_count == 2
    assert first_batch.sample_ids == ["left"]
    assert selected.sample_ids == ["right"]
    np.testing.assert_allclose(selected.outputs, [[1]])
