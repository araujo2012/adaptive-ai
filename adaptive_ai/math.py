from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


ArrayList = list[np.ndarray]


def as_2d_float64(values: object, *, name: str, one_dim_as_column: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if array.ndim == 1:
        if one_dim_as_column:
            array = array.reshape(-1, 1)
        else:
            array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return array.astype(np.float64, copy=False)


def validate_outputs(outputs: np.ndarray) -> None:
    if ((outputs < 0.0) | (outputs > 1.0)).any():
        raise ValueError("outputs must be between 0 and 1 for sigmoid training")


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def with_bias(values: np.ndarray) -> np.ndarray:
    bias = np.ones((values.shape[0], 1), dtype=np.float64)
    return np.concatenate([values, bias], axis=1)


def validate_matrices(input_size: int, output_size: int, matrices: Sequence[np.ndarray]) -> None:
    if not matrices:
        raise ValueError("matrices must contain at least one matrix")
    previous_size = input_size
    for index, matrix in enumerate(matrices):
        if matrix.ndim != 2:
            raise ValueError(f"matrix {index} must be 2D")
        expected_rows = previous_size + 1
        if matrix.shape[0] != expected_rows:
            raise ValueError(
                f"matrix {index} row count must be {expected_rows}, got {matrix.shape[0]}"
            )
        previous_size = matrix.shape[1]
    if previous_size != output_size:
        raise ValueError(
            f"final matrix output dimension must be {output_size}, got {previous_size}"
        )


def predict(inputs: object, matrices: Sequence[np.ndarray]) -> np.ndarray:
    activation = as_2d_float64(inputs, name="inputs")
    for matrix in matrices:
        activation = sigmoid(with_bias(activation) @ matrix)
    return activation.astype(np.float64, copy=False)


def forward_cache(inputs: np.ndarray, matrices: Sequence[np.ndarray]) -> tuple[ArrayList, ArrayList]:
    activations: ArrayList = [inputs]
    z_values: ArrayList = []
    activation = inputs
    for matrix in matrices:
        z_value = with_bias(activation) @ matrix
        activation = sigmoid(z_value)
        z_values.append(z_value)
        activations.append(activation)
    return activations, z_values


def evaluate_predictions(
    predicted: object,
    expected: object,
    tolerances: Sequence[float],
) -> dict[str, float | int]:
    predicted_array = as_2d_float64(predicted, name="predicted", one_dim_as_column=True)
    expected_array = as_2d_float64(expected, name="expected", one_dim_as_column=True)
    if predicted_array.shape != expected_array.shape:
        raise ValueError("predicted and expected must have the same shape")
    tolerance_array = np.asarray(tolerances, dtype=np.float64)
    if tolerance_array.ndim != 1 or tolerance_array.shape[0] != expected_array.shape[1]:
        raise ValueError("tolerances must match the output dimension")
    if (tolerance_array < 0.0).any() or not np.isfinite(tolerance_array).all():
        raise ValueError("tolerances must contain non-negative finite numbers")

    differences = np.abs(predicted_array - expected_array)
    accepted_mask = (differences <= tolerance_array).all(axis=1)
    mse = float(np.mean(np.square(predicted_array - expected_array)))
    accepted_count = int(np.sum(accepted_mask))
    total = int(expected_array.shape[0])
    return {
        "accepted_count": accepted_count,
        "accepted_rate": float(accepted_count / total) if total else 0.0,
        "mse": mse,
    }


def evaluate_matrices(
    inputs: object,
    outputs: object,
    matrices: Sequence[np.ndarray],
    tolerances: Sequence[float],
) -> dict[str, float | int]:
    input_array = as_2d_float64(inputs, name="inputs")
    output_array = as_2d_float64(outputs, name="outputs", one_dim_as_column=True)
    predictions = predict(input_array, matrices)
    return evaluate_predictions(predictions, output_array, tolerances)


def is_better(
    candidate: dict[str, float | int],
    baseline: dict[str, float | int],
) -> bool:
    candidate_rate = float(candidate["accepted_rate"])
    baseline_rate = float(baseline["accepted_rate"])
    if candidate_rate > baseline_rate:
        return True
    if candidate_rate < baseline_rate:
        return False
    return float(candidate["mse"]) < float(baseline["mse"])


def train_matrices(
    inputs: object,
    outputs: object,
    matrices: Sequence[np.ndarray],
    *,
    steps: int,
    learning_rate: float,
    stop_checker: Callable[[], bool] | None = None,
) -> ArrayList:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if learning_rate <= 0.0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be a positive finite number")

    input_array = as_2d_float64(inputs, name="inputs")
    output_array = as_2d_float64(outputs, name="outputs", one_dim_as_column=True)
    trained = [np.array(matrix, dtype=np.float64, copy=True) for matrix in matrices]
    sample_count, output_count = output_array.shape
    if sample_count == 0:
        raise ValueError("training requires at least one sample")

    for _ in range(steps):
        if stop_checker is not None and stop_checker():
            break
        activations, z_values = forward_cache(input_array, trained)
        prediction = activations[-1]
        delta = (2.0 * (prediction - output_array) / (sample_count * output_count)) * (
            prediction * (1.0 - prediction)
        )
        gradients: ArrayList = [np.zeros_like(matrix) for matrix in trained]
        for layer_index in range(len(trained) - 1, -1, -1):
            previous_activation = activations[layer_index]
            gradients[layer_index] = with_bias(previous_activation).T @ delta
            if layer_index > 0:
                matrix_without_bias = trained[layer_index][:-1, :]
                previous_output = activations[layer_index]
                delta = (delta @ matrix_without_bias.T) * (
                    previous_output * (1.0 - previous_output)
                )
        for index, gradient in enumerate(gradients):
            trained[index] = trained[index] - learning_rate * gradient

    return trained


def random_matrix(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=0.1, size=(rows, cols)).astype(np.float64)


def architecture_from_matrices(matrices: Sequence[np.ndarray]) -> list[list[int]]:
    return [[int(matrix.shape[0]), int(matrix.shape[1])] for matrix in matrices]


def mutate_matrices(matrices: Sequence[np.ndarray], rng: np.random.Generator) -> ArrayList:
    copied = [np.array(matrix, dtype=np.float64, copy=True) for matrix in matrices]
    mutations = ["add_layer"]
    if len(copied) >= 2:
        mutations.extend(["increase_matrix", "reduce_matrix"])
    mutation = str(rng.choice(mutations))
    if mutation == "increase_matrix":
        return increase_shared_dimension(copied, rng)
    if mutation == "reduce_matrix":
        reduced = reduce_shared_dimension(copied, rng)
        if reduced is not None:
            return reduced
    return add_layer(copied, rng)


def add_layer(matrices: ArrayList, rng: np.random.Generator) -> ArrayList:
    index = int(rng.integers(0, len(matrices)))
    current_width = int(matrices[index].shape[1])
    new_matrix = random_matrix(current_width + 1, current_width, rng)
    return matrices[: index + 1] + [new_matrix] + matrices[index + 1 :]


def increase_shared_dimension(matrices: ArrayList, rng: np.random.Generator) -> ArrayList:
    boundary = int(rng.integers(0, len(matrices) - 1))
    left = matrices[boundary]
    right = matrices[boundary + 1]
    new_left_column = random_matrix(left.shape[0], 1, rng)
    new_right_row = random_matrix(1, right.shape[1], rng)
    matrices[boundary] = np.concatenate([left, new_left_column], axis=1)
    matrices[boundary + 1] = np.concatenate([right[:-1, :], new_right_row, right[-1:, :]], axis=0)
    return matrices


def reduce_shared_dimension(matrices: ArrayList, rng: np.random.Generator) -> ArrayList | None:
    eligible = [
        index
        for index in range(len(matrices) - 1)
        if matrices[index].shape[1] > 1 and matrices[index + 1].shape[0] > 2
    ]
    if not eligible:
        return None
    boundary = int(rng.choice(eligible))
    shared_width = matrices[boundary].shape[1]
    remove_index = int(rng.integers(0, shared_width))
    matrices[boundary] = np.delete(matrices[boundary], remove_index, axis=1)
    matrices[boundary + 1] = np.delete(matrices[boundary + 1], remove_index, axis=0)
    return matrices
