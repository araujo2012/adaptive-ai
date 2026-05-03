from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset import DatasetBatch, DatasetView, SampleBatch
from .math import (
    as_2d_float64,
    evaluate_matrices,
    evaluate_predictions,
    is_better,
    mutate_matrices,
    predict,
    random_matrix,
    train_matrices,
    validate_matrices,
    validate_outputs,
)
from .storage import Storage


@dataclass
class _JobControl:
    cancel_event: threading.Event
    pause_event: threading.Event
    thread: threading.Thread | None = None


class AdaptiveAI:
    def __init__(self, path: str | Path = "."):
        self.workspace_path = Path(path)
        self._storage = Storage(self.workspace_path)
        self._controls: dict[str, _JobControl] = {}
        self._controls_lock = threading.RLock()

    def set_input_output(
        self,
        inputs: object,
        outputs: object,
        *,
        sample_ids: Sequence[object] | None = None,
    ) -> None:
        input_array, output_array = self._prepare_dataset(inputs, outputs)
        self._ensure_or_set_dimensions(input_array.shape[1], output_array.shape[1])
        self._storage.replace_dataset(input_array, output_array, sample_ids=sample_ids)
        self._storage.clear_models()

    def put_input_output(
        self,
        inputs: object,
        outputs: object,
        *,
        sample_ids: Sequence[object] | None = None,
    ) -> None:
        input_array, output_array = self._prepare_dataset(inputs, outputs)
        self._require_dimensions(input_array.shape[1], output_array.shape[1])
        self._storage.append_dataset(input_array, output_array, sample_ids=sample_ids)

    def get_dataset(self) -> DatasetView:
        self._storage.raise_if_legacy_dataset_requires_migration()
        dimensions = self._storage.get_dimensions()
        if dimensions is None:
            raise ValueError("dataset has not been set")
        input_size, output_size = dimensions

        def iter_batches(batch_size: int) -> Iterator[DatasetBatch]:
            yield from self._storage.iter_dataset_batches(batch_size=batch_size)

        return DatasetView(
            sample_count=self._storage.get_sample_count(),
            input_size=input_size,
            output_size=output_size,
            batch_iterator=iter_batches,
        )

    def predict_with_matrices(self, inputs: object, matrices: Sequence[np.ndarray]) -> np.ndarray:
        input_array = as_2d_float64(inputs, name="inputs")
        dimensions = self._storage.get_dimensions()
        if dimensions is not None:
            validate_matrices(dimensions[0], dimensions[1], matrices)
        return predict(input_array, matrices)

    def evaluate_predictions(
        self,
        predicted: object,
        expected: object,
        tolerances: Sequence[float],
    ) -> dict[str, float | int]:
        return evaluate_predictions(predicted, expected, tolerances)

    def evaluate_matrices(
        self,
        inputs: object,
        outputs: object,
        matrices: Sequence[np.ndarray],
        tolerances: Sequence[float],
    ) -> dict[str, float | int]:
        return evaluate_matrices(inputs, outputs, matrices, tolerances)

    def train_matrices(
        self,
        inputs: object,
        outputs: object,
        matrices: Sequence[np.ndarray],
        *,
        steps: int,
        learning_rate: float,
    ) -> list[np.ndarray]:
        return train_matrices(inputs, outputs, matrices, steps=steps, learning_rate=learning_rate)

    def start_training(
        self,
        *,
        max_seconds: float,
        tolerances: Sequence[float],
        amount_strategy: str = "fixed",
        fixed_steps: int | None = 100,
        learning_rate: float = 0.01,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if max_seconds <= 0.0 or not math.isfinite(max_seconds):
            raise ValueError("max_seconds must be a positive finite number")
        if amount_strategy not in {"fixed", "sample_square"}:
            raise ValueError("amount_strategy must be 'fixed' or 'sample_square'")
        if amount_strategy == "fixed" and (fixed_steps is None or fixed_steps <= 0):
            raise ValueError("fixed_steps must be a positive integer when using fixed strategy")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be a positive finite number")

        dataset = self._storage.load_dataset()
        if dataset["inputs"].shape[0] < 2:
            raise ValueError("training requires at least two dataset samples")
        dimensions = self._storage.get_dimensions()
        if dimensions is None:
            raise ValueError("dataset dimensions have not been initialized")
        tolerance_array = self._validate_tolerances(tolerances, dimensions[1])

        with self._controls_lock:
            if self._storage.has_running_job():
                raise ValueError("a training job is already running")
            job_id = self._storage.create_job(
                max_seconds=max_seconds,
                amount_strategy=amount_strategy,
                fixed_steps=fixed_steps,
                learning_rate=learning_rate,
            )
            control = _JobControl(cancel_event=threading.Event(), pause_event=threading.Event())
            thread = threading.Thread(
                target=self._run_training_job,
                args=(
                    job_id,
                    max_seconds,
                    tolerance_array,
                    amount_strategy,
                    fixed_steps,
                    learning_rate,
                    seed,
                    control,
                ),
                daemon=True,
            )
            control.thread = thread
            self._controls[job_id] = control
            thread.start()
        return self.get_training_job(job_id)

    def get_models(self) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in model.items() if key != "matrix_path"}
            for model in self._storage.list_models()
        ]

    def get_model(self, model_id: str) -> dict[str, Any]:
        metadata = self._storage.get_model_meta(model_id)
        metadata = {key: value for key, value in metadata.items() if key != "matrix_path"}
        metadata["matrices"] = self._storage.load_model(model_id)
        return metadata

    def get_training_job(self, job_id: str) -> dict[str, Any]:
        return self._storage.get_job(job_id)

    def get_training_logs(self, job_id: str, limit: int = 100) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return self._storage.get_logs(job_id, limit=limit)

    def pause_training(self, job_id: str) -> None:
        with self._controls_lock:
            control = self._controls.get(job_id)
            if control is not None:
                control.pause_event.set()
                return
        job = self._storage.get_job(job_id)
        if job["status"] == "running":
            self._storage.update_job(job_id, status="paused", finish=True)

    def cancel_training(self, job_id: str) -> None:
        with self._controls_lock:
            control = self._controls.get(job_id)
            if control is not None:
                control.cancel_event.set()
                return
        job = self._storage.get_job(job_id)
        if job["status"] == "running":
            self._storage.update_job(job_id, status="canceled", finish=True)

    def _prepare_dataset(self, inputs: object, outputs: object) -> tuple[np.ndarray, np.ndarray]:
        input_array = as_2d_float64(inputs, name="inputs")
        output_array = as_2d_float64(outputs, name="outputs", one_dim_as_column=True)
        validate_outputs(output_array)
        if input_array.shape[0] != output_array.shape[0]:
            raise ValueError("inputs and outputs must contain the same number of samples")
        return input_array, output_array

    def _ensure_or_set_dimensions(self, input_size: int, output_size: int) -> None:
        dimensions = self._storage.get_dimensions()
        if dimensions is None:
            self._storage.set_dimensions(input_size, output_size)
            return
        self._require_dimensions(input_size, output_size)

    def _require_dimensions(self, input_size: int, output_size: int) -> None:
        dimensions = self._storage.get_dimensions()
        if dimensions is None:
            self._storage.set_dimensions(input_size, output_size)
            return
        expected_input, expected_output = dimensions
        if input_size != expected_input:
            raise ValueError(f"input dimension must be {expected_input}, got {input_size}")
        if output_size != expected_output:
            raise ValueError(f"output dimension must be {expected_output}, got {output_size}")

    def _validate_tolerances(self, tolerances: Sequence[float], output_size: int) -> np.ndarray:
        tolerance_array = np.asarray(tolerances, dtype=np.float64)
        if tolerance_array.ndim != 1 or tolerance_array.shape[0] != output_size:
            raise ValueError("tolerances must match the output dimension")
        if (tolerance_array < 0.0).any() or not np.isfinite(tolerance_array).all():
            raise ValueError("tolerances must contain non-negative finite numbers")
        return tolerance_array

    def _run_training_job(
        self,
        job_id: str,
        max_seconds: float,
        tolerances: np.ndarray,
        amount_strategy: str,
        fixed_steps: int | None,
        learning_rate: float,
        seed: int | None,
        control: _JobControl,
    ) -> None:
        rounds_completed = 0
        rng = np.random.default_rng(seed)
        deadline = time.monotonic() + max_seconds
        try:
            dataset = self._storage.load_dataset()
            inputs = dataset["inputs"]
            outputs = dataset["outputs"]
            input_size, output_size = self._storage.get_dimensions() or (
                inputs.shape[1],
                outputs.shape[1],
            )
            self._ensure_model_pool(input_size, output_size, inputs, outputs, tolerances, rng)

            while time.monotonic() < deadline:
                if control.cancel_event.is_set():
                    self._storage.update_job(
                        job_id,
                        status="canceled",
                        rounds_completed=rounds_completed,
                        finish=True,
                    )
                    return
                if control.pause_event.is_set():
                    self._storage.update_job(
                        job_id,
                        status="paused",
                        rounds_completed=rounds_completed,
                        finish=True,
                    )
                    return

                models = self._storage.list_models()
                if not models:
                    self._ensure_model_pool(input_size, output_size, inputs, outputs, tolerances, rng)
                    models = self._storage.list_models()
                selected = models[int(rng.integers(0, len(models)))]
                selected_id = str(selected["model_id"])
                original = self._storage.load_model(selected_id)

                full_metrics = evaluate_matrices(inputs, outputs, original, tolerances)
                validation_count = _validation_count(
                    int(full_metrics["accepted_count"]),
                    inputs.shape[0],
                )
                validation_indices = rng.choice(
                    inputs.shape[0],
                    size=validation_count,
                    replace=False,
                )
                train_mask = np.ones(inputs.shape[0], dtype=bool)
                train_mask[validation_indices] = False
                train_inputs = inputs[train_mask]
                train_outputs = outputs[train_mask]
                validation_inputs = inputs[validation_indices]
                validation_outputs = outputs[validation_indices]
                steps = (
                    int(fixed_steps)
                    if amount_strategy == "fixed"
                    else int(validation_count * validation_count)
                )

                baseline_validation = evaluate_matrices(
                    validation_inputs,
                    validation_outputs,
                    original,
                    tolerances,
                )
                trained = train_matrices(
                    train_inputs,
                    train_outputs,
                    original,
                    steps=steps,
                    learning_rate=learning_rate,
                    stop_checker=lambda: control.cancel_event.is_set()
                    or control.pause_event.is_set()
                    or time.monotonic() >= deadline,
                )
                if control.cancel_event.is_set():
                    self._storage.update_job(
                        job_id,
                        status="canceled",
                        rounds_completed=rounds_completed,
                        finish=True,
                    )
                    return
                if control.pause_event.is_set():
                    self._storage.update_job(
                        job_id,
                        status="paused",
                        rounds_completed=rounds_completed,
                        finish=True,
                    )
                    return

                trained_validation = evaluate_matrices(
                    validation_inputs,
                    validation_outputs,
                    trained,
                    tolerances,
                )
                kept_matrices = original
                message = "round reverted"
                if is_better(trained_validation, baseline_validation):
                    kept_matrices = trained
                    message = "round improved"
                    kept_metrics = evaluate_matrices(inputs, outputs, kept_matrices, tolerances)
                    self._storage.save_model(
                        kept_matrices,
                        model_id=selected_id,
                        metrics=kept_metrics,
                    )
                    child = mutate_matrices(kept_matrices, rng)
                    child_metrics = evaluate_matrices(inputs, outputs, child, tolerances)
                    self._storage.save_model(
                        child,
                        parent_id=selected_id,
                        generation=int(selected["generation"]) + 1,
                        metrics=child_metrics,
                    )
                else:
                    self._storage.save_model(
                        kept_matrices,
                        model_id=selected_id,
                        metrics=full_metrics,
                    )

                max_models = max(1, math.ceil(math.sqrt(inputs.shape[0])))
                self._storage.prune_models(max_models)
                final_metrics = evaluate_matrices(inputs, outputs, kept_matrices, tolerances)
                rounds_completed += 1
                self._storage.add_log(
                    job_id,
                    message=message,
                    model_id=selected_id,
                    validation_count=validation_count,
                    steps=steps,
                    accepted_rate=float(final_metrics["accepted_rate"]),
                    mse=float(final_metrics["mse"]),
                )
                self._storage.update_job(job_id, rounds_completed=rounds_completed)

            self._storage.update_job(
                job_id,
                status="completed",
                rounds_completed=rounds_completed,
                finish=True,
            )
        except Exception as exc:
            self._storage.update_job(
                job_id,
                status="failed",
                rounds_completed=rounds_completed,
                error=str(exc),
                finish=True,
            )
        finally:
            with self._controls_lock:
                self._controls.pop(job_id, None)

    def _ensure_model_pool(
        self,
        input_size: int,
        output_size: int,
        inputs: np.ndarray,
        outputs: np.ndarray,
        tolerances: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        if self._storage.list_models():
            return
        matrix = random_matrix(input_size + 1, output_size, rng)
        matrices = [matrix]
        metrics = evaluate_matrices(inputs, outputs, matrices, tolerances)
        self._storage.save_model(matrices, metrics=metrics)


def _validation_count(accepted_count: int, dataset_size: int) -> int:
    if dataset_size < 2:
        raise ValueError("training requires at least two dataset samples")
    return min(max(accepted_count, 1), dataset_size - 1)
