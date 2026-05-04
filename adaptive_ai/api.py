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
    evaluate_matrices_batches,
    evaluate_predictions,
    is_better,
    mutate_matrices,
    predict,
    random_matrix,
    train_matrices,
    train_matrices_batches,
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

    def get_samples(self, sample_ids: Sequence[object]) -> SampleBatch:
        self._storage.raise_if_legacy_dataset_requires_migration()
        return self._storage.load_samples_by_ids(sample_ids)

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
        train_ratio: float = 0.8,
        batch_size: int = 1024,
    ) -> dict[str, Any]:
        if max_seconds <= 0.0 or not math.isfinite(max_seconds):
            raise ValueError("max_seconds must be a positive finite number")
        if amount_strategy not in {"fixed", "sample_square"}:
            raise ValueError("amount_strategy must be 'fixed' or 'sample_square'")
        if amount_strategy == "fixed" and (fixed_steps is None or fixed_steps <= 0):
            raise ValueError("fixed_steps must be a positive integer when using fixed strategy")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be a positive finite number")
        if train_ratio <= 0.0 or train_ratio >= 1.0 or not math.isfinite(train_ratio):
            raise ValueError("train_ratio must be greater than 0 and less than 1")
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
            raise ValueError("batch_size must be positive integer")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive integer")

        if self._storage.get_sample_count() < 2:
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
                train_ratio=train_ratio,
                batch_size=batch_size,
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
                    train_ratio,
                    batch_size,
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
        train_ratio: float,
        batch_size: int,
        control: _JobControl,
    ) -> None:
        rounds_completed = 0
        rng = np.random.default_rng(seed)
        deadline = time.monotonic() + max_seconds
        try:
            sample_count = self._storage.get_sample_count()
            dimensions = self._storage.get_dimensions()
            if dimensions is None:
                raise ValueError("dataset dimensions have not been initialized")
            input_size, output_size = dimensions
            split = self._storage.get_or_create_training_split(
                job_id,
                seed=seed,
                train_ratio=train_ratio,
            )
            self._ensure_model_pool(
                input_size,
                output_size,
                split.validation_keys,
                tolerances,
                rng,
                batch_size=batch_size,
            )

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
                    self._ensure_model_pool(
                        input_size,
                        output_size,
                        split.validation_keys,
                        tolerances,
                        rng,
                        batch_size=batch_size,
                    )
                    models = self._storage.list_models()
                selected = models[int(rng.integers(0, len(models)))]
                selected_id = str(selected["model_id"])
                original = self._storage.load_model(selected_id)

                baseline_validation = self._evaluate_matrices_for_keys(
                    original,
                    split.validation_keys,
                    tolerances,
                    batch_size=batch_size,
                )
                validation_count = int(split.validation_keys.shape[0])
                steps = (
                    int(fixed_steps)
                    if amount_strategy == "fixed"
                    else int(validation_count * validation_count)
                )

                trained = train_matrices_batches(
                    lambda: self._iter_input_output_batches(
                        split.train_keys,
                        batch_size=batch_size,
                    ),
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

                trained_validation = self._evaluate_matrices_for_keys(
                    trained,
                    split.validation_keys,
                    tolerances,
                    batch_size=batch_size,
                )
                kept_matrices = original
                kept_metrics = baseline_validation
                message = "round reverted"
                if is_better(trained_validation, baseline_validation):
                    kept_matrices = trained
                    kept_metrics = trained_validation
                    message = "round improved"
                    self._storage.save_model(
                        kept_matrices,
                        model_id=selected_id,
                        metrics=kept_metrics,
                    )
                    child = mutate_matrices(kept_matrices, rng)
                    child_metrics = self._evaluate_matrices_for_keys(
                        child,
                        split.validation_keys,
                        tolerances,
                        batch_size=batch_size,
                    )
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
                        metrics=kept_metrics,
                    )

                max_models = max(1, math.ceil(math.sqrt(sample_count)))
                self._storage.prune_models(max_models)
                rounds_completed += 1
                self._storage.add_log(
                    job_id,
                    message=message,
                    model_id=selected_id,
                    validation_count=validation_count,
                    steps=steps,
                    accepted_rate=float(kept_metrics["accepted_rate"]),
                    mse=float(kept_metrics["mse"]),
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

    def _iter_input_output_batches(
        self,
        sample_keys: np.ndarray,
        *,
        batch_size: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for batch in self._storage.iter_key_batches(sample_keys, batch_size=batch_size):
            yield batch.inputs, batch.outputs

    def _evaluate_matrices_for_keys(
        self,
        matrices: Sequence[np.ndarray],
        sample_keys: np.ndarray,
        tolerances: np.ndarray,
        *,
        batch_size: int,
    ) -> dict[str, float | int]:
        return evaluate_matrices_batches(
            self._iter_input_output_batches(sample_keys, batch_size=batch_size),
            matrices,
            tolerances,
        )

    def _ensure_model_pool(
        self,
        input_size: int,
        output_size: int,
        validation_keys: np.ndarray,
        tolerances: np.ndarray,
        rng: np.random.Generator,
        *,
        batch_size: int,
    ) -> None:
        if self._storage.list_models():
            return
        matrix = random_matrix(input_size + 1, output_size, rng)
        matrices = [matrix]
        metrics = self._evaluate_matrices_for_keys(
            matrices,
            validation_keys,
            tolerances,
            batch_size=batch_size,
        )
        self._storage.save_model(matrices, metrics=metrics)
