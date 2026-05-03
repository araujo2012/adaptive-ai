from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetBatch:
    sample_keys: np.ndarray
    sample_ids: list[Any]
    inputs: np.ndarray
    outputs: np.ndarray


@dataclass(frozen=True)
class SampleBatch:
    sample_keys: np.ndarray
    sample_ids: list[Any]
    inputs: np.ndarray
    outputs: np.ndarray


@dataclass(frozen=True)
class TrainingSplit:
    train_keys: np.ndarray
    validation_keys: np.ndarray
    train_path: Path
    validation_path: Path
    train_ratio: float
    seed: int | None


class DatasetView:
    def __init__(
        self,
        *,
        sample_count: int,
        input_size: int,
        output_size: int,
        batch_iterator: Callable[[int], Iterator[DatasetBatch]],
    ):
        self.sample_count = sample_count
        self.input_size = input_size
        self.output_size = output_size
        self._batch_iterator = batch_iterator

    def iter_batches(self, *, batch_size: int = 1024) -> Iterator[DatasetBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        yield from self._batch_iterator(batch_size)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in {"inputs", "outputs"}:
            raise KeyError(key)

        batches = list(self.iter_batches(batch_size=max(self.sample_count, 1)))
        if batches:
            return np.vstack([getattr(batch, key) for batch in batches])

        width = self.input_size if key == "inputs" else self.output_size
        return np.empty((0, width), dtype=np.float64)
