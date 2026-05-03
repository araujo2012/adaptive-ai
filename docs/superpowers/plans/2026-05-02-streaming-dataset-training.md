# Streaming Dataset Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace full-dataset `.npz` storage and full-array training with chunked append-only storage, opaque sample IDs, persisted random train/validation splits, and batch-based training/evaluation.

**Architecture:** Keep SQLite as metadata source of truth and NumPy as the array format. Store input/output samples in append-only chunk directories, address rows with compact internal keys, expose dataset access through a streaming view, and make training consume key batches instead of full arrays.

**Tech Stack:** Python 3.11+, NumPy 2.x, SQLite via `sqlite3`, pytest.

---

## File Structure

- Create `adaptive_ai/dataset.py`
  - Owns lightweight data containers: `DatasetBatch`, `DatasetView`, `SampleBatch`, and `TrainingSplit`.
  - `DatasetView.iter_batches()` delegates to storage and never holds the complete collection.
- Modify `adaptive_ai/storage.py`
  - Adds chunk directories under `.adaptive_ai/arrays/dataset/`.
  - Adds SQLite tables for chunks, samples, ingestions, and training splits.
  - Replaces canonical `save_dataset()`/`load_dataset()` usage with `replace_dataset()`, `append_dataset()`, `iter_dataset_batches()`, `load_samples_by_ids()`, and `iter_key_batches()`.
- Modify `adaptive_ai/math.py`
  - Adds streaming metric accumulation and batch training helpers while keeping direct `train_matrices()` available for caller-provided arrays.
- Modify `adaptive_ai/api.py`
  - Adds `sample_ids`, `train_ratio`, and `batch_size` arguments.
  - Makes `get_dataset()` return `DatasetView`.
  - Rewrites background training to use compact keys and batch loaders.
- Modify `tests/test_adaptive_ai.py`
  - Updates existing API expectations from full-array dataset loading to streaming views.
- Create `tests/test_streaming_dataset.py`
  - Covers chunking, sample IDs, idempotency, sample lookup, and split persistence.
- Create `tests/test_streaming_math.py`
  - Covers streaming metrics and batch training helpers.
- Modify `README.md`
  - Documents chunked storage, sample IDs, streaming dataset access, and training memory rules.

---

### Task 1: Add Dataset View Types And Chunked Storage Schema

**Files:**
- Create: `adaptive_ai/dataset.py`
- Modify: `adaptive_ai/storage.py`
- Modify: `adaptive_ai/api.py`
- Test: `tests/test_streaming_dataset.py`

- [ ] **Step 1: Write the failing dataset-view test**

Create `tests/test_streaming_dataset.py` with:

```python
import numpy as np

from adaptive_ai import AdaptiveAI


def test_set_and_put_input_output_create_chunked_collection(tmp_path):
    ai = AdaptiveAI(path=tmp_path)

    ai.set_input_output([[0, 0], [1, 1]], [[0], [1]])
    ai.put_input_output([[1, 0]], [[1]])

    dataset = ai.get_dataset()
    assert dataset.sample_count == 3
    assert dataset.input_size == 2
    assert dataset.output_size == 1
    assert not (tmp_path / ".adaptive_ai" / "arrays" / "dataset.npz").exists()

    chunk_dirs = sorted((tmp_path / ".adaptive_ai" / "arrays" / "dataset" / "chunks").iterdir())
    assert len(chunk_dirs) == 2
    assert all((chunk_dir / "inputs.npy").exists() for chunk_dir in chunk_dirs)
    assert all((chunk_dir / "outputs.npy").exists() for chunk_dir in chunk_dirs)
    assert all((chunk_dir / "sample_keys.npy").exists() for chunk_dir in chunk_dirs)

    batches = list(dataset.iter_batches(batch_size=2))
    assert [batch.inputs.shape[0] for batch in batches] == [2, 1]
    np.testing.assert_allclose(
        np.vstack([batch.inputs for batch in batches]),
        [[0, 0], [1, 1], [1, 0]],
    )
    np.testing.assert_allclose(
        np.vstack([batch.outputs for batch in batches]),
        [[0], [1], [1]],
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_streaming_dataset.py::test_set_and_put_input_output_create_chunked_collection -q
```

Expected: FAIL because `get_dataset()` currently returns a dict from `dataset.npz` and no chunked collection exists.

- [ ] **Step 3: Add dataset data containers**

Create `adaptive_ai/dataset.py`:

```python
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
```

- [ ] **Step 4: Add chunk paths and SQLite schema**

Modify `adaptive_ai/storage.py` imports:

```python
import json
import pickle
import shutil
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
```

Add dataset paths in `Storage.__init__` after `self.arrays_path`:

```python
self.dataset_path = self.arrays_path / "dataset"
self.chunks_path = self.dataset_path / "chunks"
self.job_splits_path = self.dataset_path / "job_splits"
```

Create directories in `Storage.__init__`:

```python
self.dataset_path.mkdir(parents=True, exist_ok=True)
self.chunks_path.mkdir(parents=True, exist_ok=True)
self.job_splits_path.mkdir(parents=True, exist_ok=True)
```

Extend `_init_db()` with these tables inside the existing script:

```sql
CREATE TABLE IF NOT EXISTS dataset_chunks (
    id TEXT PRIMARY KEY,
    input_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    sample_keys_path TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    committed_at TEXT
);

CREATE TABLE IF NOT EXISTS dataset_samples (
    key INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id_blob BLOB NOT NULL UNIQUE,
    content_fingerprint TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES dataset_chunks(id)
);

CREATE TABLE IF NOT EXISTS dataset_ingestions (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    committed_rows INTEGER NOT NULL DEFAULT 0,
    skipped_rows INTEGER NOT NULL DEFAULT 0,
    conflict_rows INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS training_splits (
    job_id TEXT PRIMARY KEY,
    seed INTEGER,
    train_ratio REAL NOT NULL,
    train_path TEXT NOT NULL,
    validation_path TEXT NOT NULL,
    train_count INTEGER NOT NULL,
    validation_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);
```

Add cleanup after `mark_interrupted_jobs()` is called in `__init__`:

```python
self.cleanup_pending_dataset_chunks()
```

Add the cleanup method:

```python
def cleanup_pending_dataset_chunks(self) -> None:
    with self._lock, self._connect() as connection:
        rows = connection.execute(
            "SELECT id, input_path, output_path, sample_keys_path FROM dataset_chunks WHERE status != 'committed'"
        ).fetchall()
        connection.execute("DELETE FROM dataset_samples WHERE status != 'committed'")
        connection.execute("DELETE FROM dataset_chunks WHERE status != 'committed'")
    for row in rows:
        chunk_dir = Path(str(row["input_path"])).parent
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
```

- [ ] **Step 5: Add chunked write and view methods**

Add these helper functions near the bottom of `adaptive_ai/storage.py`:

```python
def _sample_id_to_blob(sample_id: object) -> bytes:
    try:
        return pickle.dumps(sample_id, protocol=5)
    except Exception as exc:
        raise ValueError("sample_ids must contain pickle-serializable values") from exc


def _sample_id_from_blob(blob: bytes) -> Any:
    return pickle.loads(blob)


def _generated_sample_id() -> str:
    return str(uuid.uuid4())


def _fingerprint_row(input_row: np.ndarray, output_row: np.ndarray) -> str:
    import hashlib

    input_bytes = np.ascontiguousarray(input_row, dtype=np.float64).tobytes()
    output_bytes = np.ascontiguousarray(output_row, dtype=np.float64).tobytes()
    digest = hashlib.sha256()
    digest.update(len(input_bytes).to_bytes(8, "big"))
    digest.update(input_bytes)
    digest.update(len(output_bytes).to_bytes(8, "big"))
    digest.update(output_bytes)
    return digest.hexdigest()
```

Add methods inside `Storage`:

```python
def clear_dataset(self) -> None:
    with self._lock, self._connect() as connection:
        connection.execute("DELETE FROM training_splits")
        connection.execute("DELETE FROM dataset_samples")
        connection.execute("DELETE FROM dataset_chunks")
        connection.execute("DELETE FROM dataset_ingestions")
    if self.dataset_path.exists():
        shutil.rmtree(self.dataset_path)
    self.dataset_path.mkdir(parents=True, exist_ok=True)
    self.chunks_path.mkdir(parents=True, exist_ok=True)
    self.job_splits_path.mkdir(parents=True, exist_ok=True)


def replace_dataset(
    self,
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    sample_ids: Iterable[object] | None = None,
) -> dict[str, int]:
    self.clear_dataset()
    return self.append_dataset(inputs, outputs, sample_ids=sample_ids)


def append_dataset(
    self,
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    sample_ids: Iterable[object] | None = None,
) -> dict[str, int]:
    ids = list(sample_ids) if sample_ids is not None else [_generated_sample_id() for _ in range(inputs.shape[0])]
    if len(ids) != inputs.shape[0]:
        raise ValueError("sample_ids must contain one id per sample")

    ingestion_id = str(uuid.uuid4())
    now = utc_now()
    new_rows: list[tuple[object, bytes, str, np.ndarray, np.ndarray]] = []
    skipped_rows = 0
    conflict_rows = 0

    with self._lock, self._connect() as connection:
        connection.execute(
            """
            INSERT INTO dataset_ingestions(id, status, started_at)
            VALUES(?, 'running', ?)
            """,
            (ingestion_id, now),
        )
        for sample_id, input_row, output_row in zip(ids, inputs, outputs, strict=True):
            sample_blob = _sample_id_to_blob(sample_id)
            fingerprint = _fingerprint_row(input_row, output_row)
            existing = connection.execute(
                """
                SELECT content_fingerprint
                FROM dataset_samples
                WHERE sample_id_blob = ? AND status = 'committed'
                """,
                (sample_blob,),
            ).fetchone()
            if existing is not None:
                if str(existing["content_fingerprint"]) == fingerprint:
                    skipped_rows += 1
                    continue
                conflict_rows += 1
                connection.execute(
                    """
                    UPDATE dataset_ingestions
                    SET status = 'failed', conflict_rows = ?, finished_at = ?, error = ?
                    WHERE id = ?
                    """,
                    (
                        conflict_rows,
                        utc_now(),
                        "conflicting sample_id with different input/output content",
                        ingestion_id,
                    ),
                )
                raise ValueError("conflicting sample_id with different input/output content")
            new_rows.append((sample_id, sample_blob, fingerprint, input_row, output_row))

    if not new_rows:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE dataset_ingestions
                SET status = 'committed', skipped_rows = ?, finished_at = ?
                WHERE id = ?
                """,
                (skipped_rows, utc_now(), ingestion_id),
            )
        return {"committed_rows": 0, "skipped_rows": skipped_rows, "conflict_rows": conflict_rows}

    chunk_id = str(uuid.uuid4())
    chunk_dir = self.chunks_path / chunk_id
    chunk_dir.mkdir(parents=True, exist_ok=False)
    input_path = chunk_dir / "inputs.npy"
    output_path = chunk_dir / "outputs.npy"
    sample_keys_path = chunk_dir / "sample_keys.npy"
    chunk_inputs = np.asarray([row[3] for row in new_rows], dtype=np.float64)
    chunk_outputs = np.asarray([row[4] for row in new_rows], dtype=np.float64)
    np.save(input_path, chunk_inputs)
    np.save(output_path, chunk_outputs)

    sample_keys: list[int] = []
    with self._lock, self._connect() as connection:
        connection.execute(
            """
            INSERT INTO dataset_chunks(
                id, input_path, output_path, sample_keys_path, row_count, status, created_at
            )
            VALUES(?, ?, ?, ?, ?, 'pending', ?)
            """,
            (chunk_id, str(input_path), str(output_path), str(sample_keys_path), len(new_rows), utc_now()),
        )
        for row_index, (_, sample_blob, fingerprint, _, _) in enumerate(new_rows):
            cursor = connection.execute(
                """
                INSERT INTO dataset_samples(
                    sample_id_blob, content_fingerprint, chunk_id, row_index, status, created_at
                )
                VALUES(?, ?, ?, ?, 'pending', ?)
                """,
                (sample_blob, fingerprint, chunk_id, row_index, utc_now()),
            )
            sample_keys.append(int(cursor.lastrowid))
        np.save(sample_keys_path, np.asarray(sample_keys, dtype=np.uint64))
        connection.execute(
            "UPDATE dataset_samples SET status = 'committed' WHERE chunk_id = ?",
            (chunk_id,),
        )
        connection.execute(
            "UPDATE dataset_chunks SET status = 'committed', committed_at = ? WHERE id = ?",
            (utc_now(), chunk_id),
        )
        connection.execute(
            """
            UPDATE dataset_ingestions
            SET status = 'committed', committed_rows = ?, skipped_rows = ?, conflict_rows = ?, finished_at = ?
            WHERE id = ?
            """,
            (len(new_rows), skipped_rows, conflict_rows, utc_now(), ingestion_id),
        )

    return {
        "committed_rows": len(new_rows),
        "skipped_rows": skipped_rows,
        "conflict_rows": conflict_rows,
    }


def get_sample_count(self) -> int:
    with self._lock, self._connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM dataset_samples WHERE status = 'committed'"
        ).fetchone()
    return int(row["count"])
```

Add the batch iterator:

```python
def iter_dataset_batches(self, *, batch_size: int) -> Iterable["DatasetBatch"]:
    from .dataset import DatasetBatch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    with self._lock, self._connect() as connection:
        rows = connection.execute(
            """
            SELECT key
            FROM dataset_samples
            WHERE status = 'committed'
            ORDER BY key
            """
        ).fetchall()
    keys = np.asarray([int(row["key"]) for row in rows], dtype=np.uint64)
    for start in range(0, keys.shape[0], batch_size):
        yield self.load_samples_by_keys(keys[start : start + batch_size])
```

- [ ] **Step 6: Add basic key batch loading used by `iter_dataset_batches()`**

Add `load_samples_by_keys()` to `Storage`:

```python
def load_samples_by_keys(self, sample_keys: np.ndarray) -> "SampleBatch":
    from .dataset import SampleBatch

    keys = np.asarray(sample_keys, dtype=np.uint64)
    if keys.ndim != 1:
        raise ValueError("sample_keys must be a 1D array")
    if keys.shape[0] == 0:
        dimensions = self.get_dimensions()
        if dimensions is None:
            raise ValueError("dataset dimensions have not been initialized")
        input_size, output_size = dimensions
        return SampleBatch(
            sample_keys=keys,
            sample_ids=[],
            inputs=np.empty((0, input_size), dtype=np.float64),
            outputs=np.empty((0, output_size), dtype=np.float64),
        )

    bind_marks = ",".join("?" for _ in keys)
    with self._lock, self._connect() as connection:
        rows = connection.execute(
            f"""
            SELECT key, sample_id_blob, chunk_id, row_index
            FROM dataset_samples
            WHERE status = 'committed' AND key IN ({bind_marks})
            """,
            [int(key) for key in keys],
        ).fetchall()

    by_key = {int(row["key"]): row for row in rows}
    missing = [int(key) for key in keys if int(key) not in by_key]
    if missing:
        raise ValueError(f"sample keys were not found: {missing[:3]}")

    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    sample_ids: list[Any] = []
    chunk_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in keys:
        row = by_key[int(key)]
        chunk_id = str(row["chunk_id"])
        if chunk_id not in chunk_cache:
            with self._lock, self._connect() as connection:
                chunk = connection.execute(
                    """
                    SELECT input_path, output_path
                    FROM dataset_chunks
                    WHERE id = ? AND status = 'committed'
                    """,
                    (chunk_id,),
                ).fetchone()
            if chunk is None:
                raise ValueError(f"dataset chunk {chunk_id} was not found")
            chunk_cache[chunk_id] = (
                np.load(str(chunk["input_path"]), mmap_mode="r"),
                np.load(str(chunk["output_path"]), mmap_mode="r"),
            )
        chunk_inputs, chunk_outputs = chunk_cache[chunk_id]
        row_index = int(row["row_index"])
        inputs.append(np.asarray(chunk_inputs[row_index], dtype=np.float64))
        outputs.append(np.asarray(chunk_outputs[row_index], dtype=np.float64))
        sample_ids.append(_sample_id_from_blob(bytes(row["sample_id_blob"])))

    return SampleBatch(
        sample_keys=keys,
        sample_ids=sample_ids,
        inputs=np.asarray(inputs, dtype=np.float64),
        outputs=np.asarray(outputs, dtype=np.float64),
    )
```

- [ ] **Step 7: Wire `AdaptiveAI` to the view**

Modify imports in `adaptive_ai/api.py`:

```python
from .dataset import DatasetView, SampleBatch
```

Update signatures and methods:

```python
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
    dimensions = self._storage.get_dimensions()
    if dimensions is None:
        raise ValueError("dataset dimensions have not been initialized")
    input_size, output_size = dimensions
    return DatasetView(
        sample_count=self._storage.get_sample_count(),
        input_size=input_size,
        output_size=output_size,
        batch_iterator=lambda batch_size: self._storage.iter_dataset_batches(batch_size=batch_size),
    )
```

- [ ] **Step 8: Run test to verify it passes**

Run:

```powershell
python -m pytest tests/test_streaming_dataset.py::test_set_and_put_input_output_create_chunked_collection -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```powershell
git add adaptive_ai/dataset.py adaptive_ai/storage.py adaptive_ai/api.py tests/test_streaming_dataset.py
git commit -m "feat: add chunked dataset collection"
```

---

### Task 2: Add Opaque Sample IDs And Idempotent Appends

**Files:**
- Modify: `adaptive_ai/storage.py`
- Modify: `adaptive_ai/api.py`
- Modify: `tests/test_streaming_dataset.py`

- [ ] **Step 1: Write failing sample ID and idempotency tests**

Append to `tests/test_streaming_dataset.py`:

```python
def test_sample_ids_are_exposed_and_duplicate_identical_rows_are_idempotent(tmp_path):
    ai = AdaptiveAI(path=tmp_path)

    ai.set_input_output([[0], [1]], [[0], [1]], sample_ids=["2026-05-02T00:00:00", 123])
    ai.put_input_output([[1]], [[1]], sample_ids=[123])

    dataset = ai.get_dataset()
    assert dataset.sample_count == 2
    batches = list(dataset.iter_batches(batch_size=10))
    assert batches[0].sample_ids == ["2026-05-02T00:00:00", 123]

    chunk_dirs = sorted((tmp_path / ".adaptive_ai" / "arrays" / "dataset" / "chunks").iterdir())
    assert len(chunk_dirs) == 1


def test_duplicate_sample_id_with_different_content_fails(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0]], [[0]], sample_ids=["same-id"])

    with pytest.raises(ValueError, match="conflicting sample_id"):
        ai.put_input_output([[1]], [[0]], sample_ids=["same-id"])
```

Add the missing import at the top:

```python
import pytest
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_streaming_dataset.py::test_sample_ids_are_exposed_and_duplicate_identical_rows_are_idempotent tests/test_streaming_dataset.py::test_duplicate_sample_id_with_different_content_fails -q
```

Expected: FAIL if Task 1 did not fully implement sample ID round-tripping and idempotent skip behavior.

- [ ] **Step 3: Tighten idempotency behavior**

Ensure `Storage.append_dataset()` does all of the following exactly:

```python
existing = connection.execute(
    """
    SELECT content_fingerprint
    FROM dataset_samples
    WHERE sample_id_blob = ? AND status = 'committed'
    """,
    (sample_blob,),
).fetchone()
if existing is not None:
    if str(existing["content_fingerprint"]) == fingerprint:
        skipped_rows += 1
        continue
    conflict_rows += 1
    connection.execute(
        """
        UPDATE dataset_ingestions
        SET status = 'failed', conflict_rows = ?, finished_at = ?, error = ?
        WHERE id = ?
        """,
        (
            conflict_rows,
            utc_now(),
            "conflicting sample_id with different input/output content",
            ingestion_id,
        ),
    )
    raise ValueError("conflicting sample_id with different input/output content")
```

Ensure the no-new-rows branch does not create a chunk:

```python
if not new_rows:
    with self._lock, self._connect() as connection:
        connection.execute(
            """
            UPDATE dataset_ingestions
            SET status = 'committed', skipped_rows = ?, finished_at = ?
            WHERE id = ?
            """,
            (skipped_rows, utc_now(), ingestion_id),
        )
    return {"committed_rows": 0, "skipped_rows": skipped_rows, "conflict_rows": conflict_rows}
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_streaming_dataset.py::test_sample_ids_are_exposed_and_duplicate_identical_rows_are_idempotent tests/test_streaming_dataset.py::test_duplicate_sample_id_with_different_content_fails -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add adaptive_ai/storage.py adaptive_ai/api.py tests/test_streaming_dataset.py
git commit -m "feat: support idempotent sample ids"
```

---

### Task 3: Add Public Sample Lookup And Internal Key Batch Iteration

**Files:**
- Modify: `adaptive_ai/storage.py`
- Modify: `adaptive_ai/api.py`
- Modify: `tests/test_streaming_dataset.py`

- [ ] **Step 1: Write failing lookup tests**

Append to `tests/test_streaming_dataset.py`:

```python
def test_get_samples_returns_requested_sample_ids_in_requested_order(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0, 0], [1, 1], [2, 2]],
        [[0], [1], [1]],
        sample_ids=["a", "b", "c"],
    )

    samples = ai.get_samples(["c", "a"])

    assert samples.sample_ids == ["c", "a"]
    np.testing.assert_allclose(samples.inputs, [[2, 2], [0, 0]])
    np.testing.assert_allclose(samples.outputs, [[1], [0]])


def test_missing_sample_id_fails_clearly(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0]], [[0]], sample_ids=["a"])

    with pytest.raises(ValueError, match="sample_ids were not found"):
        ai.get_samples(["missing"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_streaming_dataset.py::test_get_samples_returns_requested_sample_ids_in_requested_order tests/test_streaming_dataset.py::test_missing_sample_id_fails_clearly -q
```

Expected: FAIL because `AdaptiveAI.get_samples()` does not exist.

- [ ] **Step 3: Add storage lookup by original sample IDs**

Add to `Storage`:

```python
def load_samples_by_ids(self, sample_ids: Iterable[object]) -> "SampleBatch":
    ids = list(sample_ids)
    blobs = [_sample_id_to_blob(sample_id) for sample_id in ids]
    if not blobs:
        return self.load_samples_by_keys(np.asarray([], dtype=np.uint64))
    bind_marks = ",".join("?" for _ in blobs)
    with self._lock, self._connect() as connection:
        rows = connection.execute(
            f"""
            SELECT key, sample_id_blob
            FROM dataset_samples
            WHERE status = 'committed' AND sample_id_blob IN ({bind_marks})
            """,
            blobs,
        ).fetchall()
    key_by_blob = {bytes(row["sample_id_blob"]): int(row["key"]) for row in rows}
    missing = [sample_id for sample_id, blob in zip(ids, blobs, strict=True) if blob not in key_by_blob]
    if missing:
        raise ValueError(f"sample_ids were not found: {missing[:3]}")
    keys = np.asarray([key_by_blob[blob] for blob in blobs], dtype=np.uint64)
    return self.load_samples_by_keys(keys)
```

Add internal key iteration:

```python
def iter_key_batches(self, sample_keys: np.ndarray, *, batch_size: int) -> Iterable["SampleBatch"]:
    keys = np.asarray(sample_keys, dtype=np.uint64)
    if keys.ndim != 1:
        raise ValueError("sample_keys must be a 1D array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, keys.shape[0], batch_size):
        yield self.load_samples_by_keys(keys[start : start + batch_size])
```

- [ ] **Step 4: Add public API method**

In `AdaptiveAI`, add:

```python
def get_samples(self, sample_ids: Sequence[object]) -> SampleBatch:
    return self._storage.load_samples_by_ids(sample_ids)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_streaming_dataset.py::test_get_samples_returns_requested_sample_ids_in_requested_order tests/test_streaming_dataset.py::test_missing_sample_id_fails_clearly -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add adaptive_ai/storage.py adaptive_ai/api.py tests/test_streaming_dataset.py
git commit -m "feat: load samples by id or key batches"
```

---

### Task 4: Add Streaming Metrics And Batch Training Helpers

**Files:**
- Modify: `adaptive_ai/math.py`
- Create: `tests/test_streaming_math.py`

- [ ] **Step 1: Write failing streaming math tests**

Create `tests/test_streaming_math.py`:

```python
import numpy as np

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_streaming_math.py -q
```

Expected: FAIL because `evaluate_matrices_batches()` and `train_matrices_batches()` do not exist.

- [ ] **Step 3: Add streaming metric accumulator**

Add imports in `adaptive_ai/math.py`:

```python
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
```

Add after `evaluate_matrices()`:

```python
@dataclass
class MetricAccumulator:
    accepted_count: int = 0
    sample_count: int = 0
    squared_error_sum: float = 0.0
    output_count: int | None = None

    def update(
        self,
        predicted: object,
        expected: object,
        tolerances: Sequence[float],
    ) -> None:
        predicted_array = as_2d_float64(predicted, name="predicted", one_dim_as_column=True)
        expected_array = as_2d_float64(expected, name="expected", one_dim_as_column=True)
        if predicted_array.shape != expected_array.shape:
            raise ValueError("predicted and expected must have the same shape")
        tolerance_array = np.asarray(tolerances, dtype=np.float64)
        if tolerance_array.ndim != 1 or tolerance_array.shape[0] != expected_array.shape[1]:
            raise ValueError("tolerances must match the output dimension")
        if (tolerance_array < 0.0).any() or not np.isfinite(tolerance_array).all():
            raise ValueError("tolerances must contain non-negative finite numbers")
        if self.output_count is None:
            self.output_count = int(expected_array.shape[1])
        elif self.output_count != int(expected_array.shape[1]):
            raise ValueError("all batches must have the same output dimension")

        differences = np.abs(predicted_array - expected_array)
        accepted_mask = (differences <= tolerance_array).all(axis=1)
        self.accepted_count += int(np.sum(accepted_mask))
        self.sample_count += int(expected_array.shape[0])
        self.squared_error_sum += float(np.sum(np.square(predicted_array - expected_array)))

    def metrics(self) -> dict[str, float | int]:
        if self.sample_count == 0:
            return {"accepted_count": 0, "accepted_rate": 0.0, "mse": 0.0}
        output_count = 1 if self.output_count is None else self.output_count
        return {
            "accepted_count": self.accepted_count,
            "accepted_rate": float(self.accepted_count / self.sample_count),
            "mse": float(self.squared_error_sum / (self.sample_count * output_count)),
        }


def evaluate_matrices_batches(
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    matrices: Sequence[np.ndarray],
    tolerances: Sequence[float],
) -> dict[str, float | int]:
    accumulator = MetricAccumulator()
    for input_batch, output_batch in batches:
        predictions = predict(input_batch, matrices)
        accumulator.update(predictions, output_batch, tolerances)
    return accumulator.metrics()
```

- [ ] **Step 4: Add batch training helper**

Add after `train_matrices()`:

```python
def train_matrices_batches(
    batch_factory: Callable[[], Iterable[tuple[np.ndarray, np.ndarray]]],
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

    trained = [np.array(matrix, dtype=np.float64, copy=True) for matrix in matrices]
    iterator = iter(batch_factory())
    saw_batch = False

    for _ in range(steps):
        if stop_checker is not None and stop_checker():
            break
        try:
            input_batch, output_batch = next(iterator)
        except StopIteration:
            iterator = iter(batch_factory())
            try:
                input_batch, output_batch = next(iterator)
            except StopIteration as exc:
                if saw_batch:
                    break
                raise ValueError("training requires at least one sample") from exc
        saw_batch = True
        trained = train_matrices(
            input_batch,
            output_batch,
            trained,
            steps=1,
            learning_rate=learning_rate,
            stop_checker=stop_checker,
        )

    return trained
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_streaming_math.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add adaptive_ai/math.py tests/test_streaming_math.py
git commit -m "feat: add streaming math helpers"
```

---

### Task 5: Persist Random Train/Validation Splits With Compact Keys

**Files:**
- Modify: `adaptive_ai/storage.py`
- Modify: `tests/test_streaming_dataset.py`

- [ ] **Step 1: Write failing split persistence test**

Append to `tests/test_streaming_dataset.py`:

```python
def test_training_split_materializes_random_compact_keys_and_persists_them(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    inputs = [[float(index)] for index in range(10)]
    outputs = [[float(index % 2)] for index in range(10)]
    ai.set_input_output(inputs, outputs, sample_ids=[f"ts-{index}" for index in range(10)])
    job_id = ai._storage.create_job(
        max_seconds=1.0,
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.1,
        train_ratio=0.8,
        batch_size=2,
    )

    split = ai._storage.get_or_create_training_split(job_id, seed=42, train_ratio=0.8)
    loaded = ai._storage.get_or_create_training_split(job_id, seed=999, train_ratio=0.5)

    assert split.train_keys.dtype == np.uint64
    assert split.validation_keys.dtype == np.uint64
    assert split.train_keys.shape[0] == 8
    assert split.validation_keys.shape[0] == 2
    assert split.train_path.exists()
    assert split.validation_path.exists()
    np.testing.assert_array_equal(loaded.train_keys, split.train_keys)
    np.testing.assert_array_equal(loaded.validation_keys, split.validation_keys)
    assert set(split.train_keys.tolist()).isdisjoint(set(split.validation_keys.tolist()))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_streaming_dataset.py::test_training_split_materializes_random_compact_keys_and_persists_them -q
```

Expected: FAIL because `create_job()` does not accept `train_ratio` and `batch_size`, and split storage does not exist.

- [ ] **Step 3: Add job columns and create_job arguments**

In `Storage._init_db()`, add columns to the `jobs` table definition:

```sql
train_ratio REAL NOT NULL DEFAULT 0.8,
batch_size INTEGER NOT NULL DEFAULT 1024,
train_cursor INTEGER NOT NULL DEFAULT 0,
validation_cursor INTEGER NOT NULL DEFAULT 0
```

Add a helper after `_init_db()`:

```python
def _ensure_column(self, connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row["name"]) for row in rows}
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
```

Call it at the end of `_init_db()`:

```python
self._ensure_column(connection, "jobs", "train_ratio", "REAL NOT NULL DEFAULT 0.8")
self._ensure_column(connection, "jobs", "batch_size", "INTEGER NOT NULL DEFAULT 1024")
self._ensure_column(connection, "jobs", "train_cursor", "INTEGER NOT NULL DEFAULT 0")
self._ensure_column(connection, "jobs", "validation_cursor", "INTEGER NOT NULL DEFAULT 0")
```

Update `create_job()` signature and insert:

```python
def create_job(
    self,
    *,
    max_seconds: float,
    amount_strategy: str,
    fixed_steps: int | None,
    learning_rate: float,
    train_ratio: float = 0.8,
    batch_size: int = 1024,
) -> str:
    job_id = str(uuid.uuid4())
    with self._lock, self._connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs(
                id, status, max_seconds, amount_strategy, fixed_steps,
                learning_rate, train_ratio, batch_size, started_at, rounds_completed
            )
            VALUES(?, 'running', ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                job_id,
                max_seconds,
                amount_strategy,
                fixed_steps,
                learning_rate,
                train_ratio,
                batch_size,
                utc_now(),
            ),
        )
    return job_id
```

Update `_job_row_to_dict()`:

```python
"train_ratio": float(row["train_ratio"]),
"batch_size": int(row["batch_size"]),
"train_cursor": int(row["train_cursor"]),
"validation_cursor": int(row["validation_cursor"]),
```

- [ ] **Step 4: Add split creation**

Add to `Storage`:

```python
def list_sample_keys(self) -> np.ndarray:
    with self._lock, self._connect() as connection:
        rows = connection.execute(
            """
            SELECT key
            FROM dataset_samples
            WHERE status = 'committed'
            ORDER BY key
            """
        ).fetchall()
    return np.asarray([int(row["key"]) for row in rows], dtype=np.uint64)


def get_or_create_training_split(
    self,
    job_id: str,
    *,
    seed: int | None,
    train_ratio: float,
) -> "TrainingSplit":
    from .dataset import TrainingSplit

    with self._lock, self._connect() as connection:
        row = connection.execute(
            "SELECT * FROM training_splits WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    if row is not None:
        train_path = Path(str(row["train_path"]))
        validation_path = Path(str(row["validation_path"]))
        return TrainingSplit(
            train_keys=np.load(train_path),
            validation_keys=np.load(validation_path),
            train_path=train_path,
            validation_path=validation_path,
            train_ratio=float(row["train_ratio"]),
            seed=None if row["seed"] is None else int(row["seed"]),
        )

    keys = self.list_sample_keys()
    sample_count = int(keys.shape[0])
    if sample_count < 2:
        raise ValueError("training requires at least two dataset samples")
    if train_ratio <= 0.0 or train_ratio >= 1.0 or not np.isfinite(train_ratio):
        raise ValueError("train_ratio must be greater than 0 and less than 1")
    rng = np.random.default_rng(seed)
    shuffled = np.array(keys, dtype=np.uint64, copy=True)
    rng.shuffle(shuffled)
    train_count = min(max(1, int(sample_count * train_ratio)), sample_count - 1)
    train_keys = shuffled[:train_count]
    validation_keys = shuffled[train_count:]

    train_path = self.job_splits_path / f"{job_id}_train_keys.npy"
    validation_path = self.job_splits_path / f"{job_id}_validation_keys.npy"
    np.save(train_path, train_keys)
    np.save(validation_path, validation_keys)
    with self._lock, self._connect() as connection:
        connection.execute(
            """
            INSERT INTO training_splits(
                job_id, seed, train_ratio, train_path, validation_path,
                train_count, validation_count, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                seed,
                train_ratio,
                str(train_path),
                str(validation_path),
                int(train_keys.shape[0]),
                int(validation_keys.shape[0]),
                utc_now(),
            ),
        )
    return TrainingSplit(
        train_keys=train_keys,
        validation_keys=validation_keys,
        train_path=train_path,
        validation_path=validation_path,
        train_ratio=train_ratio,
        seed=seed,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```powershell
python -m pytest tests/test_streaming_dataset.py::test_training_split_materializes_random_compact_keys_and_persists_them -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add adaptive_ai/storage.py tests/test_streaming_dataset.py
git commit -m "feat: persist training splits"
```

---

### Task 6: Rewrite Background Training To Stream Batches

**Files:**
- Modify: `adaptive_ai/api.py`
- Modify: `adaptive_ai/math.py`
- Modify: `tests/test_adaptive_ai.py`

- [ ] **Step 1: Write failing no-full-dataset training test**

Append to `tests/test_adaptive_ai.py`:

```python
def test_training_job_streams_batches_without_loading_full_dataset(tmp_path, monkeypatch):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0], [1], [2], [3], [4]],
        [[0], [0], [1], [1], [1]],
        sample_ids=[f"sample-{index}" for index in range(5)],
    )

    def fail_if_called():
        raise AssertionError("training must not load the full dataset")

    monkeypatch.setattr(ai._storage, "load_dataset", fail_if_called, raising=False)

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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_adaptive_ai.py::test_training_job_streams_batches_without_loading_full_dataset -q
```

Expected: FAIL because `start_training()` does not accept `train_ratio` or `batch_size`, and training still loads all arrays.

- [ ] **Step 3: Update imports and start_training signature**

Modify imports in `adaptive_ai/api.py`:

```python
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
```

Update `start_training()` signature:

```python
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
```

Add validation:

```python
if train_ratio <= 0.0 or train_ratio >= 1.0 or not math.isfinite(train_ratio):
    raise ValueError("train_ratio must be greater than 0 and less than 1")
if batch_size <= 0:
    raise ValueError("batch_size must be positive")
```

Replace the dataset-size check:

```python
if self._storage.get_sample_count() < 2:
    raise ValueError("training requires at least two dataset samples")
```

Update `create_job()` call:

```python
job_id = self._storage.create_job(
    max_seconds=max_seconds,
    amount_strategy=amount_strategy,
    fixed_steps=fixed_steps,
    learning_rate=learning_rate,
    train_ratio=train_ratio,
    batch_size=batch_size,
)
```

Pass `train_ratio` and `batch_size` into `_run_training_job()`.

- [ ] **Step 4: Add streaming helper methods to AdaptiveAI**

Add inside `AdaptiveAI`:

```python
def _iter_input_output_batches(
    self,
    sample_keys: np.ndarray,
    *,
    batch_size: int,
):
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
```

- [ ] **Step 5: Rewrite `_ensure_model_pool()` to evaluate by keys**

Replace `_ensure_model_pool()` with:

```python
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
```

- [ ] **Step 6: Rewrite `_run_training_job()` around split keys**

Replace the beginning of `_run_training_job()` with:

```python
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
```

Inside the training loop, replace all full-array train/validation logic with:

```python
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
    lambda: self._iter_input_output_batches(split.train_keys, batch_size=batch_size),
    original,
    steps=steps,
    learning_rate=learning_rate,
    stop_checker=lambda: control.cancel_event.is_set()
    or control.pause_event.is_set()
    or time.monotonic() >= deadline,
)
```

After pause/cancel checks, evaluate and save:

```python
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
```

- [ ] **Step 7: Run the focused training test**

Run:

```powershell
python -m pytest tests/test_adaptive_ai.py::test_training_job_streams_batches_without_loading_full_dataset -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```powershell
git add adaptive_ai/api.py adaptive_ai/math.py tests/test_adaptive_ai.py
git commit -m "feat: stream background training"
```

---

### Task 7: Update Existing Tests For Streaming Dataset Semantics

**Files:**
- Modify: `tests/test_adaptive_ai.py`
- Modify: `adaptive_ai/api.py`
- Modify: `adaptive_ai/storage.py`

- [ ] **Step 1: Update the old dataset persistence test**

Replace `test_set_and_put_input_output_persist_float64_dataset()` in `tests/test_adaptive_ai.py` with:

```python
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
```

- [ ] **Step 2: Update training tests to pass batch parameters where useful**

In `test_fixed_training_job_completes_and_logs_progress()`, leave defaults unless the test is flaky. In `test_sample_square_strategy_records_validation_count_and_steps()`, keep the assertion:

```python
assert logs[0]["validation_count"] >= 1
assert logs[0]["steps"] == logs[0]["validation_count"] ** 2
```

In pause/cancel tests, lower `fixed_steps` if needed because each step now processes a batch:

```python
fixed_steps=50
```

- [ ] **Step 3: Run existing suite to reveal compatibility misses**

Run:

```powershell
python -m pytest -q
```

Expected: FAIL only for remaining references to full-array dataset behavior or `create_job()` signature mismatches.

- [ ] **Step 4: Fix `create_job()` call sites**

Any test or production call to `create_job()` must pass or rely on defaults:

```python
self._storage.create_job(
    max_seconds=max_seconds,
    amount_strategy=amount_strategy,
    fixed_steps=fixed_steps,
    learning_rate=learning_rate,
    train_ratio=train_ratio,
    batch_size=batch_size,
)
```

Direct test calls can use:

```python
job_id = ai._storage.create_job(
    max_seconds=1.0,
    amount_strategy="fixed",
    fixed_steps=1,
    learning_rate=0.1,
)
```

because defaults are now available.

- [ ] **Step 5: Ensure legacy full dataset loading is not used**

Keep `Storage.dataset_file()` only as a legacy path helper if tests or docs need it, but do not call `save_dataset()` or `load_dataset()` from `AdaptiveAI` internals. If those methods remain, implement them as explicit failures:

```python
def save_dataset(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
    raise RuntimeError("full-array dataset storage has been replaced by chunked storage")


def load_dataset(self) -> dict[str, np.ndarray]:
    raise RuntimeError("full-array dataset loading is not available; use streaming batches")
```

- [ ] **Step 6: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git add adaptive_ai/api.py adaptive_ai/storage.py tests/test_adaptive_ai.py
git commit -m "test: update dataset expectations for streaming"
```

---

### Task 8: Document Streaming Dataset Usage

**Files:**
- Modify: `README.md`
- Modify: `tests/test_adaptive_ai.py`

- [ ] **Step 1: Add a README smoke assertion test**

Append to `tests/test_adaptive_ai.py`:

```python
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
```

- [ ] **Step 2: Run smoke test to verify it passes before docs**

Run:

```powershell
python -m pytest tests/test_adaptive_ai.py::test_streaming_dataset_public_usage_smoke -q
```

Expected: PASS.

- [ ] **Step 3: Update persistence section in README**

Replace the old dataset persistence bullets with:

```markdown
- `adaptive_ai.sqlite3`: metadados de modelos, jobs, chunks, amostras e splits.
- `arrays/dataset/chunks/*`: chunks append-only com `inputs.npy`, `outputs.npy` e `sample_keys.npy`.
- `arrays/dataset/job_splits/*`: splits persistidos de treino/validacao por job.
- `models/*.npz`: matrizes dos modelos salvos.
```

- [ ] **Step 4: Update API table in README**

Use these rows:

```markdown
| `set_input_output(inputs, outputs, sample_ids=None)` | Cria uma nova collection chunked e grava as amostras recebidas. |
| `put_input_output(inputs, outputs, sample_ids=None)` | Acrescenta novas amostras em chunks sem reler chunks antigos. |
| `get_dataset()` | Retorna uma view streaming com metadados e `iter_batches()`. |
| `get_samples(sample_ids)` | Busca somente as amostras solicitadas por ID. |
| `start_training(...)` | Inicia um job de treinamento em segundo plano lendo batches por IDs compactos. |
```

- [ ] **Step 5: Add sample ID usage example**

Add this example under data format:

```python
ai.put_input_output(
    inputs=[[0.1, 0.2], [0.3, 0.4]],
    outputs=[[0], [1]],
    sample_ids=[1714665600000, "2026-05-02T12:01:00Z"],
)
```

Add this text:

```markdown
`sample_ids` e opcional. Quando informado, cada ID pertence ao projeto chamador e pode ser qualquer valor serializavel pelo Python. Se o mesmo ID for enviado novamente com o mesmo input/output, a escrita e idempotente. Se o mesmo ID vier com conteudo diferente, a chamada falha para evitar sobrescrita silenciosa.
```

- [ ] **Step 6: Add streaming access example**

Add:

```python
dataset = ai.get_dataset()

for batch in dataset.iter_batches(batch_size=1024):
    print(batch.sample_ids)
    print(batch.inputs.shape, batch.outputs.shape)

sample = ai.get_samples(["2026-05-02T12:01:00Z"])
```

- [ ] **Step 7: Run smoke test and full suite**

Run:

```powershell
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```powershell
git add README.md tests/test_adaptive_ai.py
git commit -m "docs: explain streaming datasets"
```

---

### Task 9: Final Verification And Regression Checks

**Files:**
- Inspect: `adaptive_ai/api.py`
- Inspect: `adaptive_ai/storage.py`
- Inspect: `adaptive_ai/math.py`
- Inspect: `README.md`
- Inspect: `tests/test_adaptive_ai.py`
- Inspect: `tests/test_streaming_dataset.py`
- Inspect: `tests/test_streaming_math.py`

- [ ] **Step 1: Search for forbidden full-dataset training calls**

Run:

```powershell
Get-ChildItem -Path adaptive_ai -Recurse -File -Include *.py | Select-String -Pattern 'load_dataset\\(|save_dataset\\(|np.concatenate\\(\\[dataset|train_mask|validation_inputs|validation_outputs|train_inputs|train_outputs'
```

Expected: no matches inside active training code. Matches inside explicit failing legacy methods are acceptable only if those methods are not called by `AdaptiveAI`.

- [ ] **Step 2: Search for pairwise or quadratic materialization**

Run:

```powershell
Get-ChildItem -Path adaptive_ai -Recurse -File -Include *.py | Select-String -Pattern 'meshgrid|cartesian|pairwise|for .* in .* for .* in|np.ones\\(.*shape\\[0\\]'
```

Expected: no dataset-pair generation or full dataset masks.

- [ ] **Step 3: Run focused streaming tests**

Run:

```powershell
python -m pytest tests/test_streaming_dataset.py tests/test_streaming_math.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full suite**

Run:

```powershell
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 5: Review git diff**

Run:

```powershell
git diff --stat HEAD
git diff -- adaptive_ai tests README.md
```

Expected: changes are limited to streaming dataset storage, training, tests, and docs.

- [ ] **Step 6: Commit final cleanup if needed**

If Step 5 reveals minor cleanup changes after the previous commits, run:

```powershell
git add adaptive_ai tests README.md
git commit -m "chore: finalize streaming dataset training"
```

Expected: a commit is created only if there are remaining tracked changes.

---

## Self-Review

- Spec coverage: Tasks 1-3 cover chunked append-only storage, sample IDs, idempotency, and batch lookup. Task 4 covers streaming metric and training helpers. Tasks 5-6 cover persisted random splits and background training without full-array loading. Task 7 removes old dataset assumptions. Task 8 documents public usage. Task 9 verifies no full-dataset or quadratic training path remains.
- Red-flag scan: The plan contains concrete test code, implementation snippets, commands, and expected outputs for every task.
- Type consistency: Public IDs are `Sequence[object]`/`Iterable[object]`; internal compact keys are `np.uint64`; dataset batches expose `sample_keys`, `sample_ids`, `inputs`, and `outputs`; training split files persist compact key arrays.
