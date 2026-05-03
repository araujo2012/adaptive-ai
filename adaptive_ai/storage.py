from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import hashlib
import json
import pickle
import shutil
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DatasetBatch, SampleBatch
from .math import architecture_from_matrices


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Storage:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.base_path = root_path / ".adaptive_ai"
        self.arrays_path = self.base_path / "arrays"
        self.dataset_path = self.arrays_path / "dataset"
        self.chunks_path = self.dataset_path / "chunks"
        self.job_splits_path = self.dataset_path / "job_splits"
        self.models_path = self.base_path / "models"
        self.db_path = self.base_path / "adaptive_ai.sqlite3"
        self._lock = threading.RLock()

        self.root_path.mkdir(parents=True, exist_ok=True)
        self.arrays_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.chunks_path.mkdir(parents=True, exist_ok=True)
        self.job_splits_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.mark_interrupted_jobs()
        self.cleanup_pending_dataset_chunks()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    generation INTEGER NOT NULL,
                    architecture TEXT NOT NULL,
                    matrix_path TEXT NOT NULL,
                    accepted_rate REAL,
                    accepted_count INTEGER,
                    mse REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    max_seconds REAL NOT NULL,
                    amount_strategy TEXT NOT NULL,
                    fixed_steps INTEGER,
                    learning_rate REAL NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    rounds_completed INTEGER NOT NULL DEFAULT 0,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    message TEXT NOT NULL,
                    model_id TEXT,
                    validation_count INTEGER,
                    steps INTEGER,
                    accepted_rate REAL,
                    mse REAL,
                    FOREIGN KEY(job_id) REFERENCES jobs(id)
                );

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
                """
            )

    def mark_interrupted_jobs(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = 'canceled', finished_at = ?, error = 'process interrupted'
                WHERE status = 'running'
                """,
                (utc_now(),),
            )

    def cleanup_pending_dataset_chunks(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE dataset_ingestions
                SET status = 'failed', finished_at = ?, error = 'process interrupted'
                WHERE status = 'running'
                """,
                (utc_now(),),
            )
            rows = connection.execute(
                """
                SELECT id, input_path, output_path, sample_keys_path
                FROM dataset_chunks
                WHERE status != 'committed'
                """
            ).fetchall()
            committed_rows = connection.execute(
                """
                SELECT id
                FROM dataset_chunks
                WHERE status = 'committed'
                """
            ).fetchall()
            noncommitted_chunk_ids = [str(row["id"]) for row in rows]
            if noncommitted_chunk_ids:
                bind_marks = ",".join("?" for _ in noncommitted_chunk_ids)
                connection.execute(
                    f"DELETE FROM dataset_samples WHERE chunk_id IN ({bind_marks})",
                    noncommitted_chunk_ids,
                )
            connection.execute("DELETE FROM dataset_samples WHERE status != 'committed'")
            connection.execute("DELETE FROM dataset_chunks WHERE status != 'committed'")

        committed_chunk_ids = {str(row["id"]) for row in committed_rows}
        for row in rows:
            chunk_dir = Path(str(row["input_path"])).parent
            self._remove_tree_best_effort(chunk_dir)
        if self.chunks_path.exists():
            for child in self.chunks_path.iterdir():
                if child.is_dir() and child.name not in committed_chunk_ids:
                    self._remove_tree_best_effort(child)

    def get_dimensions(self) -> tuple[int, int] | None:
        input_size = self.get_setting("input_size")
        output_size = self.get_setting("output_size")
        if input_size is None or output_size is None:
            return None
        return int(input_size), int(output_size)

    def set_dimensions(self, input_size: int, output_size: int) -> None:
        self.set_setting("input_size", str(input_size))
        self.set_setting("output_size", str(output_size))

    def get_setting(self, key: str) -> str | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO settings(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def dataset_file(self) -> Path:
        return self.arrays_path / "dataset.npz"

    def has_legacy_dataset_without_chunks(self) -> bool:
        return self.dataset_file().exists() and self.get_sample_count() == 0

    def raise_if_legacy_dataset_requires_migration(self) -> None:
        if self.has_legacy_dataset_without_chunks():
            raise ValueError(
                "legacy dataset.npz storage requires chunked migration before streaming access"
            )

    def save_dataset(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        self.replace_dataset(inputs, outputs)

    def load_dataset(self) -> dict[str, np.ndarray]:
        if self.get_sample_count() > 0:
            batches = list(self.iter_dataset_batches(batch_size=1024))
            return {
                "inputs": np.vstack([batch.inputs for batch in batches]),
                "outputs": np.vstack([batch.outputs for batch in batches]),
            }

        path = self.dataset_file()
        if path.exists():
            self.raise_if_legacy_dataset_requires_migration()
        raise ValueError("dataset has not been set")

    def clear_dataset(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM training_splits")
            connection.execute("DELETE FROM dataset_samples")
            connection.execute("DELETE FROM dataset_chunks")
            connection.execute("DELETE FROM dataset_ingestions")

        legacy_file = self.dataset_file()
        if legacy_file.exists():
            legacy_file.unlink()
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
        rows = _prepare_dataset_rows(inputs, outputs, sample_ids=sample_ids)
        return self._replace_prepared_dataset(rows)

    def append_dataset(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        *,
        sample_ids: Iterable[object] | None = None,
    ) -> dict[str, int]:
        self.raise_if_legacy_dataset_requires_migration()
        rows = _prepare_dataset_rows(inputs, outputs, sample_ids=sample_ids)
        return self._append_prepared_dataset(rows)

    def _append_prepared_dataset(
        self,
        rows: list[tuple[bytes, str, np.ndarray, np.ndarray]],
    ) -> dict[str, int]:
        ingestion_id = str(uuid.uuid4())
        new_rows: list[tuple[bytes, str, np.ndarray, np.ndarray]] = []
        skipped_rows = 0
        conflict_rows = 0
        conflict_error: ValueError | None = None

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_ingestions(id, status, started_at)
                VALUES(?, 'running', ?)
                """,
                (ingestion_id, utc_now()),
            )
            for sample_blob, fingerprint, input_row, output_row in rows:
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
                    conflict_error = ValueError(
                        "conflicting sample_id with different input/output content"
                    )
                    break
                new_rows.append((sample_blob, fingerprint, input_row, output_row))

        if conflict_error is not None:
            raise conflict_error

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
            return {
                "committed_rows": 0,
                "skipped_rows": skipped_rows,
                "conflict_rows": conflict_rows,
            }

        chunk_id = str(uuid.uuid4())
        chunk_dir = self.chunks_path / chunk_id
        try:
            chunk_dir.mkdir(parents=True, exist_ok=False)
            input_path = chunk_dir / "inputs.npy"
            output_path = chunk_dir / "outputs.npy"
            sample_keys_path = chunk_dir / "sample_keys.npy"
            chunk_inputs = np.asarray([row[2] for row in new_rows], dtype=np.float64)
            chunk_outputs = np.asarray([row[3] for row in new_rows], dtype=np.float64)
            np.save(input_path, chunk_inputs)
            np.save(output_path, chunk_outputs)

            sample_keys: list[int] = []
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO dataset_chunks(
                        id, input_path, output_path, sample_keys_path,
                        row_count, status, created_at
                    )
                    VALUES(?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        chunk_id,
                        str(input_path),
                        str(output_path),
                        str(sample_keys_path),
                        len(new_rows),
                        utc_now(),
                    ),
                )
                for row_index, (sample_blob, fingerprint, _, _) in enumerate(new_rows):
                    cursor = connection.execute(
                        """
                        INSERT INTO dataset_samples(
                            sample_id_blob, content_fingerprint, chunk_id,
                            row_index, status, created_at
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
                    SET status = 'committed',
                        committed_rows = ?,
                        skipped_rows = ?,
                        conflict_rows = ?,
                        finished_at = ?
                    WHERE id = ?
                    """,
                    (len(new_rows), skipped_rows, conflict_rows, utc_now(), ingestion_id),
                )
        except Exception as exc:
            self._mark_dataset_ingestion_failed(ingestion_id, str(exc))
            self._remove_tree_best_effort(chunk_dir)
            raise

        return {
            "committed_rows": len(new_rows),
            "skipped_rows": skipped_rows,
            "conflict_rows": conflict_rows,
        }

    def _replace_prepared_dataset(
        self,
        rows: list[tuple[bytes, str, np.ndarray, np.ndarray]],
    ) -> dict[str, int]:
        ingestion_id = str(uuid.uuid4())
        old_chunk_dirs: list[Path] = []
        old_split_paths: list[Path] = []

        if not rows:
            with self._lock, self._connect() as connection:
                old_chunk_dirs = self._dataset_chunk_dirs(connection)
                old_split_paths = self._training_split_paths(connection)
                connection.execute("DELETE FROM training_splits")
                connection.execute("DELETE FROM dataset_samples")
                connection.execute("DELETE FROM dataset_chunks")
                connection.execute("DELETE FROM dataset_ingestions")
                connection.execute(
                    """
                    INSERT INTO dataset_ingestions(
                        id, status, committed_rows, skipped_rows, conflict_rows,
                        started_at, finished_at
                    )
                    VALUES(?, 'committed', 0, 0, 0, ?, ?)
                    """,
                    (ingestion_id, utc_now(), utc_now()),
                )
            self._cleanup_replaced_dataset_files(old_chunk_dirs, old_split_paths)
            return {"committed_rows": 0, "skipped_rows": 0, "conflict_rows": 0}

        chunk_id = str(uuid.uuid4())
        chunk_dir = self.chunks_path / chunk_id
        input_path = chunk_dir / "inputs.npy"
        output_path = chunk_dir / "outputs.npy"
        sample_keys_path = chunk_dir / "sample_keys.npy"

        try:
            chunk_dir.mkdir(parents=True, exist_ok=False)
            chunk_inputs = np.asarray([row[2] for row in rows], dtype=np.float64)
            chunk_outputs = np.asarray([row[3] for row in rows], dtype=np.float64)
            np.save(input_path, chunk_inputs)
            np.save(output_path, chunk_outputs)

            sample_keys: list[int] = []
            with self._lock, self._connect() as connection:
                old_chunk_dirs = self._dataset_chunk_dirs(connection)
                old_split_paths = self._training_split_paths(connection)
                connection.execute("DELETE FROM training_splits")
                connection.execute("DELETE FROM dataset_samples")
                connection.execute("DELETE FROM dataset_chunks")
                connection.execute("DELETE FROM dataset_ingestions")
                connection.execute(
                    """
                    INSERT INTO dataset_ingestions(id, status, started_at)
                    VALUES(?, 'running', ?)
                    """,
                    (ingestion_id, utc_now()),
                )
                connection.execute(
                    """
                    INSERT INTO dataset_chunks(
                        id, input_path, output_path, sample_keys_path,
                        row_count, status, created_at
                    )
                    VALUES(?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        chunk_id,
                        str(input_path),
                        str(output_path),
                        str(sample_keys_path),
                        len(rows),
                        utc_now(),
                    ),
                )
                for row_index, (sample_blob, fingerprint, _, _) in enumerate(rows):
                    cursor = connection.execute(
                        """
                        INSERT INTO dataset_samples(
                            sample_id_blob, content_fingerprint, chunk_id,
                            row_index, status, created_at
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
                    SET status = 'committed',
                        committed_rows = ?,
                        skipped_rows = 0,
                        conflict_rows = 0,
                        finished_at = ?
                    WHERE id = ?
                    """,
                    (len(rows), utc_now(), ingestion_id),
                )
        except Exception:
            self._remove_tree_best_effort(chunk_dir)
            raise

        self._cleanup_replaced_dataset_files(
            old_chunk_dirs,
            old_split_paths,
            preserve_chunk_dir=chunk_dir,
        )
        return {
            "committed_rows": len(rows),
            "skipped_rows": 0,
            "conflict_rows": 0,
        }

    def _dataset_chunk_dirs(self, connection: sqlite3.Connection) -> list[Path]:
        rows = connection.execute("SELECT input_path FROM dataset_chunks").fetchall()
        return [Path(str(row["input_path"])).parent for row in rows]

    def _training_split_paths(self, connection: sqlite3.Connection) -> list[Path]:
        rows = connection.execute(
            "SELECT train_path, validation_path FROM training_splits"
        ).fetchall()
        paths: list[Path] = []
        for row in rows:
            paths.append(Path(str(row["train_path"])))
            paths.append(Path(str(row["validation_path"])))
        return paths

    def _cleanup_replaced_dataset_files(
        self,
        old_chunk_dirs: list[Path],
        old_split_paths: list[Path],
        *,
        preserve_chunk_dir: Path | None = None,
    ) -> None:
        legacy_file = self.dataset_file()
        if legacy_file.exists():
            self._unlink_best_effort(legacy_file)
        for path in old_split_paths:
            self._unlink_best_effort(path)
        for chunk_dir in old_chunk_dirs:
            if preserve_chunk_dir is not None and chunk_dir == preserve_chunk_dir:
                continue
            self._remove_tree_best_effort(chunk_dir)

    def _mark_dataset_ingestion_failed(self, ingestion_id: str, error: str) -> None:
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    UPDATE dataset_ingestions
                    SET status = 'failed', finished_at = ?, error = ?
                    WHERE id = ? AND status = 'running'
                    """,
                    (utc_now(), error, ingestion_id),
                )
        except Exception:
            pass

    def _remove_tree_best_effort(self, path: Path) -> None:
        try:
            if path.exists():
                shutil.rmtree(path)
        except Exception:
            pass

    def _unlink_best_effort(self, path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def get_sample_count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM dataset_samples WHERE status = 'committed'"
            ).fetchone()
        return int(row["count"])

    def iter_dataset_batches(self, *, batch_size: int) -> Iterator[DatasetBatch]:
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
            batch = self.load_samples_by_keys(keys[start : start + batch_size])
            yield DatasetBatch(
                sample_keys=batch.sample_keys,
                sample_ids=batch.sample_ids,
                inputs=batch.inputs,
                outputs=batch.outputs,
            )

    def load_samples_by_keys(self, sample_keys: np.ndarray) -> SampleBatch:
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

    def clear_models(self) -> None:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT matrix_path FROM models").fetchall()
            connection.execute("DELETE FROM models")
        for row in rows:
            path = Path(str(row["matrix_path"]))
            if path.exists():
                path.unlink()

    def save_model(
        self,
        matrices: list[np.ndarray],
        *,
        model_id: str | None = None,
        parent_id: str | None = None,
        generation: int = 0,
        metrics: dict[str, float | int] | None = None,
    ) -> str:
        model_id = model_id or str(uuid.uuid4())
        matrix_path = self.models_path / f"{model_id}.npz"
        arrays = {f"matrix_{index}": matrix.astype(np.float64) for index, matrix in enumerate(matrices)}
        np.savez_compressed(matrix_path, **arrays)
        now = utc_now()
        metrics = metrics or {}
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at, parent_id, generation FROM models WHERE id = ?",
                (model_id,),
            ).fetchone()
            created_at = now if existing is None else str(existing["created_at"])
            parent_id = parent_id if parent_id is not None else (
                None if existing is None else existing["parent_id"]
            )
            generation = generation if existing is None else int(existing["generation"])
            connection.execute(
                """
                INSERT INTO models(
                    id, parent_id, generation, architecture, matrix_path,
                    accepted_rate, accepted_count, mse, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    parent_id = excluded.parent_id,
                    generation = excluded.generation,
                    architecture = excluded.architecture,
                    matrix_path = excluded.matrix_path,
                    accepted_rate = excluded.accepted_rate,
                    accepted_count = excluded.accepted_count,
                    mse = excluded.mse,
                    updated_at = excluded.updated_at
                """,
                (
                    model_id,
                    parent_id,
                    generation,
                    json.dumps(architecture_from_matrices(matrices)),
                    str(matrix_path),
                    _optional_float(metrics.get("accepted_rate")),
                    _optional_int(metrics.get("accepted_count")),
                    _optional_float(metrics.get("mse")),
                    created_at,
                    now,
                ),
            )
        return model_id

    def load_model(self, model_id: str) -> list[np.ndarray]:
        row = self.get_model_meta(model_id)
        path = Path(str(row["matrix_path"]))
        if not path.exists():
            raise ValueError(f"model matrix file is missing for {model_id}")
        with np.load(path) as data:
            keys = sorted(data.files, key=lambda key: int(key.split("_")[1]))
            return [data[key].astype(np.float64, copy=False) for key in keys]

    def get_model_meta(self, model_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if row is None:
            raise ValueError(f"model {model_id} was not found")
        return _model_row_to_dict(row)

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM models
                ORDER BY COALESCE(accepted_rate, -1.0) DESC,
                         COALESCE(mse, 1.0e308) ASC,
                         updated_at DESC
                """
            ).fetchall()
        return [_model_row_to_dict(row) for row in rows]

    def delete_model(self, model_id: str) -> None:
        try:
            row = self.get_model_meta(model_id)
        except ValueError:
            return
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM models WHERE id = ?", (model_id,))
        path = Path(str(row["matrix_path"]))
        if path.exists():
            path.unlink()

    def prune_models(self, max_count: int) -> None:
        models = self.list_models()
        for model in models[max_count:]:
            self.delete_model(str(model["model_id"]))

    def create_job(
        self,
        *,
        max_seconds: float,
        amount_strategy: str,
        fixed_steps: int | None,
        learning_rate: float,
    ) -> str:
        job_id = str(uuid.uuid4())
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(
                    id, status, max_seconds, amount_strategy, fixed_steps,
                    learning_rate, started_at, rounds_completed
                )
                VALUES(?, 'running', ?, ?, ?, ?, ?, 0)
                """,
                (job_id, max_seconds, amount_strategy, fixed_steps, learning_rate, utc_now()),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        rounds_completed: int | None = None,
        error: str | None = None,
        finish: bool = False,
    ) -> None:
        fields: list[str] = []
        values: list[object] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if rounds_completed is not None:
            fields.append("rounds_completed = ?")
            values.append(rounds_completed)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if finish:
            fields.append("finished_at = ?")
            values.append(utc_now())
        if not fields:
            return
        values.append(job_id)
        with self._lock, self._connect() as connection:
            connection.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"training job {job_id} was not found")
        return _job_row_to_dict(row)

    def has_running_job(self) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM jobs WHERE status = 'running' LIMIT 1"
            ).fetchone()
        return row is not None

    def add_log(
        self,
        job_id: str,
        *,
        message: str,
        model_id: str | None = None,
        validation_count: int | None = None,
        steps: int | None = None,
        accepted_rate: float | None = None,
        mse: float | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO job_logs(
                    job_id, created_at, message, model_id, validation_count,
                    steps, accepted_rate, mse
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    utc_now(),
                    message,
                    model_id,
                    validation_count,
                    steps,
                    accepted_rate,
                    mse,
                ),
            )

    def get_logs(self, job_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM job_logs
                WHERE job_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (job_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _prepare_dataset_rows(
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    sample_ids: Iterable[object] | None,
) -> list[tuple[bytes, str, np.ndarray, np.ndarray]]:
    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError("inputs and outputs must contain the same number of samples")

    ids = (
        list(sample_ids)
        if sample_ids is not None
        else [_generated_sample_id() for _ in range(inputs.shape[0])]
    )
    if len(ids) != inputs.shape[0]:
        raise ValueError("sample_ids must contain one id per sample")

    rows: list[tuple[bytes, str, np.ndarray, np.ndarray]] = []
    seen_blobs: set[bytes] = set()
    for sample_id, input_row, output_row in zip(ids, inputs, outputs, strict=True):
        sample_blob = _sample_id_to_blob(sample_id)
        if sample_blob in seen_blobs:
            raise ValueError("duplicate sample_ids are not allowed in one dataset write")
        seen_blobs.add(sample_blob)
        rows.append((sample_blob, _fingerprint_row(input_row, output_row), input_row, output_row))
    return rows


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
    input_bytes = np.ascontiguousarray(input_row, dtype=np.float64).tobytes()
    output_bytes = np.ascontiguousarray(output_row, dtype=np.float64).tobytes()
    digest = hashlib.sha256()
    digest.update(len(input_bytes).to_bytes(8, "big"))
    digest.update(input_bytes)
    digest.update(len(output_bytes).to_bytes(8, "big"))
    digest.update(output_bytes)
    return digest.hexdigest()


def _model_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "model_id": row["id"],
        "parent_id": row["parent_id"],
        "generation": int(row["generation"]),
        "architecture": json.loads(str(row["architecture"])),
        "matrix_path": row["matrix_path"],
        "accepted_rate": row["accepted_rate"],
        "accepted_count": row["accepted_count"],
        "mse": row["mse"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _job_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "job_id": row["id"],
        "status": row["status"],
        "max_seconds": row["max_seconds"],
        "amount_strategy": row["amount_strategy"],
        "fixed_steps": row["fixed_steps"],
        "learning_rate": row["learning_rate"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "rounds_completed": int(row["rounds_completed"]),
        "error": row["error"],
    }
