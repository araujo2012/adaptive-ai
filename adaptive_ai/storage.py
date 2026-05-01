from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .math import architecture_from_matrices


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Storage:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.base_path = root_path / ".adaptive_ai"
        self.arrays_path = self.base_path / "arrays"
        self.models_path = self.base_path / "models"
        self.db_path = self.base_path / "adaptive_ai.sqlite3"
        self._lock = threading.RLock()

        self.root_path.mkdir(parents=True, exist_ok=True)
        self.arrays_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.mark_interrupted_jobs()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

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

    def save_dataset(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        np.savez_compressed(self.dataset_file(), inputs=inputs, outputs=outputs)

    def load_dataset(self) -> dict[str, np.ndarray]:
        path = self.dataset_file()
        if not path.exists():
            raise ValueError("dataset has not been set")
        with np.load(path) as data:
            return {
                "inputs": data["inputs"].astype(np.float64, copy=False),
                "outputs": data["outputs"].astype(np.float64, copy=False),
            }

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
