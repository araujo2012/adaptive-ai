from contextlib import contextmanager
import sqlite3
import threading

import numpy as np
import pytest

from adaptive_ai import AdaptiveAI
import adaptive_ai.storage as storage_module


def _all_sample_ids(ai):
    return [
        sample_id
        for batch in ai.get_dataset().iter_batches(batch_size=10)
        for sample_id in batch.sample_ids
    ]


def _contains_integrity_error(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, sqlite3.IntegrityError):
            return True
        seen.add(id(exc))
        if exc.__cause__ is not None:
            exc = exc.__cause__
        elif not exc.__suppress_context__:
            exc = exc.__context__
        else:
            exc = None
    return False


def _run_concurrent_appends(tmp_path, monkeypatch, jobs):
    start_barrier = threading.Barrier(len(jobs))
    write_barrier = threading.Barrier(len(jobs))
    outcomes = []
    outcomes_lock = threading.Lock()
    original_save = storage_module.np.save

    def synchronized_chunk_input_save(path, *args, **kwargs):
        if str(path).endswith("inputs.npy"):
            write_barrier.wait(timeout=5)
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(storage_module.np, "save", synchronized_chunk_input_save)

    def run_job(job):
        ai = AdaptiveAI(path=tmp_path)
        try:
            start_barrier.wait(timeout=5)
            ai.put_input_output(
                job["inputs"],
                job["outputs"],
                sample_ids=[job["sample_id"]],
            )
        except Exception as exc:
            outcome = exc
        else:
            outcome = None
        with outcomes_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=run_job, args=(job,)) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert len(outcomes) == len(jobs)
    return outcomes


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


def test_sample_ids_are_exposed_and_duplicate_identical_rows_are_idempotent(tmp_path):
    ai = AdaptiveAI(path=tmp_path)

    ai.set_input_output(
        [[0], [1]],
        [[0], [1]],
        sample_ids=["2026-05-02T00:00:00", 123],
    )
    ai.put_input_output([[1]], [[1]], sample_ids=[123])

    dataset = ai.get_dataset()
    assert dataset.sample_count == 2
    batches = list(dataset.iter_batches(batch_size=10))
    assert batches[0].sample_ids == ["2026-05-02T00:00:00", 123]

    chunk_dirs = sorted(
        (tmp_path / ".adaptive_ai" / "arrays" / "dataset" / "chunks").iterdir()
    )
    assert len(chunk_dirs) == 1


def test_duplicate_sample_id_with_different_content_fails(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0]], [[0]], sample_ids=["same-id"])
    chunks_dir = tmp_path / ".adaptive_ai" / "arrays" / "dataset" / "chunks"
    before_count = ai.get_dataset().sample_count
    before_chunks = sorted(chunks_dir.iterdir())

    with pytest.raises(ValueError, match="conflicting sample_id"):
        ai.put_input_output([[1]], [[0]], sample_ids=["same-id"])

    assert ai.get_dataset().sample_count == before_count
    assert sorted(chunks_dir.iterdir()) == before_chunks


def test_concurrent_identical_appends_are_idempotent(tmp_path, monkeypatch):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0]], [[0]], sample_ids=["base"])

    outcomes = _run_concurrent_appends(
        tmp_path,
        monkeypatch,
        [
            {"inputs": [[1]], "outputs": [[1]], "sample_id": "dup"},
            {"inputs": [[1]], "outputs": [[1]], "sample_id": "dup"},
        ],
    )

    assert not any(_contains_integrity_error(outcome) for outcome in outcomes)
    assert all(outcome is None for outcome in outcomes)
    dataset = ai.get_dataset()
    assert dataset.sample_count == 2
    assert _all_sample_ids(ai).count("dup") == 1


def test_concurrent_conflicting_appends_return_clear_conflict(tmp_path, monkeypatch):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0]], [[0]], sample_ids=["base"])

    outcomes = _run_concurrent_appends(
        tmp_path,
        monkeypatch,
        [
            {"inputs": [[1]], "outputs": [[1]], "sample_id": "race"},
            {"inputs": [[2]], "outputs": [[1]], "sample_id": "race"},
        ],
    )

    errors = [outcome for outcome in outcomes if outcome is not None]
    assert not any(_contains_integrity_error(outcome) for outcome in outcomes)
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "conflicting sample_id" in str(errors[0])
    assert ai.get_dataset().sample_count == 2
    assert _all_sample_ids(ai).count("race") == 1


def test_equal_opaque_sample_ids_are_idempotent_across_pickle_order(tmp_path):
    ai = AdaptiveAI(path=tmp_path)

    ai.set_input_output([[0]], [[0]], sample_ids=[{"left": 1, "right": 2}])
    ai.put_input_output([[0]], [[0]], sample_ids=[{"right": 2, "left": 1}])

    dataset = ai.get_dataset()
    assert dataset.sample_count == 1
    batches = list(dataset.iter_batches(batch_size=10))
    assert batches[0].sample_ids == [{"left": 1, "right": 2}]

    chunk_dirs = sorted(
        (tmp_path / ".adaptive_ai" / "arrays" / "dataset" / "chunks").iterdir()
    )
    assert len(chunk_dirs) == 1


def test_sample_id_lookup_uses_indexed_canonical_key(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0]], [[0]], sample_ids=[{"left": 1, "right": 2}])

    db_path = tmp_path / ".adaptive_ai" / "adaptive_ai.sqlite3"
    with sqlite3.connect(db_path) as connection:
        sample_id_key = connection.execute(
            """
            SELECT sample_id_key
            FROM dataset_samples
            WHERE status = 'committed'
            """
        ).fetchone()[0]
        plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT content_fingerprint
            FROM dataset_samples
            WHERE sample_id_key = ? AND status = 'committed'
            """,
            (sample_id_key,),
        ).fetchall()

    details = [str(row[3]).upper() for row in plan]
    assert not any("SCAN DATASET_SAMPLES" in detail for detail in details)
    assert any("INDEX" in detail for detail in details)


def test_duplicate_mapping_sample_ids_use_canonical_batch_key(tmp_path, monkeypatch):
    def fail_value_comparison(*args):
        raise AssertionError("batch duplicate detection should not compare mappings")

    monkeypatch.setattr(storage_module, "_sample_ids_equal", fail_value_comparison, raising=False)
    ai = AdaptiveAI(path=tmp_path)

    with pytest.raises(ValueError, match="duplicate sample_ids"):
        ai.set_input_output(
            [[0], [1]],
            [[0], [1]],
            sample_ids=[{"left": 1, "right": 2}, {"right": 2, "left": 1}],
        )


def test_canonical_key_migration_reports_duplicate_ids_clearly(tmp_path):
    base_path = tmp_path / ".adaptive_ai"
    chunk_dir = base_path / "arrays" / "dataset" / "chunks" / "old-chunk"
    chunk_dir.mkdir(parents=True)
    inputs = np.array([[0.0], [0.0]], dtype=np.float64)
    outputs = np.array([[0.0], [0.0]], dtype=np.float64)
    input_path = chunk_dir / "inputs.npy"
    output_path = chunk_dir / "outputs.npy"
    sample_keys_path = chunk_dir / "sample_keys.npy"
    np.save(input_path, inputs)
    np.save(output_path, outputs)
    np.save(sample_keys_path, np.array([1, 2], dtype=np.uint64))

    db_path = base_path / "adaptive_ai.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE dataset_chunks (
                id TEXT PRIMARY KEY,
                input_path TEXT NOT NULL,
                output_path TEXT NOT NULL,
                sample_keys_path TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                committed_at TEXT
            );

            CREATE TABLE dataset_samples (
                key INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id_blob BLOB NOT NULL UNIQUE,
                content_fingerprint TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES dataset_chunks(id)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO dataset_chunks(
                id, input_path, output_path, sample_keys_path,
                row_count, status, created_at, committed_at
            )
            VALUES('old-chunk', ?, ?, ?, 2, 'committed', ?, ?)
            """,
            (
                str(input_path),
                str(output_path),
                str(sample_keys_path),
                "2026-05-03T00:00:00+00:00",
                "2026-05-03T00:00:00+00:00",
            ),
        )
        for row_index, sample_id in enumerate(
            [{"left": 1, "right": 2}, {"right": 2, "left": 1}]
        ):
            connection.execute(
                """
                INSERT INTO dataset_samples(
                    sample_id_blob, content_fingerprint, chunk_id,
                    row_index, status, created_at
                )
                VALUES(?, ?, 'old-chunk', ?, 'committed', ?)
                """,
                (
                    storage_module._sample_id_to_blob(sample_id),
                    storage_module._fingerprint_row(inputs[row_index], outputs[row_index]),
                    row_index,
                    "2026-05-03T00:00:00+00:00",
                ),
            )

    with pytest.raises(ValueError, match="(?i)duplicate.*canonical.*sample_id"):
        AdaptiveAI(path=tmp_path)

    with sqlite3.connect(db_path) as connection:
        index_row = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index' AND name = 'dataset_samples_sample_id_key_idx'
            """
        ).fetchone()
    assert index_row is None


def test_set_sample_ids_are_idempotent_with_different_order(tmp_path):
    ai = AdaptiveAI(path=tmp_path)

    first_id = {0, 2**61 - 1}
    second_id = {2**61 - 1, 0}
    ai.set_input_output([[0]], [[0]], sample_ids=[first_id])
    ai.put_input_output([[0]], [[0]], sample_ids=[second_id])

    dataset = ai.get_dataset()
    assert dataset.sample_count == 1
    chunk_dirs = sorted(
        (tmp_path / ".adaptive_ai" / "arrays" / "dataset" / "chunks").iterdir()
    )
    assert len(chunk_dirs) == 1


def test_dataset_sample_count_uses_committed_chunk_rows(tmp_path, monkeypatch):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0], [1]], [[0], [1]], sample_ids=["left", "right"])
    ai.put_input_output([[2]], [[0]], sample_ids=["extra"])

    original_connect = ai._storage._connect

    class GuardedConnection:
        def __init__(self, connection):
            self._connection = connection

        def execute(self, sql, *args, **kwargs):
            normalized = " ".join(str(sql).split()).upper()
            if "COUNT(*) AS COUNT FROM DATASET_SAMPLES" in normalized:
                raise AssertionError("sample count must use committed chunk row counts")
            return self._connection.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._connection, name)

    @contextmanager
    def guarded_connect():
        with original_connect() as connection:
            yield GuardedConnection(connection)

    monkeypatch.setattr(ai._storage, "_connect", guarded_connect)

    assert ai.get_dataset().sample_count == 3


def test_dataset_iteration_does_not_materialize_all_sample_keys(tmp_path, monkeypatch):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0], [1]], [[0], [1]], sample_ids=["left", "right"])
    ai.put_input_output([[2], [3]], [[0], [1]], sample_ids=["third", "fourth"])
    ai.put_input_output([[4]], [[0]], sample_ids=["fifth"])

    original_connect = ai._storage._connect
    original_load_samples_by_keys = ai._storage.load_samples_by_keys
    call_sizes = []

    class GuardedConnection:
        def __init__(self, connection):
            self._connection = connection

        def execute(self, sql, *args, **kwargs):
            normalized = " ".join(str(sql).split()).upper()
            if (
                normalized.startswith("SELECT KEY FROM DATASET_SAMPLES")
                and " LIMIT " not in normalized
            ):
                raise AssertionError("iteration must not scan all sample keys at once")
            return self._connection.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._connection, name)

    @contextmanager
    def guarded_connect():
        with original_connect() as connection:
            yield GuardedConnection(connection)

    def record_load_samples_by_keys(sample_keys):
        call_sizes.append(np.asarray(sample_keys).shape[0])
        return original_load_samples_by_keys(sample_keys)

    monkeypatch.setattr(ai._storage, "_connect", guarded_connect)
    monkeypatch.setattr(ai._storage, "load_samples_by_keys", record_load_samples_by_keys)

    batches = list(ai.get_dataset().iter_batches(batch_size=2))

    assert [batch.inputs.shape[0] for batch in batches] == [2, 2, 1]
    assert call_sizes
    assert max(call_sizes) <= 2
    assert len(call_sizes) >= 3


def test_dataset_view_does_not_support_full_array_indexing(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0, 0], [1, 1]], [[0], [1]])

    dataset = ai.get_dataset()

    with pytest.raises(TypeError):
        dataset["inputs"]


def test_legacy_dataset_npz_without_chunks_requires_explicit_migration(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai._storage.set_dimensions(2, 1)
    np.savez_compressed(
        tmp_path / ".adaptive_ai" / "arrays" / "dataset.npz",
        inputs=np.array([[0, 0], [1, 1]], dtype=np.float64),
        outputs=np.array([[0], [1]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="legacy.*chunked.*migration"):
        ai.get_dataset()


def test_failed_replacement_with_duplicate_sample_ids_preserves_existing_collection(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0, 0], [1, 1]], [[0], [1]], sample_ids=["left", "right"])

    with pytest.raises(ValueError, match="duplicate sample_ids"):
        ai.set_input_output([[1, 0], [0, 1]], [[1], [0]], sample_ids=["dup", "dup"])

    dataset = ai.get_dataset()
    assert dataset.sample_count == 2
    batches = list(dataset.iter_batches(batch_size=10))
    np.testing.assert_allclose(batches[0].inputs, [[0, 0], [1, 1]])
    np.testing.assert_allclose(batches[0].outputs, [[0], [1]])


def test_failed_valid_replacement_write_preserves_existing_collection(tmp_path, monkeypatch):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0, 0], [1, 1]], [[0], [1]], sample_ids=["left", "right"])

    original_save = storage_module.np.save

    def fail_first_replacement_chunk_write(path, *args, **kwargs):
        if str(path).endswith("inputs.npy"):
            raise OSError("simulated replacement failure")
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(storage_module.np, "save", fail_first_replacement_chunk_write)

    with pytest.raises(OSError, match="simulated replacement failure"):
        ai.set_input_output([[9, 9], [8, 8]], [[1], [0]], sample_ids=["new-left", "new-right"])

    dataset = ai.get_dataset()
    assert dataset.sample_count == 2
    batches = list(dataset.iter_batches(batch_size=10))
    np.testing.assert_allclose(batches[0].inputs, [[0, 0], [1, 1]])
    np.testing.assert_allclose(batches[0].outputs, [[0], [1]])


def test_failed_append_cleans_pending_chunk_and_finishes_ingestion(tmp_path, monkeypatch):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0]], [[0]], sample_ids=["base"])

    def fail_after_chunk_dir_created(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(storage_module.np, "save", fail_after_chunk_dir_created)

    with pytest.raises(OSError, match="simulated disk failure"):
        ai.put_input_output([[1]], [[1]], sample_ids=["new"])

    chunk_dirs = list((tmp_path / ".adaptive_ai" / "arrays" / "dataset" / "chunks").iterdir())
    assert len(chunk_dirs) == 1
    assert (chunk_dirs[0] / "inputs.npy").exists()
    assert (chunk_dirs[0] / "outputs.npy").exists()
    assert (chunk_dirs[0] / "sample_keys.npy").exists()

    with sqlite3.connect(tmp_path / ".adaptive_ai" / "adaptive_ai.sqlite3") as connection:
        running_count = connection.execute(
            "SELECT COUNT(*) FROM dataset_ingestions WHERE status = 'running'"
        ).fetchone()[0]
    assert running_count == 0


def test_failed_append_marks_ingestion_failed_when_cleanup_fails(tmp_path, monkeypatch):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0]], [[0]], sample_ids=["base"])

    def fail_append_write(*args, **kwargs):
        raise OSError("simulated append failure")

    def fail_cleanup(*args, **kwargs):
        raise PermissionError("locked chunk")

    monkeypatch.setattr(storage_module.np, "save", fail_append_write)
    monkeypatch.setattr(storage_module.shutil, "rmtree", fail_cleanup)

    with pytest.raises(OSError, match="simulated append failure"):
        ai.put_input_output([[1]], [[1]], sample_ids=["new"])

    with sqlite3.connect(tmp_path / ".adaptive_ai" / "adaptive_ai.sqlite3") as connection:
        running_count = connection.execute(
            "SELECT COUNT(*) FROM dataset_ingestions WHERE status = 'running'"
        ).fetchone()[0]
    assert running_count == 0


def test_startup_cleanup_marks_running_ingestions_and_removes_orphan_chunk_dirs(tmp_path):
    AdaptiveAI(path=tmp_path)

    db_path = tmp_path / ".adaptive_ai" / "adaptive_ai.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO dataset_ingestions(id, status, started_at)
            VALUES('stale-ingestion', 'running', '2026-05-03T00:00:00+00:00')
            """
        )

    orphan_dir = tmp_path / ".adaptive_ai" / "arrays" / "dataset" / "chunks" / "orphan"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "dummy.npy").write_bytes(b"stale")

    AdaptiveAI(path=tmp_path)

    with sqlite3.connect(db_path) as connection:
        running_count = connection.execute(
            "SELECT COUNT(*) FROM dataset_ingestions WHERE status = 'running'"
        ).fetchone()[0]
    assert running_count == 0
    assert not orphan_dir.exists()


def test_storage_connection_context_closes_connection(tmp_path):
    ai = AdaptiveAI(path=tmp_path)

    with ai._storage._connect() as connection:
        connection.execute("SELECT 1").fetchone()

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1").fetchone()
