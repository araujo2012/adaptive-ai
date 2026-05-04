from contextlib import contextmanager
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time

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


def _wait_for_path(path, process, stdout, stderr, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            out = stdout.read() if stdout is not None else ""
            err = stderr.read() if stderr is not None else ""
            pytest.fail(
                "append subprocess exited before signaling pending chunk\n"
                f"stdout:\n{out}\nstderr:\n{err}"
            )
        if time.monotonic() >= deadline:
            pytest.fail("append subprocess did not signal pending chunk in time")
        time.sleep(0.01)


def _guard_dataset_sample_in_bind_count(ai, monkeypatch, *, max_binds):
    original_connect = ai._storage._connect
    bind_counts = []

    class GuardedConnection:
        def __init__(self, connection):
            self._connection = connection

        def execute(self, sql, parameters=(), *args, **kwargs):
            normalized = " ".join(str(sql).split()).upper()
            if "FROM DATASET_SAMPLES" in normalized and " IN (" in normalized:
                bind_count = len(parameters)
                bind_counts.append(bind_count)
                if bind_count > max_binds:
                    raise sqlite3.OperationalError("too many SQL variables")
            return self._connection.execute(sql, parameters, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._connection, name)

    @contextmanager
    def guarded_connect():
        with original_connect() as connection:
            yield GuardedConnection(connection)

    monkeypatch.setattr(storage_module, "MAX_SQL_BIND_PARAMETERS", max_binds, raising=False)
    monkeypatch.setattr(ai._storage, "_connect", guarded_connect)
    return bind_counts


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


def test_get_samples_preserves_duplicate_requested_ids(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0, 0], [1, 1]],
        [[0], [1]],
        sample_ids=["a", "b"],
    )

    samples = ai.get_samples(["b", "a", "b"])

    assert samples.sample_ids == ["b", "a", "b"]
    np.testing.assert_allclose(samples.inputs, [[1, 1], [0, 0], [1, 1]])
    np.testing.assert_allclose(samples.outputs, [[1], [0], [1]])


def test_empty_get_samples_and_empty_key_batches_return_empty_batches(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0, 0, 0]],
        [[0, 0]],
        sample_ids=["seed"],
    )

    samples = ai.get_samples([])

    assert samples.sample_keys.tolist() == []
    assert samples.sample_ids == []
    assert samples.inputs.shape == (0, 3)
    assert samples.outputs.shape == (0, 2)
    assert list(
        ai._storage.iter_key_batches(np.asarray([], dtype=np.uint64), batch_size=2)
    ) == []


def test_large_get_samples_lookup_is_paged_below_sqlite_bind_limit(
    tmp_path, monkeypatch
):
    ai = AdaptiveAI(path=tmp_path)
    sample_ids = [f"id-{index}" for index in range(12)]
    ai.set_input_output(
        [[index, index + 100] for index in range(12)],
        [[index % 2] for index in range(12)],
        sample_ids=sample_ids,
    )
    bind_counts = _guard_dataset_sample_in_bind_count(ai, monkeypatch, max_binds=3)

    samples = ai.get_samples(sample_ids)

    assert samples.sample_ids == sample_ids
    np.testing.assert_allclose(
        samples.inputs, [[index, index + 100] for index in range(12)]
    )
    np.testing.assert_allclose(samples.outputs, [[index % 2] for index in range(12)])
    assert bind_counts
    assert max(bind_counts) <= 3


def test_large_load_samples_by_keys_is_paged_below_sqlite_bind_limit(
    tmp_path, monkeypatch
):
    ai = AdaptiveAI(path=tmp_path)
    sample_ids = [f"id-{index}" for index in range(12)]
    ai.set_input_output(
        [[index, index + 100] for index in range(12)],
        [[index % 2] for index in range(12)],
        sample_ids=sample_ids,
    )
    all_keys = np.concatenate(
        [batch.sample_keys for batch in ai.get_dataset().iter_batches(batch_size=4)]
    )
    bind_counts = _guard_dataset_sample_in_bind_count(ai, monkeypatch, max_binds=3)

    samples = ai._storage.load_samples_by_keys(all_keys)

    assert samples.sample_keys.tolist() == all_keys.tolist()
    assert samples.sample_ids == sample_ids
    np.testing.assert_allclose(
        samples.inputs, [[index, index + 100] for index in range(12)]
    )
    np.testing.assert_allclose(samples.outputs, [[index % 2] for index in range(12)])
    assert bind_counts
    assert max(bind_counts) <= 3


def test_missing_sample_id_fails_clearly(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0]], [[0]], sample_ids=["a"])

    with pytest.raises(ValueError, match="sample_ids were not found"):
        ai.get_samples(["missing"])


def test_iter_key_batches_yields_requested_keys_in_batch_size(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0], [1], [2], [3], [4]],
        [[0], [0], [1], [1], [1]],
        sample_ids=[f"id-{index}" for index in range(5)],
    )
    all_keys = np.concatenate(
        [batch.sample_keys for batch in ai.get_dataset().iter_batches(batch_size=2)]
    )
    selected_keys = all_keys[[4, 1, 3]]

    batches = list(ai._storage.iter_key_batches(selected_keys, batch_size=2))

    assert [batch.sample_keys.tolist() for batch in batches] == [
        selected_keys[:2].tolist(),
        selected_keys[2:].tolist(),
    ]
    assert [batch.sample_ids for batch in batches] == [["id-4", "id-1"], ["id-3"]]
    np.testing.assert_allclose(
        np.vstack([batch.inputs for batch in batches]), [[4], [1], [3]]
    )


def test_list_sample_keys_returns_committed_compact_keys(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0.0], [1.0], [2.0]],
        [[0.0], [1.0], [0.0]],
        sample_ids=["left", "middle", "right"],
    )

    sample_keys = ai._storage.list_sample_keys()

    assert sample_keys.dtype == np.uint64
    assert sample_keys.shape == (3,)


def test_training_split_materializes_random_compact_keys_and_persists_them(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    inputs = [[float(index)] for index in range(10)]
    outputs = [[float(index % 2)] for index in range(10)]
    ai.set_input_output(
        inputs,
        outputs,
        sample_ids=[f"ts-{index}" for index in range(10)],
    )
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

    job = ai.get_training_job(job_id)
    assert job["train_ratio"] == 0.8
    assert job["batch_size"] == 2
    assert job["train_cursor"] == 0
    assert job["validation_cursor"] == 0


@pytest.mark.parametrize("train_ratio", [0.0, 1.0, float("nan"), float("inf")])
def test_create_job_rejects_invalid_train_ratio(tmp_path, train_ratio):
    ai = AdaptiveAI(path=tmp_path)

    with pytest.raises(
        ValueError,
        match="train_ratio must be finite and greater than 0 and less than 1",
    ):
        ai._storage.create_job(
            max_seconds=1.0,
            amount_strategy="fixed",
            fixed_steps=1,
            learning_rate=0.1,
            train_ratio=train_ratio,
        )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, float("nan"), float("inf")])
def test_create_job_rejects_invalid_batch_size(tmp_path, batch_size):
    ai = AdaptiveAI(path=tmp_path)

    with pytest.raises(ValueError, match="batch_size must be positive integer"):
        ai._storage.create_job(
            max_seconds=1.0,
            amount_strategy="fixed",
            fixed_steps=1,
            learning_rate=0.1,
            batch_size=batch_size,
        )


def test_concurrent_training_split_creation_is_atomic(tmp_path, monkeypatch):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[float(index)] for index in range(12)],
        [[float(index % 2)] for index in range(12)],
        sample_ids=[f"split-race-{index}" for index in range(12)],
    )
    job_id = ai._storage.create_job(
        max_seconds=1.0,
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.1,
        train_ratio=0.8,
        batch_size=2,
    )

    original_save = storage_module.np.save
    entered_train_writes = 0
    writers_ready = threading.Event()
    save_lock = threading.Lock()

    def pause_concurrent_train_key_writes(path, *args, **kwargs):
        nonlocal entered_train_writes
        path_text = str(path)
        if "job_splits" in path_text and path_text.endswith("_train_keys.npy"):
            with save_lock:
                entered_train_writes += 1
                if entered_train_writes == 2:
                    writers_ready.set()
            writers_ready.wait(timeout=0.5)
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(storage_module.np, "save", pause_concurrent_train_key_writes)

    start_barrier = threading.Barrier(2)
    outcomes = []
    outcomes_lock = threading.Lock()

    def create_split(seed):
        storage = storage_module.Storage(tmp_path)
        try:
            start_barrier.wait(timeout=5)
            outcome = storage.get_or_create_training_split(
                job_id,
                seed=seed,
                train_ratio=0.8,
            )
        except Exception as exc:
            outcome = exc
        with outcomes_lock:
            outcomes.append(outcome)

    threads = [
        threading.Thread(target=create_split, args=(123,)),
        threading.Thread(target=create_split, args=(456,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert len(outcomes) == 2
    assert not any(
        isinstance(outcome, Exception) and _contains_integrity_error(outcome)
        for outcome in outcomes
    ), outcomes
    assert all(not isinstance(outcome, Exception) for outcome in outcomes), outcomes

    persisted = ai._storage.get_or_create_training_split(
        job_id,
        seed=999,
        train_ratio=0.5,
    )
    with sqlite3.connect(tmp_path / ".adaptive_ai" / "adaptive_ai.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT train_path, validation_path FROM training_splits WHERE job_id = ?",
            (job_id,),
        ).fetchone()

    assert row is not None
    np.testing.assert_array_equal(np.load(row["train_path"]), persisted.train_keys)
    np.testing.assert_array_equal(
        np.load(row["validation_path"]),
        persisted.validation_keys,
    )


def test_training_split_requires_at_least_two_samples(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0.0]], [[0.0]], sample_ids=["only"])
    job_id = ai._storage.create_job(
        max_seconds=1.0,
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.1,
    )

    with pytest.raises(ValueError, match="at least 2 samples"):
        ai._storage.get_or_create_training_split(job_id, seed=1, train_ratio=0.8)


@pytest.mark.parametrize("train_ratio", [0.0, 1.0, float("nan"), float("inf")])
def test_training_split_rejects_invalid_train_ratio(tmp_path, train_ratio):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0.0], [1.0]],
        [[0.0], [1.0]],
        sample_ids=["left", "right"],
    )
    job_id = ai._storage.create_job(
        max_seconds=1.0,
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.1,
    )

    with pytest.raises(
        ValueError,
        match="train_ratio must be finite and greater than 0 and less than 1",
    ):
        ai._storage.get_or_create_training_split(job_id, seed=1, train_ratio=train_ratio)


@pytest.mark.parametrize("train_ratio", [0.0, 1.0, float("nan"), float("inf")])
def test_existing_training_split_rejects_invalid_requested_train_ratio(
    tmp_path, train_ratio
):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0.0], [1.0], [2.0]],
        [[0.0], [1.0], [0.0]],
        sample_ids=["left", "middle", "right"],
    )
    job_id = ai._storage.create_job(
        max_seconds=1.0,
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.1,
    )
    ai._storage.get_or_create_training_split(job_id, seed=1, train_ratio=0.8)

    with pytest.raises(
        ValueError,
        match="train_ratio must be finite and greater than 0 and less than 1",
    ):
        ai._storage.get_or_create_training_split(
            job_id,
            seed=2,
            train_ratio=train_ratio,
        )


def test_training_split_reports_missing_key_file_as_value_error(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[0.0], [1.0], [2.0]],
        [[0.0], [1.0], [0.0]],
        sample_ids=["left", "middle", "right"],
    )
    job_id = ai._storage.create_job(
        max_seconds=1.0,
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.1,
    )
    split = ai._storage.get_or_create_training_split(job_id, seed=1, train_ratio=0.8)
    split.train_path.unlink()

    with pytest.raises(ValueError, match="training split train keys file is missing"):
        ai._storage.get_or_create_training_split(job_id, seed=2, train_ratio=0.8)


def test_training_split_rejects_key_files_that_do_not_match_metadata(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[float(index)] for index in range(5)],
        [[float(index % 2)] for index in range(5)],
        sample_ids=[f"count-{index}" for index in range(5)],
    )
    job_id = ai._storage.create_job(
        max_seconds=1.0,
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.1,
    )
    split = ai._storage.get_or_create_training_split(job_id, seed=1, train_ratio=0.8)
    np.save(split.train_path, split.train_keys[:1])

    with pytest.raises(
        ValueError,
        match="training split train keys count does not match metadata",
    ):
        ai._storage.get_or_create_training_split(job_id, seed=2, train_ratio=0.8)


def test_training_split_rejects_non_1d_key_files(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[float(index)] for index in range(5)],
        [[float(index % 2)] for index in range(5)],
        sample_ids=[f"shape-{index}" for index in range(5)],
    )
    job_id = ai._storage.create_job(
        max_seconds=1.0,
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.1,
    )
    split = ai._storage.get_or_create_training_split(job_id, seed=1, train_ratio=0.8)
    np.save(split.validation_path, split.validation_keys.reshape(1, -1))

    with pytest.raises(ValueError, match="training split validation keys must be 1D"):
        ai._storage.get_or_create_training_split(job_id, seed=2, train_ratio=0.8)


def test_training_split_uses_job_train_ratio_for_new_split(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output(
        [[float(index)] for index in range(10)],
        [[float(index % 2)] for index in range(10)],
        sample_ids=[f"job-ratio-{index}" for index in range(10)],
    )
    job_id = ai._storage.create_job(
        max_seconds=1.0,
        amount_strategy="fixed",
        fixed_steps=1,
        learning_rate=0.1,
        train_ratio=0.7,
        batch_size=2,
    )

    with pytest.raises(ValueError, match="train_ratio must match"):
        ai._storage.get_or_create_training_split(job_id, seed=1, train_ratio=0.8)

    split = ai._storage.get_or_create_training_split(job_id, seed=None, train_ratio=0.7)

    assert split.seed is None
    assert split.train_ratio == 0.7
    assert split.train_keys.shape[0] == 7
    assert split.validation_keys.shape[0] == 3


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


def test_storage_startup_cleanup_does_not_delete_live_append_chunk(tmp_path):
    ai = AdaptiveAI(path=tmp_path)
    ai.set_input_output([[0]], [[0]], sample_ids=["base"])

    ready_path = tmp_path / "append-ready.txt"
    release_path = tmp_path / "append-release.txt"
    result_path = tmp_path / "append-result.txt"
    script = textwrap.dedent(
        """
        from pathlib import Path
        import sys
        import time
        import traceback

        from adaptive_ai import AdaptiveAI
        import adaptive_ai.storage as storage_module

        root = Path(sys.argv[1])
        ready_path = Path(sys.argv[2])
        release_path = Path(sys.argv[3])
        result_path = Path(sys.argv[4])

        ai = AdaptiveAI(path=root)
        original_save = storage_module.np.save
        paused = False

        def pause_first_append_chunk_save(path, *args, **kwargs):
            global paused
            if not paused and str(path).endswith("inputs.npy"):
                paused = True
                ready_path.write_text("ready", encoding="utf-8")
                deadline = time.monotonic() + 10
                while not release_path.exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("append release was not signaled")
                    time.sleep(0.01)
            return original_save(path, *args, **kwargs)

        storage_module.np.save = pause_first_append_chunk_save

        try:
            ai.put_input_output(
                [[1], [2]],
                [[1], [1]],
                sample_ids=["live-1", "live-2"],
            )
        except BaseException:
            result_path.write_text(traceback.format_exc(), encoding="utf-8")
            raise
        else:
            result_path.write_text("OK", encoding="utf-8")
        """
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            str(ready_path),
            str(release_path),
            str(result_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout = ""
    stderr = ""
    try:
        _wait_for_path(ready_path, process, process.stdout, process.stderr)
        AdaptiveAI(path=tmp_path)
        release_path.write_text("release", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=10)
    finally:
        release_path.write_text("release", encoding="utf-8")
        if process.poll() is None:
            process.kill()
            stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (
        result_path.read_text(encoding="utf-8") if result_path.exists() else stderr
    )
    assert stdout == ""
    assert stderr == ""
    assert sorted(_all_sample_ids(ai)) == ["base", "live-1", "live-2"]

    db_path = tmp_path / ".adaptive_ai" / "adaptive_ai.sqlite3"
    with sqlite3.connect(db_path) as connection:
        running_count = connection.execute(
            "SELECT COUNT(*) FROM dataset_ingestions WHERE status = 'running'"
        ).fetchone()[0]
        chunk = connection.execute(
            """
            SELECT id, sample_keys_path, row_count
            FROM dataset_chunks
            WHERE status = 'committed' AND row_count = 2
            """
        ).fetchone()
        assert chunk is not None
        sample_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM dataset_samples
            WHERE chunk_id = ? AND status = 'committed'
            """,
            (chunk[0],),
        ).fetchone()[0]

    assert running_count == 0
    assert sample_count == 2
    sample_keys = np.load(chunk[1])
    assert sample_keys.shape == (2,)
    assert int(chunk[2]) == 2


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
