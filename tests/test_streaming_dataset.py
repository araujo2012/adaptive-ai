import sqlite3

import numpy as np
import pytest

from adaptive_ai import AdaptiveAI
import adaptive_ai.storage as storage_module


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
