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
