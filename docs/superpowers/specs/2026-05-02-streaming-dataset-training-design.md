# Streaming Dataset And Training Design

## Context

`AdaptiveAI` currently stores the full dataset in `arrays/dataset.npz`. `put_input_output()` loads the existing arrays, concatenates the new samples in memory, and writes the whole dataset again. `start_training()` then loads the full dataset, builds train/validation copies with masks, and evaluates models against complete arrays.

That design does not scale to millions of samples or thousands of input columns. The project must treat every dataset as potentially huge. There should be no separate "small dataset" path, no full-dataset loading in internal workflows, and no `n^2` materialization.

## Goals

- Append input/output samples incrementally without reading old samples.
- Support caller-provided sample IDs of any serializable type.
- Preserve idempotency when the same sample ID is written again.
- Store progress metadata for ingestion and training resume.
- Train and evaluate from batches without loading the whole dataset.
- Keep memory bounded by current batch size, model size, and compact split IDs.
- Avoid masks, pair matrices, full train/validation copies, or other structures that scale as `n^2`.

## Non-Goals

- Add a separate optimization path for "small" datasets.
- Add a server, remote storage backend, or distributed training layer.
- Introduce a heavy storage dependency unless the current NumPy + SQLite approach proves insufficient.
- Preserve `dataset.npz` as the canonical storage format.

## Storage Architecture

The canonical dataset becomes an append-only collection under `.adaptive_ai/arrays/dataset/`.

Each committed chunk stores only the new samples from one ingestion batch:

```text
.adaptive_ai/
  arrays/
    dataset/
      chunks/
        <chunk_id>/
          inputs.npy
          outputs.npy
          sample_keys.npy
      job_splits/
        <job_id>_train_keys.npy
        <job_id>_validation_keys.npy
```

SQLite remains the metadata source of truth. It should track:

- dataset dimensions: input size and output size;
- chunks: `chunk_id`, row count, file paths, status, creation time;
- samples: internal compact key, original sample ID, normalized sample ID, content fingerprint, `chunk_id`, `row_index`, status;
- ingestion progress: committed rows, skipped idempotent rows, rejected conflicts;
- training split metadata: job ID, seed, train ratio, split file paths, counts;
- training progress: current round, cursor positions, metrics, status.

Chunks are written to a temporary status first and marked committed only after all files and sample metadata are durable. Interrupted pending chunks are ignored or cleaned up during storage initialization.

## Sample IDs

`sample_ids` is optional on `set_input_output()` and `put_input_output()`.

If provided, it must contain exactly one ID per sample. The ID is owned by the caller and may be an integer, timestamp, string, UUID, or another serializable value. The API should expose the original value unchanged where practical.

Internally, storage assigns each sample a compact monotonic key, such as SQLite row ID or `uint64`. Training and split files use these compact keys, not the original IDs, to keep memory use predictable.

For lookup and idempotency, the original sample ID is normalized into a deterministic serialized form. If a `sample_id` already exists:

- when the stored content fingerprint matches the new input/output row, ingestion treats it as already committed and skips it;
- when the fingerprint differs, ingestion fails with a clear conflict error instead of overwriting data.

If `sample_ids` is not provided, storage generates sample IDs and internal keys. Generated IDs must still be stable once committed.

## Public API Shape

The current method names can stay, but their behavior changes to collection-first storage:

```python
ai.set_input_output(inputs, outputs, sample_ids=None)
ai.put_input_output(inputs, outputs, sample_ids=None)
```

`set_input_output()` starts a new collection generation and clears existing models. It writes the provided samples as chunked data without creating a full dataset array.

`put_input_output()` appends new chunks. It validates dimensions from the incoming batch and writes only the new samples.

`get_dataset()` should no longer return full `inputs` and `outputs` arrays. To avoid a misleading API that suggests full loading is normal, it should return a lightweight dataset collection view with metadata and batch iteration. Internal training must not call it for full-array loading.

```python
dataset = ai.get_dataset()
dataset.sample_count
dataset.input_size
dataset.output_size
dataset.iter_batches(batch_size=1024)
ai.get_samples(sample_ids_or_keys)
```

There should be no implicit full-array compatibility path. If a future debugging helper needs to export arrays, it must be named explicitly as an export operation and must not be used by training.

## Batch Loading

Storage provides batch loaders that accept internal compact keys. A batch loader resolves:

```text
compact_key -> sample metadata -> chunk_id + row_index -> chunk file row
```

The loader groups requested keys by chunk to minimize file opens, loads only the needed rows, returns `inputs` and `outputs` arrays for the batch, and releases file handles immediately after use.

The only arrays held in memory during training are:

- model matrices;
- train/validation split key arrays;
- current input/output batch;
- small metric accumulators.

## Train/Validation Split

Random split quality is important. At job start, training materializes only compact sample keys in memory.

The split process is:

1. Load committed compact sample keys from SQLite into a NumPy array.
2. Shuffle keys with the job seed.
3. Split by configured ratio, defaulting to `80%` training and `20%` validation.
4. Persist train and validation key arrays under `job_splits/`.
5. Resume from persisted split files when a paused or interrupted job continues.

This gives an actual random split across the full collection while avoiding sample-array loading. Millions of compact `uint64` keys are acceptable in memory; millions of full samples are not.

## Streaming Training

Training should stop calling `train_matrices()` with the whole dataset. Instead, it should use batch-aware training helpers.

A training round:

1. Select a model from storage.
2. Evaluate baseline validation metrics by streaming validation batches.
3. Train a candidate model by streaming train batches for the configured number of steps.
4. Re-evaluate the candidate by streaming validation batches.
5. Save the candidate only if validation metrics improve.
6. Save logs and progress metadata.

Batch training applies gradients for the current batch, discards the batch arrays, checks pause/cancel/deadline, and moves to the next batch.

Validation metrics are accumulated as:

- `accepted_count`;
- `sample_count`;
- `squared_error_sum`;
- derived `accepted_rate`;
- derived `mse`.

No validation input/output array is assembled.

## Strategy Rules

`fixed` remains a bounded step count.

`sample_square` may remain as a step-count strategy, but it must never materialize sample pairs, pairwise matrices, or any `n^2` data structure. It can compute an integer from counts and cap or stream work as needed.

Any future strategy must satisfy the same invariant: memory grows with batch size and model size, not dataset size beyond compact ID arrays.

## Error Handling And Resume

Ingestion writes pending chunk metadata before files are committed. On startup, storage marks or removes pending chunks that were interrupted.

Training jobs persist enough progress to resume:

- split file paths;
- train/validation counts;
- current round;
- cursor or RNG state needed for deterministic batch selection;
- current job status and metrics.

Pause and cancel continue to use job control events, but batch loops check those events between batches and within long-running step loops.

## Testing Strategy

Tests should prove behavior by observing storage and memory-safe access patterns, not by relying on giant fixtures.

Key tests:

- `put_input_output()` appends chunks without reading existing chunks.
- caller-provided `sample_ids` are stored and exposed.
- duplicate `sample_id` with identical content is idempotent.
- duplicate `sample_id` with different content raises an error.
- train/validation split persists compact keys and respects the configured ratio.
- training calls storage batch loaders instead of `load_dataset()`.
- streaming evaluation matches full-array metrics on small fixtures used only as correctness oracles.
- `sample_square` computes steps without allocating pairwise structures.

## Legacy `dataset.npz`

Existing `dataset.npz` workspaces are legacy storage. Internal training must not load them.

The first implementation should support new chunked workspaces and fail clearly if only legacy `dataset.npz` storage exists. A separate migration tool can be added later, but it must be explicit about the memory implications of compressed `.npz` files and must not become part of normal training startup.
