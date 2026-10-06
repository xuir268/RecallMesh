# Durable embeddings and mmap checkpoint recovery

`ManagedMemory` already persisted records, association events and logical ticks in SQLite. Schema version 2 additionally persists each embedding vector as a float32 blob. Existing version-1 databases migrate vectors with the existing deterministic HashEmbedder. That encoder remains a placeholder, not a semantic-quality claim. Restart reads persisted vectors without re-encoding records. The legacy `Memory(path=...)` API remains a prototype; use ManagedMemory or the CLI for complete recovery.

```sh
recallmesh checkpoint --config memory.json
recallmesh stats --config memory.json
```

Python: `MemoryClient.checkpoint()` or `ManagedMemory.checkpoint()`.

Checkpoint generation writes a pruned read snapshot, full packed delta CSR (including dead edges), and an embedding NPY array into a new immutable directory. It fsyncs files and directory, then atomically replaces and fsyncs a manifest. The manifest includes revision, capacity, learning parameters, stable-ID mapping, tick, and SHA-256 hashes. CSR format version 1 is 64-bit little-endian with checked boundaries, sorted rows, reverse-edge consistency, packed values and a corruption checksum. The currently implemented mapper requires POSIX (macOS/Linux); Windows uses durable log replay and does not have a mapped restore implementation.

On matching revision, immutable CSR and embeddings are mapped read-only; packed delta values and their ticks are restored into a fresh mutable slot table. On stale, truncated, corrupt or incompatible checkpoints, SQLite event replay is the recovery path. Malformed persisted embeddings are rejected rather than silently regenerated. A subsequent write copies mapped embeddings before mutation and normal snapshot publication resumes. Persisted dead-edge delta values allow an edge to resurrect without losing its small remaining weight. Readers retain old mapped snapshot ownership while files are unlinked on POSIX.

`storage.max_checkpoint_bytes` defaults to 64 MiB and bounds one checkpoint generation's data files. The main SQLite ceiling remains separate. Manifest overhead, a previous generation plus a new one during publication, and other temporary files require additional space. The cap is checked after generation creation, so it is not a peak disk/RAM cap. No checkpoint is made automatically on every write. A later database revision makes the checkpoint stale and currently causes full replay, not checkpoint-plus-tail replay. Checkpoints do not truncate or fold the authoritative event history. Records and lexical postings still load/rebuild on restart; this is not an LSM hierarchy or a zero-copy whole-service restore.

## Executed restart diagnostic

64 records, 101,046 association events, five fresh-process restarts per mode, alternating replay/checkpoint order with warm file caches. Median in-process open time: replay 51.03 ms; mmap checkpoint 7.24 ms. All trials recovered identical tick 17, embeddings, selected edge weight and retrieval packet. These timings exclude Python process startup and do not predict performance for large corpora or cold storage.

```sh
.venv/bin/python benchmarks/recovery.py
```

The full local timing artifact is `benchmarks/results/recovery/restart.json`. Regression tests cover exact packed-weight/tick restoration and future updates, dead-edge resurrection, mapped copy-on-write, no embedding regeneration, v1 migration, stale/corrupt fallback, forged offsets despite valid checksum, and abrupt CLI exit after a committed checkpoint. Abrupt process exit is not a physical power-loss test.

Native counters describe the current engine instance; restoring packed state does not reconstruct historical insert/update counter totals. `association_events` is the durable event count.
