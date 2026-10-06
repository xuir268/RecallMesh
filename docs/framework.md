# Agent memory framework

RecallMesh is a local memory service, independent of the model. Your agent writes observations, retrieves a bounded evidence bundle before a model call, and includes that bundle in the prompt. This does not automatically attach memory to an existing Claude or Codex chat, change model weights, or guarantee improved answers.

```sh
recallmesh init --config memory.json
recallmesh remember --config memory.json --text "Mira keeps the silver pebble in the blue drawer."
recallmesh recall --config memory.json --query "Where is Mira's pebble?"
recallmesh stats --config memory.json
```

`init` creates a config without overwriting an existing one. Partial configs inherit defaults; unknown fields and invalid limits are rejected. Relative storage paths resolve against the config's directory. See `configs/memory.example.json` for all fields.

## Agent integration

```python
from recallmesh import MemoryClient

with MemoryClient("memory.json") as memory:
    memory.remember("Mira keeps the silver pebble in the blue drawer.")
    evidence = memory.recall("Where is Mira's pebble?")
    # Pass evidence["memories"] with the question to your own model provider.
    # Preserve source IDs so the agent can cite the supporting records.
```

`ManagedMemory` is the direct Python API. `FrameworkAgent(memory, provider)` adapts it to the existing `EvidencePipeline` and AnswerProvider interface. Other languages can spawn `recallmesh serve --config memory.json` and use newline-delimited JSON on stdin/stdout. This is a stdio protocol, not HTTP or MCP.

```json
{"id":1,"method":"remember","params":{"text":"Mira likes blue drawers."}}
{"id":2,"method":"recall","params":{"query":"Mira","limit":4,"mode":"associative"}}
{"id":3,"method":"associate","params":{"ids":[1,2]}}
{"id":4,"method":"advance","params":{"steps":1}}
{"id":5,"method":"forget","params":{"id":1}}
{"id":6,"method":"stats","params":{}}
```

Send each request on one line; each reply is one JSON line with the same request ID and either `ok: true, result: ...` or `ok: false, error: {code, message}`. Requests run sequentially within one connection. Invalid input does not terminate the server. Methods also include `get`, `compact`, and `checkpoint`. `associate` requires existing IDs. Errors include `quota_exceeded`, `invalid_request`, `config_error`, `storage_error`, and `request_too_large`. Recall IDs are strings for compatibility with model citations; mutation arguments use integers.

Python SDK calls have a timeout (30 seconds by default). No shell command interpolation is used. Independent clients can share one database; SQLite serializes writes and revision checks refresh stale caches.

Existing model connectors can use persistent memory:

```sh
recallmesh chat --config memory.json --backend codex --model gpt-6-astra
recallmesh chat --config memory.json --backend claude --model claude-sonnet-4-6
```

These require the corresponding installed/authenticated model CLI. `--backend local --model ALIAS --local-url http://127.0.0.1:8080` uses a running compatible local server. Model calls send retrieved evidence to the selected provider; the memory-only commands do not call a model. Persistent chat does not automatically seed demo facts. Use `--facts facts.json` to import observations.

## Resource and retention configuration

| Field | Default | Meaning |
|---|---:|---|
| storage.max_database_bytes | 64 MiB | Hard main SQLite file ceiling, rounded down to pages |
| storage.max_checkpoint_bytes | 64 MiB | One immutable checkpoint generation data budget; temporary generations need extra space |
| limits.max_nodes | 10,000 | Maximum retained records |
| limits.max_text_bytes | 16 MiB | Total UTF-8 text plus writer bytes |
| limits.max_record_bytes | 16 KiB | Per-record text plus writer bytes |
| limits.max_request_bytes | 64 KiB | Stdio request ceiling |
| graph.initial_capacity | 8,192 | Starting hash slots, power of two |
| graph.max_capacity | 262,144 | Maximum hash slots, power of two |
| graph.allow_growth | true | Rebuild at a larger capacity when insertion cannot fit |
| retention.overflow | reject | Reject additions; `oldest` opts into oldest-record eviction |
| retention.ttl_ticks | 0 | Disabled; positive values expire records after this many ticks |
| retrieval.max_context_bytes | 12,000 | Serialized evidence byte budget; whole records only |
| retrieval.default_limit / max_limit | 8 / 64 | Default and maximum returned record count |
| retrieval.hops | 2 | Default graph traversal depth, at most 8 |

Call `advance` at your chosen session boundary. Ticks are logical time, not elapsed wall-clock seconds. Reading does not reinforce links. Writing associates overlapping/nearby observations according to `graph.edge_policy` (`star`, `combined`, or `adjacent`); `associate` explicitly reinforces a set of records. Decay uses `lambda`, reinforcement uses `eta`, and `floor` controls pruning. Changing these three learning parameters on an existing database is rejected: use a new database so historical event replay retains its meaning.

Quotas reject the whole write transaction rather than silently losing graph updates. Opt-in eviction and TTL also remove the deleted records' association events. Public record IDs remain stable even when internal dense IDs change after deletion.

**These are not a total process-RAM cap.** Native hash slots consume 16 bytes each (4 MiB at the default maximum). CSR snapshots, embeddings, lexical indexes, Python objects, and temporary rebuilds use additional RAM. SQLite journals and VACUUM can require additional temporary disk space beyond the main-file ceiling. Do not set the database ceiling equal to every last free byte on disk.

SQLite records, persisted embeddings and association events are authoritative. On restart, the native graph is recovered from a valid matching mmap checkpoint or event replay. Lexical indexes still rebuild. Growth doubles the capacity up to the configured maximum and rebuilds; it does not migrate a live lock-free table. Repeated association events consume database space and increase startup/rebuild time until deleted or the database ceiling rejects more writes. Optional `recallmesh checkpoint` captures packed delta state, a mapped read snapshot, and embedding arrays; it does not truncate event history. Stale checkpoints currently fall back to full replay. This version is intended for bounded local workloads. `compact` vacuums unused database space; it does not fold event history. Changing resource config invalidates active clients using the older config; reconnect them.

## Distribution

The package code is MIT licensed. Build a wheel or source archive from `pyproject.toml`. Downloaded model weights, benchmark data/results, legacy copies, and compiled runtimes are excluded. See `THIRD_PARTY_NOTICES.md`; benchmark data is not relicensed as MIT. The project has not been published to PyPI by this change.

See [checkpoint recovery](recovery.md), [native concurrency measurements](concurrency.md), and [natural-reference retrieval evaluation](natural-evaluation.md) for tested behavior and limits.
