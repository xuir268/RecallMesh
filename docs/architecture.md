# RecallMesh architecture

![User-provided proposed architecture](assets/architecture-proposal.png)

The image describes the target design, not a claim that every component exists.

| Diagram component | Current status |
|---|---|
| Multiple agents | Independent JSON CLI/Python clients can share one SQLite database; writes serialize |
| Go control plane / MCP / gRPC | Planned; implemented control plane is Python with JSON stdio |
| ANN seed + graph walk | BM25 seeding plus bounded graph walk implemented; HashEmbedder is a stub, production ANN is not implemented |
| Co-activation log | Durable SQLite association events with restart replay |
| C++ hot delta | Atomic packed edge table; rebuild for managed growth; lock-freedom depends on target atomic support |
| Cold CSR segments | Immutable CSR snapshots and optional POSIX mmap checkpoint snapshots implemented; multi-segment LSM consolidation remains planned |
| Consolidator | Lazy edge decay and snapshot pruning implemented; hub detection and background distillation planned |

Current data flow: agent → JSON CLI or Python API → bounded retrieval → C++ CSR snapshot → compact evidence → model → optional support check. Writes go to SQLite records/events and update the C++ cache. On restart, indexes rebuild from records and persisted embeddings; the graph restores a matching mmap checkpoint or replays the durable association log.

An edge is a statistical co-activation association, not a verified semantic relationship. Retrieval walks help discover related records; the model still has to interpret the evidence correctly. Quotas, explicit ticks, and retention control growth. This is a bounded local framework; it is not a distributed database.
