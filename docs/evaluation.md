# Test results, agent impact, and limitations

Research inspiration: [HeLa-Mem](research.md). The reported results below belong to RecallMesh, not the paper.

## Framework checks

Local validation on October 6, 2026: 58 Python tests and 2 native CTests passed. Wheel and source archive built successfully. Tests cover restart preservation of associations and ticks, stable IDs after deletion, rejection rollback, eviction, TTL, UTF-8 byte quotas, graph growth, database limits, stale-cache refresh, independent CLI processes, malformed/oversized JSON, and agent pipeline integration. ThreadSanitizer could not initialize on this machine; no TSan-clean claim is made.

[Latest executed test output](validation-latest.md): 58 Python tests and 2 native tests passed after the documentation and branch update.

These checks establish implementation behavior. They do not establish improved model answers.

## Earlier answer-quality pilot

The earlier pilot used 12 fixed questions (six generated alias/arithmetic/temporal cases and six public LoCoMo questions), three retrieval arms, and four model artifacts. Generated scores are exact answer accuracy; LoCoMo scores are answer F1, not exact accuracy. Format failures remain failures. These are small diagnostic samples. They predate the framework rename and persistent service; the full quality pilot has not been rerun through this service.

| Model | Generated exact, single → bounded | LoCoMo mean F1, single → bounded |
|---|---:|---:|
| Astra | 2/6 → 5/6 | 45.18 → 45.18 |
| Claude | 2/6 → 5/6 | 66.15 → 77.26 |
| Qwen3 1.7B Q8_0 | 2/6 → 1/6 | 44.85 → 73.61 |
| Qwen3 4B Q4_K_M | 1/6 → 1/6 | 60.28 → 52.70 |

The bounded pipeline uses follow-up searches, joint evidence selection, whole-record compression, and a support check. Its gains cannot be attributed solely to the graph: it also uses more model calls and sees more candidates. The wider one-call control was included in the original report. Astra's natural final packet averaged about 71% fewer bytes, but total input across calls and latency increased. Claude generated median latency increased from 4.47 to 16.94 seconds; Astra generated median latency from 9.95 to 34.35 seconds. Cross-provider latency includes different transports and is not pure inference speed.

The full local report is `benchmarks/results/pipeline/REPORT.md` with per-question traces in that directory. Those artifacts are excluded from package distributions. The public source dataset is [LoCoMo](https://github.com/snap-research/locomo/blob/main/data/locomo10.json), licensed CC BY-NC 4.0. Post-pilot source-ID schema hardening has unit coverage; reported pilot failures were retained and the pilot was not rerun after that hardening.

## Methodology coverage

[The methodology guide](methodology.md) separates implemented follow-up search, joint fact selection, citation validation, support checks, and bounded resource controls from proposed typed semantic edges. The pilot compares whole pipelines; it does not isolate the effect of each addition or benchmark guardrails against HeLa-Mem.

## What changes for an agent

The agent can retain observations between sessions and retrieve linked source records before answering. This can recover bridge facts that a single search misses and reduce the final evidence packet. It does not change the model's learned reasoning ability. Ordinary app chats are not automatically connected. The agent must call the memory interface and pass the returned evidence into its model prompt.

Use `remember` for observations, `recall` before model calls, and preserve record IDs for citations. `associate` explicitly strengthens known records; `advance` sets session boundaries. Do not automatically save generated answers as observed facts. The existing `FrameworkAgent` plus `EvidencePipeline` supports the bounded workflow. See [the integration guide](framework.md).

## Limits

- Better answers are task/model dependent; Qwen4 regressed on the sampled natural questions. A larger held-out evaluation is needed.
- Current seed retrieval is lexical; hashed embeddings are not evidence of semantic retrieval quality.
- Co-activation edges are untyped and can connect irrelevant records. The support checker is another fallible model, not a proof of correctness.
- Graph traversal has bounded work but still reads adjacency entries. No zero-latency or universal HPC-performance claim is made.
- Resource quotas cover records, text, main database file, and graph slots; additional indexes, snapshots, rebuilds, journals, and VACUUM need RAM/disk.
- Startup and growth replay association history. There is no event checkpoint folding or persistent mmap CSR yet.
- Concurrent clients share serialized SQLite writes; this is not a distributed or tenant-isolated service.
- Go/MCP/gRPC, production ANN, hub detection, and background distillation shown in the proposal are future work.
