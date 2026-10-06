# Shared-graph contention artifact — October 7, 2026

A standalone C++ benchmark runs 1, 2, 4, 8, and 16 logical agents as threads against **one EdgeStore**. Each thread writes and performs a one-hop snapshot graph walk every four writes. A separate publisher freezes approximately every 2 ms plus build time. There are three repetitions per cell and 100,000 writes per agent per run. The hot-edge arm shares one key; disjoint agents use separate keys; uniform access covers 4,096 preinserted edges.

All 45 runs passed exact final reinforcement checks and reported no rejected updates. Reported 64-bit atomics are lock-free on this machine. Snapshot shared-pointer publication/retention is not claimed to be lock-free; it can contend independently of edge CAS operations. Weight decay is disabled and tick zero is fixed to isolate reinforcement/count correctness. This workload does not evaluate tick ordering, live table growth, distributed agents, or power-loss recovery.

| Workload | Agents | Median million writes/s | Median CAS retries/write |
|---|---:|---:|---:|
| hot_edge | 1 | 20.37 | 0.00000 |
| hot_edge | 2 | 25.07 | 0.19786 |
| hot_edge | 4 | 15.12 | 1.41645 |
| hot_edge | 8 | 4.94 | 4.10508 |
| hot_edge | 16 | 4.62 | 4.72688 |
| disjoint | 1 | 58.78 | 0.00000 |
| disjoint | 2 | 23.89 | 0.00000 |
| disjoint | 4 | 18.98 | 0.00000 |
| disjoint | 8 | 11.83 | 0.00000 |
| disjoint | 16 | 10.37 | 0.00000 |
| uniform | 1 | 41.74 | 0.00000 |
| uniform | 2 | 25.67 | 0.00000 |
| uniform | 4 | 22.02 | 0.00000 |
| uniform | 8 | 7.17 | 0.00010 |
| uniform | 16 | 6.43 | 0.00009 |

Throughput does not scale linearly. Even disjoint writers slow down while CAS retries stay zero, so the bottleneck cannot be explained solely by edge CAS conflicts. Counter publication, snapshot ownership, CPU scheduling, and false sharing are possible contributors; these data do not isolate their individual costs. Hot-edge retries rise sharply with contention. No universal speedup or priority/novelty claim against published memory systems follows from this artifact.

The clock covers concurrent agent work and excludes setup and final validation. Reader calls are included in the workload, not counted as write throughput. Timer sampling occurs once per 64 writes; CSV latency fields are sampled means, not p95/p99. Agents are not pinned, order is fixed, frequency/thermal behavior is uncontrolled, and small-thread runs are short. Treat the table as a reproducible diagnostic on one machine, not a production capacity estimate.

Raw measurements are [concurrency-results.csv](concurrency-results.csv). Environment: macOS-27.0.1-arm64-arm-64bit-Mach-O, arm64, AppleClang 15, Release C++20. No Python/GIL or SQLite is in these timings.

```sh
cmake -S . -B build/concurrency -DASSOC_BUILD_PYTHON=OFF -DASSOC_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/concurrency -j 4
build/concurrency/concurrent_agents 100000 > agents.csv
```

The framework's independent CLI clients share SQLite durability but serialize transactions and maintain per-process native caches; they do not inherit the throughput of this single shared native-engine benchmark. `ManagedMemory` also serializes its API calls per instance. The native engine can be shared by caller-managed threads with compatible lifetimes; do not infer whole-agent framework lock-freedom.

Local UBSan checks passed. Local TSan still aborts during runtime initialization with “Interceptors are not working,” including after explicitly preloading the compiler's runtime. This is not a TSan pass and does not prove or disprove a data race. A Linux sanitizer workflow has been added; its status must be checked separately before claiming a clean TSan artifact.
