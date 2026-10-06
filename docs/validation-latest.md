# Validation rerun — October 7, 2026

Executed after durable embedding storage, mmap checkpoints, natural-reference protocols and concurrency artifact changes. Answer-generation benchmarks were not rerun.

```text
$ .venv/bin/python -m pytest tests -q
69 passed in 1.50s

$ ctest --test-dir build/native --output-on-failure
assoc_smoke: passed
csr_build_stress: passed
segment_recovery: passed
100% tests passed, 0 tests failed out of 3
Total Test time: 0.42 sec
```

Native build uses `-fsanitize=undefined`. All 45 native concurrent-agent runs passed exact final update checks, at 1/2/4/8/16 threads across three workloads and three repetitions. Wheel and source archive build checks passed. Source archives exclude datasets, model weights, result directories and environments.

Local ThreadSanitizer built successfully but all three CTests aborted during runtime initialization with “Interceptors are not working.” Explicit runtime preloading also failed. This is not a TSan pass. A Linux sanitizer workflow is added; check its result independently.

Natural-reference retrieval results are in [natural-evaluation.md](natural-evaluation.md), restart results in [recovery.md](recovery.md), and contention measurements in [concurrency.md](concurrency.md). These measurements do not prove better agent answer quality or a distinctive co-activation identity-resolution advantage.
