# Small Astra/Claude answer-quality evaluation

October 7, 2026. Six randomly selected original questions, two evidence conditions, two models: **24 first-attempt answers**. No retries or extra judge calls. The pilot found no answer-correctness advantage for session-50 Hebbian retrieval.

| Model | Adjacency correct | Hebbian correct | Adjacency mean F1 | Hebbian mean F1 |
|---|---:|---:|---:|---:|
| Astra (`gpt-6-astra`) | 1/6 | 1/6 | 5.56% | 5.56% |
| Claude Sonnet 4.6 | 1/6 | 1/6 | 5.26% | 3.03% |

F1 is the preregistered primary metric, using the existing upstream-style token normalization and category handling. The correct counts are a **separate post-hoc author review**, not a blinded independent adjudication. Verbose correct paraphrases receive low F1 against the short reference; this explains why all strict F1=1 counts are zero while review found one correct answer per condition per model. Do not interpret the F1 difference as established semantic harm.

Both evidence arms supplied all required upstream sources for only **one of six questions**. That same question received a correct, source-supported answer from both models in both conditions. Astra abstained on the other five questions in each arm. Claude abstained five times with adjacency and four times with Hebbian evidence; its additional answer proposed a career different from the benchmark reference, citing generic encouragement rather than a source establishing that career. This did not improve correctness.

All 24 calls completed successfully with valid output format and no invalid citation IDs. Citation-ID validity and required-source coverage are provenance checks; they do not establish semantic entailment. The observed bottleneck is insufficient evidence in this small sample. There is no oracle arm, so this test does not isolate the models' ability to answer every question with complete evidence.

## Locked design

[Protocol](../benchmarks/protocols/small-answer-v1.json) and runner were committed as `d485c30` before model calls. Six IDs were selected by seeded random sampling from the fixed 100-question cohort, without choosing known wins or losses. The arms reproduce frozen adjacency versus the graph after 50 replay sessions from [the longitudinal evaluation](longitudinal.md). Each supplies 16 original records, identical metadata, fixed graph settings and the same question-only answer prompt. Gold labels and arm names do not enter prompts. Record count is matched; token counts are not necessarily identical.

Both models ran through existing authenticated CLI adapters, in fresh ephemeral sessions at medium effort with tools disabled. No model substitution or generated-answer learning occurred. Requested and reported model identifiers matched; Astra metadata uses the requested identifier and is not independent server-side model attestation. CLI wrappers differ, so compare evidence conditions within each model rather than treating this as a model leaderboard. The Astra structured-output invocation follows [official Codex documentation](https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex).

Claude reported approximately **$0.103** for its 12 calls; Astra cost is unavailable through this adapter. Sum of per-call latency, including CLI startup, was 139.2 seconds for Astra and 73.8 seconds for Claude; two calls ran concurrently per backend, so these are not wall-clock totals.

Six questions from a previously evaluated narrow speaker-reference cohort are too few to claim population-level improvements or significance. Adjacency remains the recommended product profile. This pilot does not overturn the longitudinal stopping rule.

[Results and per-call metrics](../benchmarks/protocols/small-answer-results-v1.json) and [manual audit labels](../benchmarks/protocols/small-answer-audit-v1.json) are versioned without raw dialogue or answer text. Full local jobs and completions remain under ignored `benchmarks/results/small-answer/`. Dataset licensing and research attribution remain in the existing notices.

```sh
.venv/bin/python benchmarks/small_answer_eval.py prepare
.venv/bin/python benchmarks/small_answer_eval.py run --backend codex
.venv/bin/python benchmarks/small_answer_eval.py run --backend claude
.venv/bin/python benchmarks/small_answer_eval.py score
```

The runner retains cached first attempts rather than requesting replacements. Reproduction of model calls requires authenticated CLIs and consumes account quota; generation is not deterministic. Saved manual review applies to the archived run, not arbitrary newly generated answers.
