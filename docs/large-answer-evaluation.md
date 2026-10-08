# Full-cohort Astra/Claude answer evaluation

October 8, 2026. **100 existing-cohort questions, three evidence conditions, two models: 600 recorded first attempts (595 successful outputs and five Astra timeouts).** This includes the 20-question and six-question pilots. All seven eligible conversations are represented. Exactly 144 matching prior answers are reused, with 456 new model calls; this is an expansion, not an independent replication or fresh sample. The primary measure is token F1, not an independently adjudicated correctness rate.

| Model | Condition | Mean F1 | Abstentions |
|---|---|---:|---:|
| Astra | adjacency | 42.88% | 24/100 |
| Astra | hebbian_50 | 39.41% | 25/100 |
| Astra | oracle | 54.96% | 6/100 |
| Claude Sonnet 4.6 | adjacency | 44.57% | 20/100 |
| Claude Sonnet 4.6 | hebbian_50 | 41.02% | 21/100 |
| Claude Sonnet 4.6 | oracle | 57.39% | 4/100 |

## Paired comparison

- Astra: Hebbian minus adjacency **-3.47 percentage points**; paired conversation-cluster bootstrap 95% interval **[-9.05, -0.47] points**. Oracle minus adjacency: +12.08 points, cluster95 [+8.06, +15.69].
- Claude: Hebbian minus adjacency **-3.55 percentage points**; paired conversation-cluster bootstrap 95% interval **[-11.91, +3.11] points**. Oracle minus adjacency: +12.82 points, cluster95 [+4.41, +18.54].

## Paired signs and distributions — October 8

These paired diagnostics use the saved first-attempt scores; reused and newly generated cases are identified in the cache manifest. Wins and losses refer to token-F1 differences, not independently adjudicated semantic correctness. They are post-hoc; the registered decision rule remains unchanged.

| Model / candidate vs adjacency | Wins | Ties | Losses | Exact sign p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Astra / hebbian_50 | 12 | 74 | 14 | 0.8450 | 1.0000 |
| Astra / oracle | 38 | 50 | 12 | 0.0003 | 0.0012 |
| Claude / hebbian_50 | 21 | 57 | 22 | 1.0000 | 1.0000 |
| Claude / oracle | 39 | 45 | 16 | 0.0027 | 0.0080 |

Two-sided exact binomial sign tests exclude ties (absolute F1 difference ≤1e-12). The four exploratory question-level tests are Holm-adjusted together. Their nominal p values assume independent signs; these questions share seven conversations, so question-level p values are not cluster-robust. As a separate sensitivity analysis, signs of equal-weight conversation mean deltas give:

| Model / candidate | Positive / tied / negative conversations | Conversation-level exact p |
|---|---:|---:|
| Astra / hebbian_50 | 0 / 2 / 5 | 0.0625 |
| Astra / oracle | 7 / 0 / 0 | 0.0156 |
| Claude / hebbian_50 | 2 / 0 / 5 | 0.4531 |
| Claude / oracle | 6 / 0 / 1 | 0.1250 |

Seven conversations give little power; the conversation tests are also exploratory and unadjusted. Question signs assume independence across correlated questions; conversation-level results and the original cluster intervals are the relevant caution when interpreting small p values. A majority of positive question signs does not guarantee a positive mean: small wording gains can be outweighed by one large loss.

Distribution of paired deltas in **F1 percentage points** (candidate minus adjacency):

| Model / candidate | Minimum | 25th percentile | Median | 75th percentile | Maximum |
|---|---:|---:|---:|---:|---:|
| Astra / hebbian_50 | -100.00 | +0.00 | +0.00 | +0.00 | +100.00 |
| Astra / oracle | -100.00 | +0.00 | +0.00 | +20.17 | +100.00 |
| Claude / hebbian_50 | -100.00 | +0.00 | +0.00 | +0.00 | +100.00 |
| Claude / oracle | -86.67 | +0.00 | +0.00 | +19.74 | +100.00 |

[Every paired per-question delta and abstention flag (CSV)](assets/large-answer-paired-deltas.csv) is published; identifiers and metrics contain no raw dialogue. The JSON includes the same per-question records and transition breakdown.

## Abstention transitions

| Model / candidate | Abstain → answer | Answer → abstain | Answer → answer | Abstain → abstain | Error involved |
|---|---:|---:|---:|---:|---:|
| Astra / hebbian_50 | 3 | 5 | 69 | 20 | 3 |
| Astra / oracle | 17 | 0 | 74 | 6 | 3 |
| Claude / hebbian_50 | 5 | 6 | 74 | 15 | 0 |
| Claude / oracle | 16 | 0 | 80 | 4 | 0 |

- Astra: oracle converted 17 adjacency abstentions to answers (16 F1 wins, 1 ties, 0 losses). These conversions contribute +10.46 points of the +12.08-point overall gain. Changes where both conditions answered contribute +2.37 points, answer-to-abstention transitions contribute +0.00 points, and error-involved pairs contribute -0.75 points.
- Claude: oracle converted 16 adjacency abstentions to answers (13 F1 wins, 3 ties, 0 losses). These conversions contribute +8.38 points of the +12.82-point overall gain. Changes where both conditions answered contribute +4.45 points, answer-to-abstention transitions contribute +0.00 points, and error-involved pairs contribute +0.00 points.

## Call failures and successful-pair sensitivity

Failed first attempts remain zero scores in the primary analysis. They are not counted as abstentions or successful answers. No timeout retries were made. Error-involved pairs are separated in the transition table.

| Model | Failed calls /300 | Comparison | Successful pairs | Mean delta (F1 points) |
|---|---:|---|---:|---:|
| Astra | 5 | hebbian_50 − adjacency | 97 | -3.43 |
| Astra | 5 | oracle − adjacency | 97 | +13.23 |
| Claude | 0 | hebbian_50 − adjacency | 100 | -3.55 |
| Claude | 0 | oracle − adjacency | 100 | +12.82 |

This is a secondary complete-pairs sensitivity analysis; excluding failures can introduce selection bias and does not replace the primary result. Sanitized failure identifiers and reasons are published in the JSON.

This arithmetic decomposition does not establish causal mediation or semantic correctness. Signs count token-F1 improvements, including small wording changes.

## Questions outside the two recent answer pilots

This secondary slice contains 74 questions; it is not independently sampled and may have appeared in earlier project studies. All three conditions on these questions are newly generated in this expansion.

| Model | Adjacency F1 | Hebbian F1 | Oracle F1 |
|---|---:|---:|---:|
| Astra | 48.93% | 44.66% | 58.05% |
| Claude | 48.73% | 45.45% | 60.98% |


The preregistered positive-signal rule required at least +5 F1 points and a positive confidence lower bound for both models. **It did not pass. A reliable answer-quality advantage remains unestablished.** Adjacency remains the recommended profile.

## Evidence and reliability

| Model / condition | Complete required evidence | Abstentions | Successful calls | Invalid citation IDs |
|---|---:|---:|---:|---:|
| Astra / adjacency | 48/100 | 24/100 | 99/100 | 0/100 |
| Astra / hebbian_50 | 43/100 | 25/100 | 98/100 | 0/100 |
| Astra / oracle | 100/100 | 6/100 | 98/100 | 0/100 |
| Claude / adjacency | 48/100 | 20/100 | 100/100 | 0/100 |
| Claude / hebbian_50 | 43/100 | 21/100 | 100/100 | 0/100 |
| Claude / oracle | 100/100 | 4/100 | 100/100 | 0/100 |

Complete evidence means all upstream annotated source IDs occur in the packet. It does not guarantee those sources are sufficient or that a generated claim is supported. Valid citation IDs also do not establish entailment. Required-source recall and per-category F1 are included in the versioned results. Abstention scores zero against the benchmark reference, even when the supplied evidence genuinely cannot resolve the question.

## Locked design

Protocol and code were committed before generation (`20fab8d`, clarified `26e013b`); all 300 evidence-packet fingerprints and the 144-answer cache manifest were committed before new model calls (`4009621`). Selection includes every member of the existing 100-question cohort, without selecting for graph gains or losses. Earlier pilot results were known; this expansion is diagnostic, not a new confirmatory study. The cohort has previous retrieval exposure and is not a fresh held-out corpus. Four new requests ran concurrently per backend; cached pilot requests previously ran at two. Six prior-pilot oracle answers are newly generated while their two retrieval-arm answers are reused, creating a timing difference for those cases. Cache reuse requires exact input and requested-model matches and preserves the earliest stored attempt, including errors. All conditions use the same 16-record budget and question prompt, fresh sessions, medium effort, no tools, no retries and no model fallback. Gold answers and arm names never enter provider inputs. No answer or evaluation question trains the graph.

The retrieval conditions reproduce the frozen adjacency control and session-50 Hebbian graph from [the longitudinal protocol](longitudinal.md). Oracle puts all annotated original sources first, then fills to16 with Hebbian distractors and chronological records. This gives oracle a source-position advantage; it is a diagnostic ceiling, not a fair competing retriever. Record count is matched; token count is not. No no-memory or BM25 answer arm, repeated generation, or external judge was added.

## Interpretation limits

Token F1 penalizes verbose paraphrases and numeric spelling: “3 years” versus “three years” can receive a partial score despite agreement. List recall is also sensitive to separators. The existing scoring rules remained fixed after generation. Treat F1 as an overlap measure, not a percentage of semantically correct answers. This evaluation has no comprehensive independent semantic adjudication.

Some open-domain reference answers require knowledge not supplied by their annotated sources. For example, the reference names a specific charity while the cited conversation only mentions sports sponsors and an intention to give back. Another reference recommends coaching although the sources discuss several other future activities. A suspected-health reference infers a condition from a vague comment. These limitations were found after generation and flagged without removing questions or changing primary scores. Oracle failures cannot automatically be attributed to the models.

Claude reported $2.397 across 300 stored calls ($1.785 for new calls); Astra cost is unavailable. Sum of available successful-call latency including CLI startup was 2795.9 seconds for Astra and 1819.1 seconds for Claude. New calls ran four at a time per backend, while reused calls ran previously. These sums include historical call time, exclude unavailable timeout latency, and are not current wall-clock duration.

With 100 questions but still only seven conversation clusters, uncertainty remains large. The original cohort is selected for speaker-reference structure; the headline mean does not estimate general conversational question prevalence. Category counts are not balanced in the full cohort. One completion per condition/model does not measure generation variance. This evaluates context-conditioned answering through the CLI adapters, not an autonomous tool-using agent or the full managed-service throughput. Within-model comparisons are the relevant controls because provider harnesses differ. Astra metadata repeats the requested identifier rather than independently attesting a server-side model version.

[Cache provenance](../benchmarks/protocols/large-answer-cache-v1.json), [Protocol](../benchmarks/protocols/large-answer-v1.json), [input fingerprints](../benchmarks/protocols/large-answer-inputs-v1.json), [results and per-call metrics](../benchmarks/protocols/large-answer-results-v1.json). Raw questions, dialogue packets and completions stay under ignored `benchmarks/results/large-answer/`. Upstream dataset licensing and research attribution remain in the project notices.

```sh
.venv/bin/python benchmarks/large_answer_eval.py prepare
.venv/bin/python benchmarks/large_answer_eval.py run --backend codex
.venv/bin/python benchmarks/large_answer_eval.py run --backend claude
.venv/bin/python benchmarks/large_answer_eval.py score
.venv/bin/python benchmarks/large_answer_report.py
```

Repeated run commands use cached first attempts with matching input/model fingerprints. A new generation consumes account quota and will not reproduce exact answers. The existing authenticated adapters request `gpt-6-astra` and `claude-sonnet-4-6`, using the previously verified [official Codex structured-output interface](https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex).
