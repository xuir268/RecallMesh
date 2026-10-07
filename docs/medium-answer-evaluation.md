# Medium Astra/Claude answer evaluation

October 7, 2026. **20 fixed questions, three evidence conditions, two models: 120 first-attempt answers.** Five questions per category (multi-hop, temporal, open-domain inference, single-hop); the six-question pilot was excluded. All seven eligible conversations are represented. The primary measure is token F1, not an independently adjudicated correctness rate.

| Model | Condition | Mean F1 | Abstentions |
|---|---|---:|---:|
| Astra | adjacency | 31.69% | 9/20 |
| Astra | hebbian_50 | 30.14% | 8/20 |
| Astra | oracle | 45.04% | 4/20 |
| Claude Sonnet 4.6 | adjacency | 40.96% | 6/20 |
| Claude Sonnet 4.6 | hebbian_50 | 36.01% | 7/20 |
| Claude Sonnet 4.6 | oracle | 47.70% | 2/20 |

## Paired comparison

- Astra: Hebbian minus adjacency **-1.56 percentage points**; paired conversation-cluster bootstrap 95% interval **[-6.47, +1.90] points**. Oracle minus adjacency: +13.35 points.
- Claude: Hebbian minus adjacency **-4.96 percentage points**; paired conversation-cluster bootstrap 95% interval **[-23.41, +3.58] points**. Oracle minus adjacency: +6.74 points.

## Paired signs and distributions — added October 8

These diagnostics use the original saved scores with no new model calls. Wins and losses refer to token-F1 differences, not independently adjudicated semantic correctness. They are post-hoc; the registered decision rule remains unchanged.

| Model / candidate vs adjacency | Wins | Ties | Losses | Exact sign p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Astra / hebbian_50 | 5 | 14 | 1 | 0.2188 | 0.5840 |
| Astra / oracle | 9 | 8 | 3 | 0.1460 | 0.5840 |
| Claude / hebbian_50 | 7 | 11 | 2 | 0.1797 | 0.5840 |
| Claude / oracle | 7 | 10 | 3 | 0.3438 | 0.5840 |

Two-sided exact binomial sign tests exclude ties (absolute F1 difference ≤1e-12). The four exploratory question-level tests are Holm-adjusted together. Their nominal p values assume independent signs; these questions share seven conversations, so question-level p values are not cluster-robust. As a separate sensitivity analysis, signs of equal-weight conversation mean deltas give:

| Model / candidate | Positive / tied / negative conversations | Conversation-level exact p |
|---|---:|---:|
| Astra / hebbian_50 | 3 / 3 / 1 | 0.6250 |
| Astra / oracle | 5 / 0 / 2 | 0.4531 |
| Claude / hebbian_50 | 2 / 3 / 2 | 1.0000 |
| Claude / oracle | 3 / 2 / 2 | 1.0000 |

Seven conversations give little power; the conversation tests are also exploratory and unadjusted. Neither level establishes a significant sign advantage here. A majority of positive question signs does not guarantee a positive mean: small wording gains can be outweighed by one large loss.

Distribution of paired deltas in **F1 percentage points** (candidate minus adjacency):

| Model / candidate | Minimum | 25th percentile | Median | 75th percentile | Maximum |
|---|---:|---:|---:|---:|---:|
| Astra / hebbian_50 | -87.50 | +0.00 | +0.00 | +0.40 | +40.00 |
| Astra / oracle | -49.35 | +0.00 | +0.00 | +25.52 | +94.00 |
| Claude / hebbian_50 | -100.00 | +0.00 | +0.00 | +2.92 | +50.00 |
| Claude / oracle | -86.67 | +0.00 | +0.00 | +12.55 | +89.00 |

[Every paired per-question delta and abstention flag (CSV)](assets/medium-answer-paired-deltas.csv) is published; identifiers and metrics contain no raw dialogue. The JSON includes the same per-question records and transition breakdown.

## Abstention transitions

| Model / candidate | Abstain → answer | Answer → abstain | Answer → answer | Abstain → abstain |
|---|---:|---:|---:|---:|
| Astra / hebbian_50 | 2 | 1 | 10 | 7 |
| Astra / oracle | 5 | 0 | 11 | 4 |
| Claude / hebbian_50 | 2 | 3 | 11 | 4 |
| Claude / oracle | 4 | 0 | 14 | 2 |

For Astra, oracle converted five adjacency abstentions into answers; all five increased F1. Those conversions contribute **+12.23 points** of the **+13.35-point** mean gain; changes where both conditions answered contribute +1.11 points. For Claude, oracle converted four abstentions: three increased F1 and one remained at zero. Conversions contribute **+8.04 points**, partly offset by **−1.30 points** where both conditions answered, leaving +6.74 points overall. No oracle answer reverted to abstention.

This is an arithmetic decomposition of observed F1 gain, not a causal mediation estimate or proof that every new answer was correct. Hebbian evidence also converted two abstentions for each model, but introduced one new abstention for Astra and three for Claude; these larger losses outweighed its smaller gains.


The preregistered positive-signal rule required at least +5 F1 points and a positive confidence lower bound for both models. **It did not pass. A reliable answer-quality advantage remains unestablished.** Adjacency remains the recommended profile.

## Evidence and reliability

| Model / condition | Complete required evidence | Abstentions | Successful calls | Invalid citation IDs |
|---|---:|---:|---:|---:|
| Astra / adjacency | 7/20 | 9/20 | 20/20 | 0/20 |
| Astra / hebbian_50 | 6/20 | 8/20 | 20/20 | 0/20 |
| Astra / oracle | 20/20 | 4/20 | 20/20 | 0/20 |
| Claude / adjacency | 7/20 | 6/20 | 20/20 | 0/20 |
| Claude / hebbian_50 | 6/20 | 7/20 | 20/20 | 0/20 |
| Claude / oracle | 20/20 | 2/20 | 20/20 | 0/20 |

Complete evidence means all upstream annotated source IDs occur in the packet. It does not guarantee those sources are sufficient or that a generated claim is supported. Valid citation IDs also do not establish entailment. Required-source recall and per-category F1 are included in the versioned results. Abstention scores zero against the benchmark reference, even when the supplied evidence genuinely cannot resolve the question.

## Locked design

Protocol and code were committed before generation (`afd9eac`); all 60 evidence-packet fingerprints were committed before model calls (`99eec1c`). Selection was category-stratified seeded sampling from the existing 100-question cohort, without choosing graph gains or losses. The cohort has previous retrieval exposure and is not a fresh held-out corpus. All conditions use the same 16-record budget and question prompt, fresh sessions, medium effort, no tools, no retries and no model fallback. Gold answers and arm names never enter provider inputs. No answer or evaluation question trains the graph.

The retrieval conditions reproduce the frozen adjacency control and session-50 Hebbian graph from [the longitudinal protocol](longitudinal.md). Oracle puts all annotated original sources first, then fills to16 with Hebbian distractors and chronological records. This gives oracle a source-position advantage; it is a diagnostic ceiling, not a fair competing retriever. Record count is matched; token count is not. No no-memory or BM25 answer arm, repeated generation, or external judge was added.

## Interpretation limits

Token F1 penalizes verbose paraphrases and numeric spelling: “3 years” versus “three years” can receive a partial score despite agreement. List recall is also sensitive to separators. The existing scoring rules remained fixed after generation. Treat F1 as an overlap measure, not a percentage of semantically correct answers. This medium pilot has no comprehensive independent semantic adjudication.

Some open-domain reference answers require knowledge not supplied by their annotated sources. For example, the reference names a specific charity while the cited conversation only mentions sports sponsors and an intention to give back. Another reference recommends coaching although the sources discuss several other future activities. A suspected-health reference infers a condition from a vague comment. These limitations were found after generation and flagged without removing questions or changing primary scores. Oracle failures cannot automatically be attributed to the models.

Claude reported $0.509 for its 60 calls; Astra cost is unavailable. Sum of individual call latency including CLI startup was 588.0 seconds for Astra and 289.7 seconds for Claude. Two calls ran concurrently per backend, so these sums are not wall-clock duration.

With 20 questions and seven conversation clusters, uncertainty remains large. Category balancing means the headline mean does not estimate the original corpus category prevalence. One completion per condition/model does not measure generation variance. This evaluates context-conditioned answering through the CLI adapters, not an autonomous tool-using agent or the full managed-service throughput. Within-model comparisons are the relevant controls because provider harnesses differ. Astra metadata repeats the requested identifier rather than independently attesting a server-side model version.

[Protocol](../benchmarks/protocols/medium-answer-v1.json), [input fingerprints](../benchmarks/protocols/medium-answer-inputs-v1.json), [results and per-call metrics](../benchmarks/protocols/medium-answer-results-v1.json). Raw questions, dialogue packets and completions stay under ignored `benchmarks/results/medium-answer/`. Upstream dataset licensing and research attribution remain in the project notices.

```sh
.venv/bin/python benchmarks/medium_answer_eval.py prepare
.venv/bin/python benchmarks/medium_answer_eval.py run --backend codex
.venv/bin/python benchmarks/medium_answer_eval.py run --backend claude
.venv/bin/python benchmarks/medium_answer_eval.py score
.venv/bin/python benchmarks/paired_answer_analysis.py
.venv/bin/python benchmarks/medium_answer_report.py
```

Repeated run commands use cached first attempts with matching input/model fingerprints. A new generation consumes account quota and will not reproduce exact answers. The existing authenticated adapters request `gpt-6-astra` and `claude-sonnet-4-6`, using the previously verified [official Codex structured-output interface](https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex).
