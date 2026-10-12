# Experimental entity-aware activation — October 11, 2026

Typed entity routing plus adjacency improved evidence retrieval in the fixed100-question cohort. The bounded answer follow-up is mixed; there is no established answer-quality improvement. The default product recommendation remains adjacency.

| Retrieval condition | Evidence recall@16 | All required evidence |
|---|---:|---:|
| BM25 | 52.08% | 42/100 |
| Frozen adjacency | 57.89% | 48/100 |
| Entity topic ranking alone | 52.77% | 42/100 |
| Entity topic ranking + adjacency | **60.36%** | **49/100** |

The combined method gained **2.47 percentage points** over adjacency, with paired conversation-cluster bootstrap95 interval **[+0.95, +4.18] points**. It improved seven questions, regressed one, and tied on92. This passes the preregistered retrieval signal of at least +2 points and a positive lower confidence bound. Entity routing alone did not beat adjacency. These are100 previously exposed speaker-reference questions across seven conversations, not a fresh holdout or validated multi-turn alias benchmark.

## What the code does

`EntityAwareRetriever` stores typed entity-to-record posting links for `authored_by` and explicit named `mentions`. A caller supplies speaker identity and optional alias mappings. First-person identity uses that supplied provenance. Authorship does not establish that every statement is about its author. No automatic third-person pronoun resolution, inferred relationships, relation entailment, or learned entity extraction is claimed.

For a query naming exactly one known entity, the retriever gathers its authored/mentioned records, removes that name from the topic query, ranks original bodies by BM25 topic relevance, and combines this with the existing adjacency activation restricted to those candidates. Scores use the existing maximum normalization and graph_weight=.4. Eight seeds, two hops and16 final records are fixed. Query name stripping and candidate gating both change retrieval; the result is not an isolated causal test of adding edges.

Queries with no recognized entity or multiple entities fall back exactly to original adjacency. Ambiguous third-person pronouns are not assigned to a person. The entity posting links live in Python; native C++ edges remain ordinary adjacency edges. This is a deterministic metadata-assisted extension, not an improved Hebbian learning policy. We did not increase eta, change decay, or tune hops after outcomes.

The structure is:

```text
Known query entity → authored/mentioned source candidates
Topic BM25 + adjacency activation → ranked original evidence
```

## Bounded answer follow-up

Twenty fixed medium-pilot questions were reused without selecting retrieval gains. Forty new first-attempt answers were generated with tools disabled, no fallback and no retries. Their comparator is the saved original adjacency answer, so different execution dates/model nondeterminism and concurrency settings confound the comparison. This is diagnostic, not a fresh held-out or fully concurrent randomized answer trial.

| Model | Cached adjacency F1 | New entity F1 | Win / tie / loss | Abstentions: adjacency / entity |
|---|---:|---:|---:|---:|
| Astra | 31.69% | 35.86% | 3 /15 /2 | 9 /7 |
| Claude Sonnet4.6 | 40.96% | 37.16% | 3 /13 /4 | 6 /6 |

Astra's paired gain is **+4.17 points**, cluster95 **[−0.51, +11.98]**. Claude's is **−3.80 points**, cluster95 **[−8.85, +3.34]**. Both intervals include zero. All40 calls completed successfully. Token F1 penalizes paraphrases and does not measure independently adjudicated semantic correctness. The existing annotation and oracle limitations remain. We cannot claim this makes either agent better.

## Use and limits

Import `EntityAwareRetriever` from `recallmesh` or `assoc_mem.entity`. It wraps an `AssociativeRetriever` whose snapshot and lexical index the caller owns. Add original body text with matching dense node IDs and trustworthy speaker provenance; provide aliases explicitly. `explain()` returns the chosen entity, topic query, candidate count and typed link provenance. Selected nodes reference original sources, not generated facts.

See [the runnable example](../examples/entity_retrieval.py):

```sh
.venv/bin/python examples/entity_retrieval.py
```

This experimental API is a build-once Python index. It is not a thread-safe mutable entity store or a new CLI retrieval mode. Rebuild it when memory records change; it rejects stale node counts. Same-size edits are not detected, so caller lifecycle discipline is required. Entity/alias metadata must be persisted by the caller if needed. Existing managed-storage configuration and quotas do not bound this extra index. Hard gating can omit evidence from another speaker or incorrectly annotated records; the observed regression is retained. Declared aliases support explicit identity links, but this dataset run uses speaker names only, not automatic natural alias recovery.

[Retrieval protocol](../benchmarks/protocols/entity-aware-v1.json) was committed before outcomes (`8f855ff`). [Answer protocol](../benchmarks/protocols/entity-answers-v1.json) and input fingerprints were committed before model calls (`ebaf6e7`). Results are [retrieval metrics and per-question scores](../benchmarks/protocols/entity-aware-results-v1.json) and [answer metrics and paired deltas](../benchmarks/protocols/entity-answer-results-v1.json). Full packets/traces/completions remain ignored under `benchmarks/results`; no raw dataset text is redistributed. This is an independent RecallMesh extension; research attribution remains in [research.md](research.md).

Reproduce from the existing upstream dataset:

```sh
.venv/bin/python benchmarks/entity_aware.py
.venv/bin/python benchmarks/entity_answers.py prepare
.venv/bin/python benchmarks/entity_answers.py run --backend codex
.venv/bin/python benchmarks/entity_answers.py run --backend claude
.venv/bin/python benchmarks/entity_answers.py score
```

The last two generation commands use authenticated accounts and retain cached first attempts. This implementation provides a new testable routing path; the positive retrieval result does not justify changing the default product or claiming general reasoning improvement.

## Mechanism diagnostics (October 13)

The original entity-aware experiment is the project's first passed explicitly preregistered retrieval criterion: +2.47 percentage points over adjacency, with a positive conversation-bootstrap lower bound. This is a narrow result on an exposed, predicate-selected cohort, not a general answer-quality result.

A post-hoc diagnostic now separates query rewriting from routing. Reproduce with `.venv/bin/python benchmarks/entity_ablation.py`; metrics and per-question outcomes are in `benchmarks/protocols/entity-ablation-results-v1.json`. All arms use the same 100 questions, budget 16, frozen graph, and no reinforcement or model calls.

| Arm | Recall@16 (%) | Complete packets / 100 |
| --- | ---: | ---: |
| Original adjacency | 57.89 | 48 |
| Adjacency with name-stripped query, no gating | 57.39 | 48 |
| Strip lexical query only; retain original graph seeds | 57.39 | 48 |
| Body-only index, original query, no gating | 55.15 | 46 |
| Body-only index, original query, gating | 56.96 | 47 |
| Body-only index, stripped query, no gating | 57.50 | 48 |
| Body-only index, stripped query, gating | 60.36 | 49 |

The cheap query preprocessor does not recover the gain: its paired change is −0.50pp, cluster95 [−1.42, 0.00]. Within the stripped-query body-index comparison, gating contributes +2.85pp, cluster95 [1.67, 4.29]. Gating includes candidate-local score normalization; this test does not isolate filtering from normalization. The original method changed corpus scoring as well as query rewriting and candidate selection. Within the body-index controls, rewriting helps even without gating, and gating helps even without rewriting. Consequently the gain should not be described as purely interaction. Arm means alone also do not establish statistical equivalence of entity-only and BM25. These diagnostics are post-hoc; no new promotion threshold or product-default change is inferred from them.

Outside the single-name predicate, all resolvable category 1–4 questions in the same seven conversations were checked: 122 multiple-name questions and seven zero-name questions. All 129 returned exactly the same ordered packets as original adjacency. Their respective recall is 51.31% and 85.71% for both methods. This is a no-harm check of exact delegation, not a fresh holdout or evidence that active entity routing generalizes. The zero-name sample is particularly small.

## Budgeting answer evaluations

Complete packets moved only 48 → 49. This is a warning against buying another answer pilot for this retrieval delta. It is not sufficient to predict an exact F1 change: net completion counts can hide gains and losses on different questions, and partial evidence and the contents of already-complete packets also affect answers.

In the saved full-cohort oracle comparison, the 52 questions newly receiving complete evidence have mean paired answer-F1 changes of +0.2434 for Astra and +0.2624 for Claude. The 48 already-complete questions change −0.0121 and −0.0172, respectively. These are descriptive, unadjusted observations including failed attempts under the original scoring policy. If one additional completion in a 100-question evaluation behaved like that oracle subgroup, its projected contribution would be roughly +0.24–0.26 F1 percentage points. That requires exchangeability and unchanged results elsewhere; an entity-selected packet is not oracle evidence, and a net gain of one does not satisfy those assumptions automatically.

Use this as a rough spending screen, not a causal conversion rate or power analysis. The proposed requirement of 8–10 completions for a 20-question pilot is not calibrated by these data. Before further model calls, specify the actual paired question set, the expected effect and acceptable uncertainty, and inspect gained/lost complete packets. No new model calls were made for these diagnostics.

The defensible product position remains metadata-assisted conversational retrieval, with adjacency as the default and entity routing experimental. Failure of this particular retrieval-feedback learning protocol does not establish that all learned co-activation methods fail.
