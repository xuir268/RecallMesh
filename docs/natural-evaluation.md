# Natural-reference retrieval evaluation — October 7, 2026

Two screens were run. Neither is a completed, independently annotated 100-case multi-turn alias benchmark. No new Astra/Claude answer-quality evaluation was run.

## 100 unedited LoCoMo QA: speaker coreference

The primary cohort contains 100 original questions with original upstream evidence IDs, sampled with a fixed seed from 258 eligible questions in the seven conversations outside the earlier three-conversation development set. Each question names one speaker; at least one evidence turn starts with I/my, was spoken by that speaker, and omits their name in its body. This establishes a narrow first-person-to-speaker identity link using the dataset's speaker annotation. It does not establish a multi-turn alias chain or require graph traversal to resolve the identity. These conversations have appeared in earlier project evaluations, so they are not a newly unseen corpus.

Records include dates, speaker metadata, original text, and supplied image captions equally across retrieval arms. Settings were inherited from the framework: star graph construction, weight 0.4, two hops, eight lexical seeds, 16 returned records. Graphs were built chronologically from conversation turns only, without QA questions, answer text, evidence labels, or mined cohort membership. Evaluation is read-only. Membership was saved before scoring. QA selection is independent of retrieval outcomes.

| Retrieval arm | Mean gold evidence recall @16 | All gold evidence returned |
|---|---:|---:|
| lexical | 52.08% | 42% |
| associative | 58.31% | 48% |
| adjacency_only | 57.89% | 48% |
| learned_only | 54.16% | 43% |
| shuffled | 52.14% | 42% |

The graph adds 6.23 percentage points over BM25 on this cohort, with 12 improved cases and two regressions. A conversation-cluster bootstrap interval for the paired mean difference is +1.61 to +12.31 points (3,000 fixed-seed resamples, only seven clusters). This is exploratory uncertainty for this selected cohort, not a population accuracy guarantee. There is only a 0.42-point difference between combined star retrieval and adjacency-only. The evidence currently supports an adjacency contribution much more strongly than a distinctive Hebbian co-activation advantage.

Five evidence-packet conditions were prepared: lexical, associative, oracle, counterfactual, and bridge removed. Oracle is supplied upstream gold. Counterfactual replaces selected reference evidence with an unchanged original record from the other speaker. Bridge removed drops a traversed non-target intermediate when present; it leaves speaker metadata intact. These controls are not equivalent to the generated pilot's altered factual-world intervention. Without model answer calls they do not test answer sensitivity or semantic entailment. Their packet coverage is recorded in the JSON artifact, but is not treated as a retrieval-method competition. The bridge-removal coverage remains unchanged, which is not a causal identity-resolution result.

## 100 pronoun-opening adjacent-turn candidates

A separate heuristic screen samples 100 of 123 original adjacent-turn pairs, where the next turn begins with a pronoun and both turns meet length rules. The query is the unedited preceding turn, not a LoCoMo QA question. BM25 retrieves the target in 30 cases; combined graph retrieval in 76; adjacency-only in 100; learned-only in 31; shuffled topology in 22. Only eight targets have zero lexical score.

This screen is strongly biased toward adjacency by its mining rule. All 100 annotations remain explicitly unreviewed: expletive “it,” affective replies, and ambiguous antecedents can be false positives. The 76-versus-30 gap must not be advertised as 100 verified identity chains or as a distinctive co-activation result. Cases, source IDs, full original excerpts and annotation placeholders are saved locally for blind review. Validated membership should be locked without resampling for favorable retrieval outcomes.

## Reproduce and inspect

```sh
.venv/bin/python benchmarks/natural_reference_qa.py
.venv/bin/python benchmarks/natural_coreference.py
```

Versioned source SHA-256, settings, source-ID membership, and summaries are in `benchmarks/protocols`. Full local per-case packets and traces are in `benchmarks/results/natural-reference-qa` and `benchmarks/results/natural-coreference`; dialogue excerpts and datasets are excluded from Git/package distributions. Download the upstream LoCoMo dataset separately under CC BY-NC 4.0.

The needed next evidence is still 100 independently reviewed multi-turn aliases/coreference chains, with explicit referents and bridge requirements, matched controls, and answer scoring. The current tests do not justify “co-activation graphs recover entity references that lexical retrieval cannot.”
