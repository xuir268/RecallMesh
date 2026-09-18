# LoCoMo benchmark runs

Run from the project root:

```sh
.venv/bin/python benchmarks/locomo_retrieval.py
.venv/bin/python benchmarks/paper_case.py
.venv/bin/python -m pytest -q
```

`results/locomo/REPORT.md` contains the measured results. `summary.json` contains
configuration, per-category metrics, conversation statistics and timings.
`predictions.jsonl` contains all per-question retrievals for audit.

Data: official `snap-research/locomo`, `data/locomo10.json`:
https://github.com/snap-research/locomo/blob/main/data/locomo10.json
The downloaded license and upstream README are in `data/`.

HeLa-Mem reference: https://arxiv.org/pdf/2604.16839
Sections 4.1–4.2 evaluate generated answers using F1/BLEU-1 with GPT-4o-mini,
GPT-4o and Qwen backbones. This project's current test evaluates annotated
source-turn retrieval only. It is not a HeLa-Mem reproduction and its percentages
must not be compared with the paper's answer F1 percentages.

No API credentials or answer-generation model were available in the environment.
The run uses the project's existing deterministic 128-dimensional HashEmbedder
and a separate BM25 diagnostic baseline; neither is a learned semantic encoder.
Graph training simulates usage by replaying each conversation turn as a query
against the prefix seen so far, with 8 seeds and no spreading during ingestion.
It is a specified local protocol, not an asserted paper protocol. One session is
one decay tick. All four QA categories are evaluated after ingestion, with no
QA-driven reinforcement, no access to gold evidence during ranking and no
supplied summaries used as memory. Evidence labels are consulted only to score.

`paper_case.py` is a synthetic regression for the paper's Dr. Sarah illustration:
association history and initial relevance are supplied manually. It verifies that
spreading reaches the linked location evidence. It does not generate an answer
or reproduce the paper's numerical scores.

To run paper-style answer evaluation, configure an answer model and a learned
embedding encoder, implement the intended consolidation/retrieval protocol, then
record generated answers and score them with the official evaluator. The existing
retrieval traces provide a starting point; the current code deliberately does not
invent answers from gold labels or relabel retrieval scores as QA accuracy.

## Answer pipeline and small-model pilot

The earlier retrieval-only limitation above describes the initial run. The
current model-backed agent evaluation is documented in
`results/pipeline/REPORT.md`. Astra, Claude, and two pinned local Qwen3 artifacts
were tested with single retrieval, wider single retrieval, and the bounded
pipeline. Public/generated input provenance, first attempts, quota-resume
manifest, usage and complete-facts diagnostics are retained.

```sh
.venv/bin/python benchmarks/pipeline_eval.py prepare
.venv/bin/python benchmarks/pipeline_eval.py run --backend codex --label astra --workers 2 --arms single bounded wide
.venv/bin/python benchmarks/pipeline_report.py
.venv/bin/python benchmarks/pipeline_markdown.py
```

The final CLI also constrains selectable source IDs to candidate IDs. That
hardening occurred after the pilot; it has unit and live CLI coverage, while the
pilot scores retain the original invalid-ID failure. See README.md for the
single-command local launcher and library integration.
