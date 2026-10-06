# RecallMesh methodology and agent-layer extensions

RecallMesh combines associative memory with a bounded evidence workflow. HeLa-Mem is the graph-memory inspiration; IRCoT is related research on interleaving retrieval and reasoning. The additional selection, verification, audit, and resource controls below are implemented in this project. This is not a full reproduction of either paper, nor evidence of superiority to either research system.

## Guide the next search

The agent retrieves candidate records, then a controller selects evidence and identifies missing facts. The controller may propose up to two follow-up queries per round using entities from the question or retrieved records. Results are pooled and previously selected bridge facts are prioritized in the next review packet. Duplicate queries are skipped. There are two rounds by default and at most three configured rounds; query text is truncated to 512 characters.

This relates to [IRCoT: Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://aclanthology.org/2023.acl-long.557/) (Trivedi et al., ACL 2023), which investigates using intermediate reasoning to guide retrieval. Our controller emits source IDs, missing-fact summaries, and queries rather than a private chain-of-thought transcript. Entity grounding is a prompt instruction; the runtime does not prove that every generated query uses grounded entities.

## Select facts that work together

The selector is prompted to preserve identity/alias links, quantities, timestamps, and relevant contradictory facts. It chooses a joint supporting set rather than independently taking the highest scoring records. Previously selected bridge records are prioritized when the review budget fills.

For example, answering where Mira's token is may require both “Mira calls her token the silver pebble” and “the silver pebble is in the blue drawer.” Keeping only the second record loses the identity bridge. Selection is still fallible: preservation is encouraged by the prompt and packet order, not guaranteed for every task.

## Give edges a meaning: current status

Edges currently mean **statistical co-activation**, with a weight and last-touched tick. They do not encode relation labels such as `alias_of`, `located_in`, `before`, `supports`, or `contradicts`. The selected source records and retrieval paths supply context for interpretation, but that does not make the stored edges semantic relations.

Typed edges are a proposed extension. Implementing them would require a relation type, direction, supporting source IDs, confidence, validation rules, and separate treatment of observed versus inferred relations. No extracted model relation should silently become an observed fact. This proposal is not included in the current test or answer-quality claims.

## Check whether the answer is supported

After answering from the compact source packet, a controller checks the draft against every selected original record, including identity links, arithmetic, dates, contradictions, and substantive claims. Unknown citation IDs are rejected; a substantive answer needs a citation. A failed support check replaces the draft with “Not enough information.” The original draft and check remain in the stage audit.

The checker is another model. It may approve an unsupported answer or reject a correct one; it is not a proof. Runtime schema validation establishes output shape, not semantic truth.

## Guardrails and operational tuning

| Control | Actual behavior |
|---|---|
| Candidate source IDs | Plan schema enumerates actual IDs; runtime rejects invented IDs |
| Structured outputs | Required fields, value types, and forbidden extra fields are checked |
| Bounded planning | 1–3 rounds, at most two follow-up queries per round, repeated queries skipped |
| Evidence budgets | Candidate/final record limits and serialized UTF-8 byte limits |
| Compression | Selects whole original records; does not truncate sentences or compress a model KV cache |
| Prompt boundary | Records and drafts are explicitly described as untrusted data; this is not proven prompt-injection immunity |
| Answer rejection | Invalid citations or failed model support checks cause abstention |
| Audit | Retrieval traces, reviewed/selected IDs, plans, drafts, checks, usage, and dropped records are recorded; failed plans are retained before validation raises |
| Memory management | Durable records/events, transactional quota rejection, stable IDs, opt-in eviction, TTL ticks, and bounded graph growth |
| Fact provenance | Generated answers are not automatically remembered as observations |

These are concrete engineering additions to this implementation. We have not performed a matched comparison of guardrail effectiveness against the original HeLa-Mem implementation. Do not describe them as a proven security or accuracy improvement over that paper.

## Implementation and evidence

- `src/assoc_mem/pipeline.py`: planning, joint selection, bounded follow-up retrieval, whole-record fitting, support checking, audit and usage.
- `src/assoc_mem/agent.py`: provider transport, answer parsing, citation validation.
- `src/assoc_mem/framework.py`, `config.py`, and `memory_cli.py`: persistence, resource limits, memory lifecycle, and input bounds.
- `tests/test_pipeline.py`, `tests/test_agent.py`, and `tests/test_framework.py`: regression checks. Typed semantic edges are not implemented or tested.

See [evaluation results and limitations](evaluation.md), [executed test output](validation-latest.md), and [agent integration](framework.md). The earlier answer-quality pilot shows mixed, model-dependent outcomes and does not isolate each methodology component. The full quality pilot was not rerun after source-ID hardening or through the new persistent service.
