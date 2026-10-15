# Missing-source and answerability audit — October 15

The six incomplete packets from the twelve-question answer pilot were reviewed against the original corpus and saved model outputs. No model calls were made. This is a post-hoc, qualitative review by the implementation assistant, not independent or blinded annotation. Original benchmark labels and scores remain unchanged.

## What the coverage count concealed

All **28** gold sources missing from16-record packets exist in the corpus. Of those, **27 pass entity gating**, **17 have zero stripped-query body BM25 score**, and **three have positive activation under the original two-hop query**. Every session is a connected adjacency chain, so finite two-hop traversal is a search limit, not graph disconnection. Positive lexical/activation scores also do not guarantee final top-k selection.

The64-record complementary packet recovers **11 of those28** sources. Its full-gold completion count remains6/12 because17 sources are still missing across those same six cases. Thus unchanged completion counts do **not** establish a hard recall ceiling or show that all missing facts are beyond ranking. They hide useful partial recovery and problematic gold-source sets.

| Case | Missing sources:16 →64 | Source/answer diagnosis |
| --- | ---: | --- |
| conv-42:q4 | 1 →1 | Allergy constraint exists, but does not establish the named safe pets in the reference. Evidence-only abstention is warranted. |
| conv-43:q18 | 4 →4 | Author query versus book-title evidence: real lexical/identity mismatch, plus weak visual/fandom annotations and author mappings absent from cited text. |
| conv-47:q34 | 1 →1 | Post-trip confirmation names a city. Needs trip linkage and city/country knowledge; nearest retrieved same-session turn is four edges away. Supplied packet contains plans, making Astra's refusal defensible. |
| conv-48:q11 | 3 →2 | Redundant marriage/husband references. Supplied D28:11 already supports the intended yes answer; incomplete gold coverage is not evidence of guessing. |
| conv-49:q11 | 18 →8 | Gold has19 sources, impossible to fit at16. Direct weight evidence is recovered at64; many gold turns are contextual and one concerns the other speaker. Another illness adds ambiguity. |
| conv-50:q68 | 1 →1 | Date-bearing reply uses a pronoun and needs its car/event context; six turns from nearest supplied same-session turn, beyond two hops. Genuine retrieval failure. |

Per-missing-source presence, lexical score/rank, gate eligibility, activation and same-session distance are saved in `benchmarks/protocols/missing-evidence-audit-v1.json`. Per-source qualitative interpretations are in `benchmarks/protocols/missing-evidence-review-v1.json`. Reproduce numeric audit with `.venv/bin/python benchmarks/audit_missing_evidence.py`; raw dialogue and answers stay in ignored local artifacts. Source: public LoCoMo, same fixed checksum and original records as prior studies.

## What can be said about the answers

On incomplete packets, Astra baseline answers4/6 and grounded2/6; Claude answers3/6 under either profile. This does not count unsupported guesses: Deborah's answer is supported by an available sufficient source, and Tim's shorter author answer has explicit support. Missing one of several redundant gold sources does not require abstention. Conversely, plans are not proof of completed travel, and a health concern without its specific condition justifies refusing to name the condition.

Astra has two new abstentions: the safe-pet and country cases. Both are defensible from the actual supplied packets. The safe-pet change has no F1 cost because baseline already scores zero. The country change accounts for the negative F1 contribution, partly offset by a phrasing gain elsewhere. Claude has no additional abstentions, so its negative F1 change cannot be explained by an increase in caution alone. One Claude F1 gain uses an external title/author mapping rather than an explicit fact in its cited evidence. Neither provider's F1 difference establishes better or worse grounding.

The correct conclusion is **no demonstrated improvement in token overlap, and grounding quality remains unresolved**, rather than declaring the grounded profile less accurate. Token F1 rewards overlap with a potentially unsupported reference and gives no credit for justified refusal. It also fails to distinguish a supported partial answer from an unsupported claim.

## Decision

Pause this answer-optimization track. Do not buy another pilot using token F1 or all-annotated-source coverage as its sole success criterion. Existing pilot F1 numbers remain historical descriptive measurements, not validated accuracy or honesty scores. The replicated narrow retrieval findings remain as reported, with corpus exposure and selection limitations.

Before any future answer evaluation, define support-sufficient source alternatives, review whether each reference is supported under the allowed knowledge policy, assess answerability from the *actual packet*, and distinguish justified refusal, supported partial answers and unsupported claims. Report answer coverage and grounding separately; a model judge alone would not certify these labels. This audit does not claim to have supplied independent labels or a validated replacement metric. Adjacency and the baseline answer profile remain defaults; no new activation method or automatic follow-up is introduced.
