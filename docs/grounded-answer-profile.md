# Grounded answer profile

An experimental single-call answer profile makes speaker attribution, utterance/event dates, completed-versus-planned activities and list completeness explicit. It retains the existing JSON answer schema, citation validation and evidence-only boundary. It does not add memory facts, enlarge retrieval packets, or train model weights.

```python
from assoc_mem.agent import CliProvider, Evidence

provider = CliProvider('claude', answer_profile='grounded')
# For Astra: CliProvider('codex', answer_profile='grounded')
response = provider.answer(question, [Evidence(record_id, original_record_text)])
```

The default profile remains `baseline`. This option also works when supplying the provider to `MemoryAgent`; it does not require changing graph construction. The general bounded evidence-selection/verification pipeline remains separately available, with additional model-call cost.

## Fixed paired pilot

Protocol and input hashes were committed as `ee338e5` before generation. Twelve questions were selected by seed 20261013, three per category, from the 159 residual entity-study questions. Prior corpus/retrieval exposure remains; this is a diagnostic pilot, not pristine holdout validation. Each model receives exactly the same original entity-retrieved 16-record packet in both arms. Both arms use contemporary fresh sessions, medium effort and one model call; 48 first attempts total, with no retries, substitute models, gold labels or arm labels in prompts.

The primary descriptive metric is unchanged upstream-style token F1, with paired conversation-bootstrap95, wins/ties/losses, abstention and failure counts. Token overlap rewards brevity and cannot establish semantic correctness. In particular, moving an explanation into rationale can improve F1 without adding any factual knowledge. Speaker/date/list instructions can also cause unnecessary abstention. No automatic product-default promotion follows from this small pilot.

Raw completions and source excerpts remain local in ignored `benchmarks/results/grounded-answers`. Only aggregate metrics, question IDs and per-call scores are versioned. The source is the public [LoCoMo benchmark](https://github.com/snap-research/locomo/blob/main/data/locomo10.json), CC BY-NC 4.0; no private user memory is sent.

The fixed-input comparison follows the principle of task-specific evaluations rather than impressions of answer quality; see [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices). No external judge was purchased for this pilot; any source review is explicitly qualitative rather than an independent correctness score.

## Outcome: no demonstrated improvement

All 48 first attempts succeeded, with no invalid citation IDs. Results are in `benchmarks/protocols/grounded-answers-results-v1.json`.

| Model | Baseline F1 (%) | Grounded F1 (%) | Change (pp) | Cluster95 (pp) | Wins / ties / losses | Abstentions |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Astra | 44.70 | 42.02 | −2.68 | [−10.00, +4.46] | 1 / 10 / 1 | 2 → 4 |
| Claude | 46.23 | 44.45 | −1.77 | [−9.22, +3.94] | 3 / 6 / 3 | 3 → 3 |

This pilot does not support better answers. The grounded profile is available for research use only and should not be presented as an upgrade; the default remains unchanged. The gain on Astra's car-modification question is chiefly phrasing rather than new facts. Claude's author-list answer gained an item, while some other differences were longer phrasing or geographic qualifiers. These are qualitative observations by the implementation agent, not independent semantic adjudication. A partial country answer became an abstention on Astra, illustrating the cost of stronger support instructions.

Only six of twelve 16-record packets contain every original required source. A subsequent exploratory preflight with 32 entity records also had six complete packets. A complementary packet of up to64 records—entity first16, original adjacency first16, ungated body-topic first16, then remaining records from those sources, deduplicated—still had six. This does not rule out gains from partially recovered evidence, but it does not support purchasing another answer pilot on a completion-improvement rationale. No model calls were made for either wider-packet preflight. The wider runner is explicitly disabled in its versioned protocol.

The remaining technical problem is finding the specific missing facts and joining them with the correct entity/event, rather than simply returning more of the current ranking. This run does not claim that problem is solved. No additional experiment is scheduled automatically.
