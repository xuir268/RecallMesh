"""Write a reproducible report without redistributing benchmark dialogue."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'benchmarks/protocols/large-answer-results-v1.json').read_text())
lines=['# Full-cohort Astra/Claude answer evaluation','',
'October 8, 2026. **100 existing-cohort questions, three evidence conditions, two models: 600 recorded first attempts (595 successful outputs and five Astra timeouts).** This includes the 20-question and six-question pilots. All seven eligible conversations are represented. Exactly 144 matching prior answers are reused, with 456 new model calls; this is an expansion, not an independent replication or fresh sample. The primary measure is token F1, not an independently adjudicated correctness rate.','',
'| Model | Condition | Mean F1 | Abstentions |', '|---|---|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude Sonnet 4.6')]:
    s=r['summary'][backend]
    for arm in ['adjacency','hebbian_50','oracle']:
        lines.append(f"| {label} | {arm} | {s[arm]['mean_answer_f1']:.2%} | {s[arm]['abstentions']}/100 |")
lines+=['','## Paired comparison','']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    p=r['paired'][backend];lo,hi=p['cluster95']
    lines.append(f"- {label}: Hebbian minus adjacency **{100*p['hebbian_minus_adjacency_f1']:+.2f} percentage points**; paired conversation-cluster bootstrap 95% interval **[{100*lo:+.2f}, {100*hi:+.2f}] points**. Oracle minus adjacency: {100*p['oracle_minus_adjacency_f1']:+.2f} points, cluster95 [{100*p['oracle_cluster95'][0]:+.2f}, {100*p['oracle_cluster95'][1]:+.2f}].")
lines+=['','## Paired signs and distributions — October 8','',
'These paired diagnostics use the saved first-attempt scores; reused and newly generated cases are identified in the cache manifest. Wins and losses refer to token-F1 differences, not independently adjudicated semantic correctness. They are post-hoc; the registered decision rule remains unchanged.', '',
'| Model / candidate vs adjacency | Wins | Ties | Losses | Exact sign p | Holm-adjusted p |', '|---|---:|---:|---:|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    for arm in ['hebbian_50','oracle']:
        d=r['paired_distribution']['comparisons'][backend][arm]
        lines.append(f"| {label} / {arm} | {d['wins']} | {d['ties']} | {d['losses']} | {d['exact_sign_p_two_sided']:.4f} | {d['holm_adjusted_question_sign_p']:.4f} |")
lines+=['', 'Two-sided exact binomial sign tests exclude ties (absolute F1 difference ≤1e-12). The four exploratory question-level tests are Holm-adjusted together. Their nominal p values assume independent signs; these questions share seven conversations, so question-level p values are not cluster-robust. As a separate sensitivity analysis, signs of equal-weight conversation mean deltas give:', '',
'| Model / candidate | Positive / tied / negative conversations | Conversation-level exact p |', '|---|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    for arm in ['hebbian_50','oracle']:
        d=r['paired_distribution']['comparisons'][backend][arm]['conversation_sign']
        lines.append(f"| {label} / {arm} | {d['wins']} / {d['ties']} / {d['losses']} | {d['exact_sign_p_two_sided']:.4f} |")
lines+=['', 'Seven conversations give little power; the conversation tests are also exploratory and unadjusted. Question signs assume independence across correlated questions; conversation-level results and the original cluster intervals are the relevant caution when interpreting small p values. A majority of positive question signs does not guarantee a positive mean: small wording gains can be outweighed by one large loss.', '',
'Distribution of paired deltas in **F1 percentage points** (candidate minus adjacency):', '',
'| Model / candidate | Minimum | 25th percentile | Median | 75th percentile | Maximum |', '|---|---:|---:|---:|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    for arm in ['hebbian_50','oracle']:
        d=r['paired_distribution']['comparisons'][backend][arm]['delta_distribution']
        lines.append(f"| {label} / {arm} | "+' | '.join(f"{100*d[k]:+.2f}" for k in ['minimum','q25','median','q75','maximum'])+' |')
lines+=['', '[Every paired per-question delta and abstention flag (CSV)](assets/large-answer-paired-deltas.csv) is published; identifiers and metrics contain no raw dialogue. The JSON includes the same per-question records and transition breakdown.', '',
'## Abstention transitions','',
'| Model / candidate | Abstain → answer | Answer → abstain | Answer → answer | Abstain → abstain | Error involved |', '|---|---:|---:|---:|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    for arm in ['hebbian_50','oracle']:
        t=r['paired_distribution']['comparisons'][backend][arm]['abstention_transitions']
        lines.append(f"| {label} / {arm} | "+' | '.join(str(t[k]['n']) for k in ['abstain_to_answer','answer_to_abstain','answer_to_answer','abstain_to_abstain'])+f" | {sum(v['n'] for k,v in t.items() if 'error' in k)} |")
lines.append('')
for backend,label in [('codex','Astra'),('claude','Claude')]:
    d=r['paired_distribution']['comparisons'][backend]['oracle'];t=d['abstention_transitions'];c=t['abstain_to_answer']
    lines.append(f"- {label}: oracle converted {c['n']} adjacency abstentions to answers ({c['wins']} F1 wins, {c['ties']} ties, {c['losses']} losses). These conversions contribute {100*c['contribution_to_overall_mean_delta']:+.2f} points of the {100*d['mean_delta_f1']:+.2f}-point overall gain. Changes where both conditions answered contribute {100*t['answer_to_answer']['contribution_to_overall_mean_delta']:+.2f} points, answer-to-abstention transitions contribute {100*t['answer_to_abstain']['contribution_to_overall_mean_delta']:+.2f} points, and error-involved pairs contribute {100*sum(v['contribution_to_overall_mean_delta'] for k,v in t.items() if 'error' in k):+.2f} points.")
lines+=['','## Call failures and successful-pair sensitivity','',
'Failed first attempts remain zero scores in the primary analysis. They are not counted as abstentions or successful answers. No timeout retries were made. Error-involved pairs are separated in the transition table.', '',
'| Model | Failed calls /300 | Comparison | Successful pairs | Mean delta (F1 points) |', '|---|---:|---|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    for arm in ['hebbian_50','oracle']:
        d=r['successful_pair_sensitivity'][backend][arm]
        lines.append(f"| {label} | {len(r['failures'][backend])} | {arm} − adjacency | {d['n']} | {100*d['mean_delta_f1']:+.2f} |")
lines.append('')
lines.append('This is a secondary complete-pairs sensitivity analysis; excluding failures can introduce selection bias and does not replace the primary result. Sanitized failure identifiers and reasons are published in the JSON.')
lines+=['', 'This arithmetic decomposition does not establish causal mediation or semantic correctness. Signs count token-F1 improvements, including small wording changes.','',
'## Questions outside the two recent answer pilots','',
'This secondary slice contains 74 questions; it is not independently sampled and may have appeared in earlier project studies. All three conditions on these questions are newly generated in this expansion.', '',
'| Model | Adjacency F1 | Hebbian F1 | Oracle F1 |', '|---|---:|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    s=r['outside_recent_pilots']['summary'][backend]
    lines.append(f"| {label} | {s['adjacency']['mean_answer_f1']:.2%} | {s['hebbian_50']['mean_answer_f1']:.2%} | {s['oracle']['mean_answer_f1']:.2%} |")
lines.append('')
lines+=['', 'The preregistered positive-signal rule required at least +5 F1 points and a positive confidence lower bound for both models. '+('It passed, but remains a diagnostic answer-pilot signal and does not reverse the separate longitudinal stopping rule.' if r['positive_both_models'] else '**It did not pass. A reliable answer-quality advantage remains unestablished.** Adjacency remains the recommended profile.'),'',
'## Evidence and reliability','', '| Model / condition | Complete required evidence | Abstentions | Successful calls | Invalid citation IDs |', '|---|---:|---:|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    for arm in ['adjacency','hebbian_50','oracle']:
        s=r['summary'][backend][arm]
        lines.append(f"| {label} / {arm} | {s['complete_evidence']}/100 | {s['abstentions']}/100 | {s['success']}/100 | {s['invalid_citations']}/100 |")
lines+=['',
'Complete evidence means all upstream annotated source IDs occur in the packet. It does not guarantee those sources are sufficient or that a generated claim is supported. Valid citation IDs also do not establish entailment. Required-source recall and per-category F1 are included in the versioned results. Abstention scores zero against the benchmark reference, even when the supplied evidence genuinely cannot resolve the question.', '',
'## Locked design','',
'Protocol and code were committed before generation (`20fab8d`, clarified `26e013b`); all 300 evidence-packet fingerprints and the 144-answer cache manifest were committed before new model calls (`4009621`). Selection includes every member of the existing 100-question cohort, without selecting for graph gains or losses. Earlier pilot results were known; this expansion is diagnostic, not a new confirmatory study. The cohort has previous retrieval exposure and is not a fresh held-out corpus. Four new requests ran concurrently per backend; cached pilot requests previously ran at two. Six prior-pilot oracle answers are newly generated while their two retrieval-arm answers are reused, creating a timing difference for those cases. Cache reuse requires exact input and requested-model matches and preserves the earliest stored attempt, including errors. All conditions use the same 16-record budget and question prompt, fresh sessions, medium effort, no tools, no retries and no model fallback. Gold answers and arm names never enter provider inputs. No answer or evaluation question trains the graph.', '',
'The retrieval conditions reproduce the frozen adjacency control and session-50 Hebbian graph from [the longitudinal protocol](longitudinal.md). Oracle puts all annotated original sources first, then fills to16 with Hebbian distractors and chronological records. This gives oracle a source-position advantage; it is a diagnostic ceiling, not a fair competing retriever. Record count is matched; token count is not. No no-memory or BM25 answer arm, repeated generation, or external judge was added.', '',
'## Interpretation limits','',
'Token F1 penalizes verbose paraphrases and numeric spelling: “3 years” versus “three years” can receive a partial score despite agreement. List recall is also sensitive to separators. The existing scoring rules remained fixed after generation. Treat F1 as an overlap measure, not a percentage of semantically correct answers. This evaluation has no comprehensive independent semantic adjudication.', '',
'Some open-domain reference answers require knowledge not supplied by their annotated sources. For example, the reference names a specific charity while the cited conversation only mentions sports sponsors and an intention to give back. Another reference recommends coaching although the sources discuss several other future activities. A suspected-health reference infers a condition from a vague comment. These limitations were found after generation and flagged without removing questions or changing primary scores. Oracle failures cannot automatically be attributed to the models.', '',
f"Claude reported ${r['runtime']['claude']['reported_cost_usd']:.3f} across 300 stored calls (${r['runtime']['claude']['new_reported_cost_usd']:.3f} for new calls); Astra cost is unavailable. Sum of available successful-call latency including CLI startup was {r['runtime']['codex']['sum_call_seconds']:.1f} seconds for Astra and {r['runtime']['claude']['sum_call_seconds']:.1f} seconds for Claude. New calls ran four at a time per backend, while reused calls ran previously. These sums include historical call time, exclude unavailable timeout latency, and are not current wall-clock duration.", '',
'With 100 questions but still only seven conversation clusters, uncertainty remains large. The original cohort is selected for speaker-reference structure; the headline mean does not estimate general conversational question prevalence. Category counts are not balanced in the full cohort. One completion per condition/model does not measure generation variance. This evaluates context-conditioned answering through the CLI adapters, not an autonomous tool-using agent or the full managed-service throughput. Within-model comparisons are the relevant controls because provider harnesses differ. Astra metadata repeats the requested identifier rather than independently attesting a server-side model version.', '',
'[Cache provenance](../benchmarks/protocols/large-answer-cache-v1.json), [Protocol](../benchmarks/protocols/large-answer-v1.json), [input fingerprints](../benchmarks/protocols/large-answer-inputs-v1.json), [results and per-call metrics](../benchmarks/protocols/large-answer-results-v1.json). Raw questions, dialogue packets and completions stay under ignored `benchmarks/results/large-answer/`. Upstream dataset licensing and research attribution remain in the project notices.', '',
'```sh', '.venv/bin/python benchmarks/large_answer_eval.py prepare', '.venv/bin/python benchmarks/large_answer_eval.py run --backend codex', '.venv/bin/python benchmarks/large_answer_eval.py run --backend claude', '.venv/bin/python benchmarks/large_answer_eval.py score',  '.venv/bin/python benchmarks/large_answer_report.py', '```', '',
'Repeated run commands use cached first attempts with matching input/model fingerprints. A new generation consumes account quota and will not reproduce exact answers. The existing authenticated adapters request `gpt-6-astra` and `claude-sonnet-4-6`, using the previously verified [official Codex structured-output interface](https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex).', '']
import csv
with (ROOT/'docs/assets/large-answer-paired-deltas.csv').open('w',newline='') as f:
    fields=['backend','candidate','reference','case','conversation','reference_f1','candidate_f1','delta_f1','reference_abstained','candidate_abstained','reference_status','candidate_status']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for backend,arms in r['paired_distribution']['comparisons'].items():
        for arm,d in arms.items():
            for row in d['per_question']:writer.writerow({'backend':backend,'candidate':arm,'reference':'adjacency',**row})
(ROOT/'docs/large-answer-evaluation.md').write_text('\n'.join(lines))
