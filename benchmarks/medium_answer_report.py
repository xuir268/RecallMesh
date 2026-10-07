"""Write a reproducible report without redistributing benchmark dialogue."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'benchmarks/protocols/medium-answer-results-v1.json').read_text())
lines=['# Medium Astra/Claude answer evaluation','',
'October 7, 2026. **20 fixed questions, three evidence conditions, two models: 120 first-attempt answers.** Five questions per category (multi-hop, temporal, open-domain inference, single-hop); the six-question pilot was excluded. All seven eligible conversations are represented. The primary measure is token F1, not an independently adjudicated correctness rate.','',
'| Model | Condition | Mean F1 | Abstentions |', '|---|---|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude Sonnet 4.6')]:
    s=r['summary'][backend]
    for arm in ['adjacency','hebbian_50','oracle']:
        lines.append(f"| {label} | {arm} | {s[arm]['mean_answer_f1']:.2%} | {s[arm]['abstentions']}/20 |")
lines+=['','## Paired comparison','']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    p=r['paired'][backend];lo,hi=p['cluster95']
    lines.append(f"- {label}: Hebbian minus adjacency **{100*p['hebbian_minus_adjacency_f1']:+.2f} percentage points**; paired conversation-cluster bootstrap 95% interval **[{100*lo:+.2f}, {100*hi:+.2f}] points**. Oracle minus adjacency: {100*p['oracle_minus_adjacency_f1']:+.2f} points.")
lines+=['','## Paired signs and distributions — added October 8','',
'These diagnostics use the original saved scores with no new model calls. Wins and losses refer to token-F1 differences, not independently adjudicated semantic correctness. They are post-hoc; the registered decision rule remains unchanged.', '',
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
lines+=['', 'Seven conversations give little power; the conversation tests are also exploratory and unadjusted. Neither level establishes a significant sign advantage here. A majority of positive question signs does not guarantee a positive mean: small wording gains can be outweighed by one large loss.', '',
'Distribution of paired deltas in **F1 percentage points** (candidate minus adjacency):', '',
'| Model / candidate | Minimum | 25th percentile | Median | 75th percentile | Maximum |', '|---|---:|---:|---:|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    for arm in ['hebbian_50','oracle']:
        d=r['paired_distribution']['comparisons'][backend][arm]['delta_distribution']
        lines.append(f"| {label} / {arm} | "+' | '.join(f"{100*d[k]:+.2f}" for k in ['minimum','q25','median','q75','maximum'])+' |')
lines+=['', '[Every paired per-question delta and abstention flag (CSV)](assets/medium-answer-paired-deltas.csv) is published; identifiers and metrics contain no raw dialogue. The JSON includes the same per-question records and transition breakdown.', '',
'## Abstention transitions','',
'| Model / candidate | Abstain → answer | Answer → abstain | Answer → answer | Abstain → abstain |', '|---|---:|---:|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    for arm in ['hebbian_50','oracle']:
        t=r['paired_distribution']['comparisons'][backend][arm]['abstention_transitions']
        lines.append(f"| {label} / {arm} | "+' | '.join(str(t[k]['n']) for k in ['abstain_to_answer','answer_to_abstain','answer_to_answer','abstain_to_abstain'])+' |')
lines+=['', 'For Astra, oracle converted five adjacency abstentions into answers; all five increased F1. Those conversions contribute **+12.23 points** of the **+13.35-point** mean gain; changes where both conditions answered contribute +1.11 points. For Claude, oracle converted four abstentions: three increased F1 and one remained at zero. Conversions contribute **+8.04 points**, partly offset by **−1.30 points** where both conditions answered, leaving +6.74 points overall. No oracle answer reverted to abstention.', '',
'This is an arithmetic decomposition of observed F1 gain, not a causal mediation estimate or proof that every new answer was correct. Hebbian evidence also converted two abstentions for each model, but introduced one new abstention for Astra and three for Claude; these larger losses outweighed its smaller gains.', '']
lines+=['', 'The preregistered positive-signal rule required at least +5 F1 points and a positive confidence lower bound for both models. '+('It passed, but remains a diagnostic answer-pilot signal and does not reverse the separate longitudinal stopping rule.' if r['positive_both_models'] else '**It did not pass. A reliable answer-quality advantage remains unestablished.** Adjacency remains the recommended profile.'),'',
'## Evidence and reliability','', '| Model / condition | Complete required evidence | Abstentions | Successful calls | Invalid citation IDs |', '|---|---:|---:|---:|---:|']
for backend,label in [('codex','Astra'),('claude','Claude')]:
    for arm in ['adjacency','hebbian_50','oracle']:
        s=r['summary'][backend][arm]
        lines.append(f"| {label} / {arm} | {s['complete_evidence']}/20 | {s['abstentions']}/20 | {s['success']}/20 | {s['invalid_citations']}/20 |")
lines+=['',
'Complete evidence means all upstream annotated source IDs occur in the packet. It does not guarantee those sources are sufficient or that a generated claim is supported. Valid citation IDs also do not establish entailment. Required-source recall and per-category F1 are included in the versioned results. Abstention scores zero against the benchmark reference, even when the supplied evidence genuinely cannot resolve the question.', '',
'## Locked design','',
'Protocol and code were committed before generation (`afd9eac`); all 60 evidence-packet fingerprints were committed before model calls (`99eec1c`). Selection was category-stratified seeded sampling from the existing 100-question cohort, without choosing graph gains or losses. The cohort has previous retrieval exposure and is not a fresh held-out corpus. All conditions use the same 16-record budget and question prompt, fresh sessions, medium effort, no tools, no retries and no model fallback. Gold answers and arm names never enter provider inputs. No answer or evaluation question trains the graph.', '',
'The retrieval conditions reproduce the frozen adjacency control and session-50 Hebbian graph from [the longitudinal protocol](longitudinal.md). Oracle puts all annotated original sources first, then fills to16 with Hebbian distractors and chronological records. This gives oracle a source-position advantage; it is a diagnostic ceiling, not a fair competing retriever. Record count is matched; token count is not. No no-memory or BM25 answer arm, repeated generation, or external judge was added.', '',
'## Interpretation limits','',
'Token F1 penalizes verbose paraphrases and numeric spelling: “3 years” versus “three years” can receive a partial score despite agreement. List recall is also sensitive to separators. The existing scoring rules remained fixed after generation. Treat F1 as an overlap measure, not a percentage of semantically correct answers. This medium pilot has no comprehensive independent semantic adjudication.', '',
'Some open-domain reference answers require knowledge not supplied by their annotated sources. For example, the reference names a specific charity while the cited conversation only mentions sports sponsors and an intention to give back. Another reference recommends coaching although the sources discuss several other future activities. A suspected-health reference infers a condition from a vague comment. These limitations were found after generation and flagged without removing questions or changing primary scores. Oracle failures cannot automatically be attributed to the models.', '',
f"Claude reported ${r['runtime']['claude']['reported_cost_usd']:.3f} for its 60 calls; Astra cost is unavailable. Sum of individual call latency including CLI startup was {r['runtime']['codex']['sum_call_seconds']:.1f} seconds for Astra and {r['runtime']['claude']['sum_call_seconds']:.1f} seconds for Claude. Two calls ran concurrently per backend, so these sums are not wall-clock duration.", '',
'With 20 questions and seven conversation clusters, uncertainty remains large. Category balancing means the headline mean does not estimate the original corpus category prevalence. One completion per condition/model does not measure generation variance. This evaluates context-conditioned answering through the CLI adapters, not an autonomous tool-using agent or the full managed-service throughput. Within-model comparisons are the relevant controls because provider harnesses differ. Astra metadata repeats the requested identifier rather than independently attesting a server-side model version.', '',
'[Protocol](../benchmarks/protocols/medium-answer-v1.json), [input fingerprints](../benchmarks/protocols/medium-answer-inputs-v1.json), [results and per-call metrics](../benchmarks/protocols/medium-answer-results-v1.json). Raw questions, dialogue packets and completions stay under ignored `benchmarks/results/medium-answer/`. Upstream dataset licensing and research attribution remain in the project notices.', '',
'```sh', '.venv/bin/python benchmarks/medium_answer_eval.py prepare', '.venv/bin/python benchmarks/medium_answer_eval.py run --backend codex', '.venv/bin/python benchmarks/medium_answer_eval.py run --backend claude', '.venv/bin/python benchmarks/medium_answer_eval.py score', '.venv/bin/python benchmarks/paired_answer_analysis.py', '.venv/bin/python benchmarks/medium_answer_report.py', '```', '',
'Repeated run commands use cached first attempts with matching input/model fingerprints. A new generation consumes account quota and will not reproduce exact answers. The existing authenticated adapters request `gpt-6-astra` and `claude-sonnet-4-6`, using the previously verified [official Codex structured-output interface](https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex).', '']
(ROOT/'docs/medium-answer-evaluation.md').write_text('\n'.join(lines))
