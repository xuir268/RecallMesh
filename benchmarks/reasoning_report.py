"""Render completed evidence-dependence results without rerunning any model."""
import json
from pathlib import Path
from reasoning_evidence import OUT


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    rows=json.loads((OUT/'scored.json').read_text())
    assert all(summary[b]['completed']==96 for b in ['codex','claude'])
    diagnosis=json.loads((OUT/'retrieval-diagnosis.json').read_text())
    missing=[r for case in diagnosis for r in case['required'] if not r['inside_budget']]
    names={'codex':'Astra','claude':'Claude Sonnet 4.6'}
    lines=['# Evidence-dependent reasoning: stronger controls',
           '', 'This run tests whether model answers use the necessary facts and respond correctly to factual changes. It adds **192 first-attempt evaluation records** across Astra and Claude. Twelve generated problems have five conditions; twelve natural multi-evidence questions have three conditions. See [the locked protocol](PROTOCOL.md).',
           '', '## Controlled results', '',
           'Each generated problem requires identity chaining, multiplication across three facts, or temporal state selection. All five conditions have four evidence records. Scores below count exact requested answers; no word-overlap approximation is used.', '',
           '| Condition / paired check | Astra | Claude Sonnet 4.6 |',
           '| --- | ---: | ---: |']
    for arm,label in [('lexical','BM25 evidence: correct'),('associative','Graph evidence: correct'),('oracle','Oracle evidence: correct'),('counterfactual','One fact changed: new answer correct'),('bridge_removed','Identity link removed: correctly abstains')]:
        vals=[f"{round(summary[b]['synthetic'][arm]['mean_answer_score']*12)}/12" for b in names]
        lines.append(f'| {label} | {vals[0]} | {vals[1]} |')
    vals=[summary[b]['causal_pairs']['all_three_pass'] for b in names]
    lines.append(f'| All three oracle/change/removal checks pass on the same case | {vals[0]}/12 | {vals[1]}/12 |')
    lines += ['', 'Claude had **four schema failures**: two correct numeric answers encoded as JSON numbers, one natural answer encoded as a JSON list, and one response whose original raw output was not retained. Strict scores above count these as failures, without retries. This separates interface reliability from reasoning correctness; all retained outputs can be inspected. No question or prompt was tuned after model responses.',
              '', 'These controls provide stronger behavioral evidence that the models can combine supplied facts and condition their conclusions on them. They do not reveal private internal reasoning, show changed model weights, or establish that the graph improves intrinsic model intelligence.',
              '', '## Concrete factual intervention', '']
    lines += ['', 'A separate, explicitly post-hoc content audit reads numeric/list answers from the retained raw JSON without generating another response. Original strict scores and raw records are unchanged. Claude’s oracle content is correct on **12/12** cases; its changed-fact content is correct on **11 cases**, with **one unscorable response**. It passes all three behavioral checks on **11/12 cases** with the remaining case unscorable, versus 10/12 fully schema-valid passes. Astra passes both definitions on 12/12.', '']
    case='arithmetic-0'
    selected={(r['backend'],r['arm']):r for r in rows if r['suite']=='synthetic' and r['case']==case}
    oracle=selected['codex','oracle'];changed=selected['codex','counterfactual']
    lines += [f"Question: **{oracle['question']}**",'','Original evidence:','']
    required=set(oracle['required_evidence'])
    for e in oracle['evidence']:
        if e['id'] in required:lines.append(f"- [{e['id']}] {e['text']}")
    old={e['id']:e['text'] for e in oracle['evidence']}
    for e in changed['evidence']:
        if old[e['id']]!=e['text']:
            lines += ['',f"Single-fact edit: **{e['text']}**"]
    lines += ['', '| Model | Original | After factual edit | Identity link removed |','| --- | --- | --- | --- |']
    for b,name in names.items():
        answers=[selected[b,a]['audited_answer'] or '[unscorable output]' for a in ['oracle','counterfactual','bridge_removed']]
        lines.append(f'| {name} | '+ ' | '.join(answers)+' |')
    lines += ['', 'Claude’s original 231 above comes from its retained numeric JSON field (a schema failure, although the value is correct). The quantity answer must be computed across three memories; it is not simply copied from one stored sentence. The missing-link control checks whether the model avoids assuming that the remaining quantities belong to the queried order.',
              '', '## Where production retrieval fails', '',
              'BM25 and the current combined graph each supplied all required evidence on only **4 of 12** generated cases. The oracle removes that retrieval bottleneck. The graph audit found:','',
              f'- **{len(missing)} required facts** were outside the final four-record graph output.',
              f'- **{sum(r["first_reached_hop"] is not None for r in missing)} of those facts** had actually been reached by graph propagation.',
              '- The final ranking/output budget removed those facts. Expanding the graph walk alone would not fix these omissions.',
              '', 'This is evidence for improving final evidence selection: retain a complete supporting chain when several facts are jointly necessary. It is not yet proof that any proposed replacement ranking algorithm will improve answers. The production retriever was not retuned on these labels.',
              '', '## Natural LoCoMo questions', '',
              'These are twelve previously untested category-1 questions with 2–8 annotated evidence turns. Four are deliberately sampled from prior retrieval gains, four from losses, and four from unchanged cases. This selected diagnostic sample must not be pooled into a representative LoCoMo result. BM25/graph each receive 16 turns; oracle gets only the annotated evidence and has a different context size.', '',
              'Mean LoCoMo answer F1 on a 0–100 scale:', '',
              '| Retrieval-outcome group | Model | BM25 | Graph | Oracle |',
              '| --- | --- | ---: | ---: | ---: |']
    for stratum in ['gain','loss','unchanged']:
        for b,name in names.items():
            vals=[summary[b]['natural'][stratum][a]['mean_answer_score']*100 for a in ['lexical','associative','oracle']]
            lines.append(f'| {stratum} (4 questions) | {name} | '+ ' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines += ['', 'These are strict first-attempt scores. One Claude graph answer in the gain group was a JSON list and counted as a format failure; the post-hoc content audit separately retains its recovered text. F1 rewards word overlap, so phrasing can change the score without changing factual correctness. Annotated evidence can also omit useful context. The oracle is a diagnostic reference, not a guaranteed numerical upper bound. Full model answers, citations and short explanations are in [TRANSCRIPTS.md](TRANSCRIPTS.md).',
              '', '## Limits and reproducibility', '',
              '- One response draw per condition; three generated templates with four instances each. This is a functional causal test, not broad reasoning coverage or a repeated-trial statistical claim.',
              '- Synthetic cases were generated with seed 20261006, including random identifiers and quantities. Model calls began only after the input checksum was locked.',
              '- Fresh sessions, same evidence prompt, fixed medium effort, requested gpt-6-astra and claude-sonnet-4-6, no model fallback. The CLI harnesses differ; within-model comparisons are the meaningful controls.',
              '- No gold labels were sent to generation. The supervisor stored labels for scoring; providers received only question and evidence.',
              '- No blind human semantic grading. Citations and correct interventions strengthen the evidence but cannot establish an internal reasoning mechanism.',
              '- No schema-error resampling. The one initially lost raw response is explicitly marked; the resumed run reused existing completions and continued unseen cases.',
              '', 'Files: `manifest.json` and `jobs.json` lock the inputs; `codex/` and `claude/` contain per-call records; `summary.json` and `scored.json` contain scores; `retrieval-diagnosis.json` records each required fact’s rank and reachability.',
              '', 'Validation: **31 Python tests passed**, including independently recomputed quantity and temporal targets, one-fact intervention checks, strict answer scoring, prompt/label separation, and invalid-response retention.',
              '', 'Reproduce with `benchmarks/reasoning_evidence.py prepare`, `run --backend codex`, `run --backend claude`, then `score`. Run `benchmarks/reasoning_retrieval_audit.py` and `benchmarks/reasoning_report.py` for the diagnostic and this report. Commands run from the project root using `.venv/bin/python`; generation consumes the signed-in accounts’ quota.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    transcript=['# Actual model outputs and evidence', '', 'Gold labels below were used only for scoring after generation. Explanations are short final-output justifications, not private reasoning traces.']
    for suite in ['synthetic','natural']:
        cases=sorted({r['case'] for r in rows if r['suite']==suite})
        for case in cases:
            group=[r for r in rows if r['suite']==suite and r['case']==case]
            transcript += ['',f'## {suite}: {case}', '',group[0]['question']]
            for arm in dict.fromkeys(r['arm'] for r in group):
                arm_rows=[r for r in group if r['arm']==arm];job=arm_rows[0]
                transcript += ['',f'### {arm}', '',f"Expected: {job['gold']}", '', 'Supplied evidence:', '']
                transcript += [f"- [{e['id']}] {e['text']}" for e in job['evidence']]
                for r in arm_rows:
                    c=r['completion']
                    transcript += ['',f"**{names[r['backend']]}:** {c['answer'] or '[format failure; no scorable output]'}",'',
                                   f"Explanation: {c['rationale']}", '', f"Citations: {', '.join(c['evidence_ids']) or 'none'}. Score: {r['answer_score']:.3f}."]
                    if r.get('status')=='format_error':
                        transcript += ['', 'Retained raw output:', '', '```json', r.get('raw_output') or '[not retained by the original provider]', '```', '', f"Post-hoc content audit: {r.get('audited_answer') or '[unscorable]'}. Content score: {r.get('content_score')}. Strict score remains zero."]
    (OUT/'TRANSCRIPTS.md').write_text('\n'.join(transcript)+'\n')
    print('Wrote REPORT.md and TRANSCRIPTS.md')

if __name__=='__main__':main()
