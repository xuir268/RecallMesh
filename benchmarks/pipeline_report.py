"""Summarize raw first attempts; provider unavailability is not answer quality."""
import json
from pathlib import Path
import statistics as stats
OUT=Path('benchmarks/results/pipeline')

def unavailable(r):
    return r.get('status')=='error' and any(x in r.get('error','').lower() for x in ['out of extra usage','usage limit','rate limit'])

def usage(r):
    result=r.get('result',{})
    stages=result.get('stages') or [{'stage':'answer',**result.get('response',{})}]
    inputs=outputs=calls=0
    for s in stages:
        if s['stage'] not in {'answer','select','verify'}:continue
        calls+=1;u=s.get('metadata',s).get('usage',{})
        inputs+=u.get('input_tokens',u.get('prompt_tokens',0))+u.get('cache_read_input_tokens',0)+u.get('cache_creation_input_tokens',0)
        outputs+=u.get('output_tokens',u.get('completion_tokens',0))
    return inputs,outputs,calls

summary={}
for folder in OUT.iterdir():
    if not folder.is_dir() or folder.name.endswith('-oracle'):continue
    rows=[json.loads(p.read_text()) for p in folder.glob('*.json')]
    if not rows:continue
    group={}
    for suite in ['generated','natural']:
        for arm in ['single','wide','bounded']:
            selected=[r for r in rows if r.get('suite')==suite and r.get('arm')==arm]
            if not selected:continue
            available=[r for r in selected if not unavailable(r)]
            good=[r for r in available if r['status']=='success']
            group[suite+'-'+arm]={
                'attempted':len(selected),'unavailable':sum(unavailable(r) for r in selected),
                'completed':len(good),'format_or_other_errors':len(available)-len(good),
                'mean_score_available':stats.mean(r['score'] for r in available) if available else None,
                'complete_required_evidence':sum(r.get('complete_evidence',False) for r in available),
                'median_seconds_success':stats.median(r['seconds'] for r in good) if good else None,
                'mean_input_tokens_all_calls':stats.mean(usage(r)[0] for r in good) if good else None,
                'mean_model_calls':stats.mean(usage(r)[2] for r in good) if good else None,
                'mean_final_evidence_bytes':stats.mean(r['result']['metrics']['final_evidence_bytes'] for r in good) if good else None,
                'second_round_queries':sum(any(s.get('stage')=='retrieve' and s.get('round',0)>0 for s in r['result'].get('stages',[])) for r in good),
                'support_rejections':sum(r['result'].get('support_rejected',False) for r in good)}
    summary[folder.name]=group
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

oracles={}
for folder in OUT.glob('*-oracle'):
    rows=[json.loads(p.read_text()) for p in folder.glob('*.json')]
    if rows:
        oracles[folder.name]={'n':len(rows),'correct':sum(r['score']==1 for r in rows),
                             'errors':sum(r['status']=='error' for r in rows)}
(OUT/'oracle-summary.json').write_text(json.dumps(oracles,indent=2))
