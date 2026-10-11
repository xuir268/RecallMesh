"""Fixed20-question follow-up: new entity answers vs cached adjacency answers."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import small_answer_eval as runner
from longitudinal import interval
ROOT=Path(__file__).resolve().parents[1]
runner.OUT=ROOT/'benchmarks/results/entity-answers'
runner.PROTOCOL=ROOT/'benchmarks/protocols/entity-answers-v1.json'

def prepare():
    ids=json.loads((ROOT/'benchmarks/protocols/medium-answer-v1.json').read_text())['case_ids']
    traces={r['id']:r for r in json.loads((ROOT/'benchmarks/results/entity-aware/traces.json').read_text())}
    data={s['sample_id']:s for s in json.loads((ROOT/'benchmarks/data/locomo10.json').read_text())}
    jobs=[]
    for case in ids:
        cid,qi=case.split(':q');sample=data[cid];qa=sample['qa'][int(qi)];records={}
        for key,session in sample['conversation'].items():
            import re
            if not re.fullmatch(r'session_\d+',key):continue
            for turn in session:
                text='['+sample['conversation'].get(key+'_date_time','')+'] '+turn['speaker']+': '+turn['text']
                if turn.get('blip_caption'):text+=' [Image caption: '+turn['blip_caption']+']'
                records[turn['dia_id']]={'id':turn['dia_id'],'text':text}
        old=json.loads((ROOT/'benchmarks/results/medium-answer/jobs.json').read_text())
        baseline=next(j for j in old if j['id']==case+'-adjacency')
        assert traces[case]['packets']['adjacency']==[e['id'] for e in baseline['evidence']]
        evidence=[records[k] for k in traces[case]['packets']['entity_topic_plus_adjacency']]
        job={**baseline,'id':case+'-entity','arm':'entity','evidence':evidence}
        job['input_sha256']=hashlib.sha256(json.dumps({'question':job['question'],'evidence':evidence},sort_keys=True).encode()).hexdigest();jobs.append(job)
    protocol={'date':'2026-10-11','case_ids':ids,'arms':['entity'],'models':{'codex':'gpt-6-astra','claude':'claude-sonnet-4-6'},'calls':40,'workers':4,'policy':'new first-attempt entity answers on the same20 medium-pilot questions; cached original adjacency comparator, no retries/tools/fallback or outcome selection. Gold and arm names withheld from models.','metric':'unchanged answer token F1, paired deltas and cluster95; abstentions. Diagnostic only, no product promotion.','limits':'cached comparator generated earlier at two workers vs four now; execution date/model nondeterminism confound. All20 questions are previously exposed and are included in the new100-question retrieval evaluation; not a heldout answer test.'}
    if runner.PROTOCOL.exists():assert json.loads(runner.PROTOCOL.read_text())==protocol
    runner.PROTOCOL.write_text(json.dumps(protocol,indent=2)+'\n');runner.OUT.mkdir(parents=True,exist_ok=True)
    (runner.OUT/'jobs.json').write_text(json.dumps(jobs,indent=2))
    (ROOT/'benchmarks/protocols/entity-answer-inputs-v1.json').write_text(json.dumps({j['id']:j['input_sha256'] for j in jobs},indent=2)+'\n')

def score():
    protocol=json.loads(runner.PROTOCOL.read_text());summary={};public={}
    for backend in ['codex','claude']:
        rows=[json.loads(p.read_text()) for p in (runner.OUT/backend).glob('*.json')];assert len(rows)==20
        paired=[];public[backend]=[]
        for row in sorted(rows,key=lambda r:r['id']):
            baseline=json.loads((ROOT/'benchmarks/results/medium-answer'/backend/(row['case']+'-adjacency.json')).read_text())
            paired.append({'conversation':row['case'].split(':')[0],'entity':row['answer_f1'],'adjacency':baseline['answer_f1']})
            public[backend].append({k:v for k,v in row.items() if k not in ['completion','raw_output','error']} | {'adjacency_f1':baseline['answer_f1'],'adjacency_abstained':baseline['abstained'],'delta_f1':row['answer_f1']-baseline['answer_f1']})
        summary[backend]={'n':20,'adjacency_f1':float(np.mean([p['adjacency'] for p in paired])),'entity_f1':float(np.mean([p['entity'] for p in paired])),'delta':float(np.mean([p['entity']-p['adjacency'] for p in paired])),'cluster95':interval(paired,'entity','adjacency'),'wins':sum(p['entity']>p['adjacency'] for p in paired),'ties':sum(p['entity']==p['adjacency'] for p in paired),'losses':sum(p['entity']<p['adjacency'] for p in paired),'entity_abstentions':sum(r['abstained'] for r in rows),'successful_calls':sum(r['status']=='success' for r in rows)}
    result={'protocol':protocol,'summary':summary,'per_call':public}
    (ROOT/'benchmarks/protocols/entity-answer-results-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','score']);p.add_argument('--backend',choices=['codex','claude']);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':runner.run(a.backend,workers=4)
    else:score()
