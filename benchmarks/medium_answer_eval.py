"""20 category-stratified questions, 120 first-attempt model answers."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import small_answer_eval as runner
from longitudinal import interval
from paired_answer_analysis import analyze

ROOT=Path(__file__).resolve().parents[1]
runner.OUT=ROOT/'benchmarks/results/medium-answer'
runner.PROTOCOL=ROOT/'benchmarks/protocols/medium-answer-v1.json'
RESULTS=ROOT/'benchmarks/protocols/medium-answer-results-v1.json'
SEED=20261009

def prepare():
    source=ROOT/'benchmarks/data/locomo10.json'
    samples={s['sample_id']:s for s in json.loads(source.read_text())}
    membership=json.loads((ROOT/'benchmarks/protocols/natural-reference-qa-v1.json').read_text())['membership']
    excluded=set(json.loads((ROOT/'benchmarks/protocols/small-answer-v1.json').read_text())['case_ids'])
    rng=random.Random(SEED);chosen=[]
    for category in [1,2,3,4]:
        pool=sorted([c for c in membership if c['id'] not in excluded and samples[c['conversation']]['qa'][c['question_index']]['category']==category],key=lambda c:c['id'])
        chosen.extend(rng.sample(pool,5))
    protocol={'seed':SEED,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'case_ids':[c['id'] for c in chosen],'arms':['adjacency','hebbian_50','oracle'],'models':{'codex':'gpt-6-astra','claude':'claude-sonnet-4-6'},'calls':120,'budget':16,'effort':'medium','selection':'five seeded random questions per category 1/2/3/4 from fixed 100-question cohort, six small-pilot questions excluded; not selected for retrieval outcomes','oracle':'all original required sources first, fill to16 with Hebbian distractors then chronological records; no gold answer text in prompt','scoring':'primary upstream-style token F1, paired differences, 3000 conversation-cluster bootstrap draws. Secondary exact F1=1, complete source coverage, required citations, invalid IDs, abstentions. Any manual audit is secondary and disclosed separately.','decision':'Diagnostic pilot only, no tuning or reversal of longitudinal stopping rule. Positive answer-quality signal requires Hebbian minus adjacency mean F1 >= .05 and paired cluster95 lower bound >0 for both models. Otherwise improvement remains unestablished.','policy':'first attempts including failures, no retries, no tools, fresh sessions; locked settings match longitudinal-v1; holdout questions/answers do not train graph','scope':'medium pilot, previously exposed narrow speaker-reference cohort; stratified scores are not population prevalence weighted; one completion per condition/model; token count not matched; oracle can still contain ambiguous or insufficient annotations'}
    runner.prepare(chosen,protocol,oracle=True)
    jobs=json.loads((runner.OUT/'jobs.json').read_text())
    assert len(jobs)==60
    for j in jobs:
        assert len(j['evidence'])==16
        if j['arm']=='oracle':assert set(j['required'])<={e['id'] for e in j['evidence']}
    hashes={j['id']:j['input_sha256'] for j in jobs}
    (ROOT/'benchmarks/protocols/medium-answer-inputs-v1.json').write_text(json.dumps(hashes,indent=2)+'\n')

def score():
    locked=json.loads((ROOT/'benchmarks/protocols/medium-answer-inputs-v1.json').read_text())
    protocol=json.loads(runner.PROTOCOL.read_text())
    for backend in ['codex','claude']:
        rows=[json.loads(p.read_text()) for p in (runner.OUT/backend).glob('*.json')]
        assert len(rows)==len(locked)==60
        assert {r['id'] for r in rows}==set(locked)
        for row in rows:
            assert row['input_sha256']==locked[row['id']]
            assert row['requested_model']==protocol['models'][backend]
    runner.score()
    result=json.loads(RESULTS.read_text())
    paired={}
    for backend,rows in result['per_call'].items():
        cases={}
        for r in rows:
            c=cases.setdefault(r['case'],{'conversation':r['case'].split(':')[0]})
            c[r['arm']]=r['answer_f1']
        assert len(rows)==60 and len(cases)==20
        values=list(cases.values())
        mean=float(np.mean([r['hebbian_50']-r['adjacency'] for r in values]));ci=interval(values,'hebbian_50','adjacency')
        paired[backend]={'hebbian_minus_adjacency_f1':mean,'cluster95':ci,'positive_signal':mean>=.05 and ci[0]>0,'oracle_minus_adjacency_f1':float(np.mean([r['oracle']-r['adjacency'] for r in values]))}
    jobs={j['id']:j for j in json.loads((runner.OUT/'jobs.json').read_text())}
    for backend,rows in result['per_call'].items():
        for row in rows:
            job=jobs[row['id']]
            row['category']=job['category']
            row['evidence_recall']=len(set(job['required']) & {e['id'] for e in job['evidence']})/len(set(job['required']))
        for arm,summary in result['summary'][backend].items():
            selected=[r for r in rows if r['arm']==arm]
            summary['mean_evidence_recall']=float(np.mean([r['evidence_recall'] for r in selected]))
            summary['by_category']={str(c):{'n':sum(r['category']==c for r in selected),'mean_answer_f1':float(np.mean([r['answer_f1'] for r in selected if r['category']==c]))} for c in [1,2,3,4]}
    result['runtime']={}
    for backend in ['codex','claude']:
        originals=[json.loads(p.read_text()) for p in (runner.OUT/backend).glob('*.json')]
        result['runtime'][backend]={'calls':len(originals),'sum_call_seconds':sum(r['completion'].get('seconds',0) for r in originals),'reported_cost_usd':sum(r['completion'].get('estimated_cost_usd') or 0 for r in originals) if backend=='claude' else None}
    result['paired']=paired;result['positive_both_models']=all(r['positive_signal'] for r in paired.values())
    result['paired_distribution']=analyze(result)
    RESULTS.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(paired,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','score']);p.add_argument('--backend',choices=['codex','claude']);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':
        if not a.backend:p.error('backend required')
        runner.run(a.backend)
    else:score()
