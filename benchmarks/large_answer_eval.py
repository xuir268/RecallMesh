"""All 100 existing-cohort questions; reuse exact matching first attempts."""
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
runner.OUT=ROOT/'benchmarks/results/large-answer'
runner.PROTOCOL=ROOT/'benchmarks/protocols/large-answer-v1.json'
RESULTS=ROOT/'benchmarks/protocols/large-answer-results-v1.json'
SEED=20261008

def prepare():
    source=ROOT/'benchmarks/data/locomo10.json'
    chosen=sorted(json.loads((ROOT/'benchmarks/protocols/natural-reference-qa-v1.json').read_text())['membership'],key=lambda c:c['id'])
    protocol={'registration_date':'2026-10-08','seed':SEED,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'case_ids':[c['id'] for c in chosen],'arms':['adjacency','hebbian_50','oracle'],'models':{'codex':'gpt-6-astra','claude':'claude-sonnet-4-6'},'calls':600,'budget':16,'effort':'medium','workers_per_backend':4,'selection':'all 100 existing speaker-coreference cohort questions; includes previously answer-tested subsets; not fresh holdout or independent replication','cache':'reuse earliest saved first attempt from medium/small pilots only if job input and requested model match; include errors; no resampling. Expected144 reused,456 new model calls. Frozen cache manifest before generation.','oracle':'all annotated original sources first, fill to16 with Hebbian distractors then chronological records; no gold answer text in prompt','scoring':'fixed token F1 primary; paired per-question distributions and wins/ties/losses, exact sign tests with ties excluded and Holm over four comparisons; conversation-cluster bootstrap3000 and conversation-sign sensitivity; abstention transition contribution. Category breakdown and separate 74-question not-yet-answer-tested subset secondary.','decision':'diagnostic expansion, no tuning or reversal of longitudinal rule; positive signal requires Hebbian minus adjacency meanF1>=.05 and paired cluster95 lower bound>0 for both models','policy':'first attempts including failures, no retries/tools/model fallback, fresh sessions, locked longitudinal retrieval and replay settings. Evaluation questions/answers never train graph. Four concurrent new requests per backend; reused requests previously ran at two.','scope':'seven conversations, narrow previously evaluated cohort, one completion per question/arm/model; cache reuse is not new evidence on prior cases. Annotation and token-F1 limitations remain; oracle does not guarantee source sufficiency. Different execution dates and model nondeterminism limit interpretation.'}
    runner.prepare(chosen,protocol,oracle=True)
    jobs=json.loads((runner.OUT/'jobs.json').read_text())
    assert len(jobs)==300
    for j in jobs:
        assert len(j['evidence'])==16
        if j['arm']=='oracle':assert set(j['required'])<={e['id'] for e in j['evidence']}
    hashes={j['id']:j['input_sha256'] for j in jobs}
    (ROOT/'benchmarks/protocols/large-answer-inputs-v1.json').write_text(json.dumps(hashes,indent=2)+'\n')
    manifest={}
    for backend in ['codex','claude']:
        folder=runner.OUT/backend;folder.mkdir(exist_ok=True)
        manifest[backend]={}
        for job in jobs:
            dest=folder/(job['id']+'.json')
            for previous in ['medium-answer','small-answer']:
                path=ROOT/'benchmarks/results'/previous/backend/dest.name
                if not path.exists():continue
                row=json.loads(path.read_text())
                if row['input_sha256']!=job['input_sha256'] or row['requested_model']!=protocol['models'][backend]:raise ValueError('Previous input/model mismatch')
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:raise ValueError('Cache changed')
                dest.write_bytes(path.read_bytes())
                manifest[backend][job['id']]={'source':previous,'sha256':digest}
                break
    (ROOT/'benchmarks/protocols/large-answer-cache-v1.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Reused',sum(len(v) for v in manifest.values()),'of600 answers',flush=True)


def score():
    locked=json.loads((ROOT/'benchmarks/protocols/large-answer-inputs-v1.json').read_text())
    protocol=json.loads(runner.PROTOCOL.read_text())
    for backend in ['codex','claude']:
        rows=[json.loads(p.read_text()) for p in (runner.OUT/backend).glob('*.json')]
        assert len(rows)==len(locked)==300
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
        assert len(rows)==300 and len(cases)==100
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
    cache=json.loads((ROOT/'benchmarks/protocols/large-answer-cache-v1.json').read_text())
    result['cache_reuse']={b:len(v) for b,v in cache.items()}
    RESULTS.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(paired,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','score']);p.add_argument('--backend',choices=['codex','claude']);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':
        if not a.backend:p.error('backend required')
        runner.run(a.backend,workers=4)
    else:score()
