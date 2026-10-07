"""Six fixed natural questions, adjacency versus session-50 Hebbian evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.agent import CliProvider, Evidence
from assoc_mem.lexical import LexicalIndex, AssociativeRetriever, stable_topk
from natural_coreference import turns
from agent_answers import answer_f1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'benchmarks/results/small-answer'
PROTOCOL = ROOT/'benchmarks/protocols/small-answer-v1.json'

def prepare(chosen=None, protocol=None, oracle=False):
    membership = json.loads((ROOT/'benchmarks/protocols/natural-reference-qa-v1.json').read_text())['membership']
    if chosen is None:
        chosen = random.Random(20261007).sample(sorted(membership, key=lambda c:c['id']), 6)
    if protocol is None:
        protocol = {'seed':20261007,'case_ids':[c['id'] for c in chosen], 'arms':['adjacency','hebbian_50'], 'models':{'codex':'gpt-6-astra','claude':'claude-sonnet-4-6'}, 'calls':24,'budget':16,'effort':'medium', 'selection':'random six from fixed 100-question cohort; no gain/loss selection', 'scoring':'upstream-style token F1 primary; exact F1=1 and required-source citation coverage secondary. Citation coverage is a provenance proxy, not semantic entailment.', 'policy':'first attempts, no retries, no tools, fresh sessions, fixed prompt, gold labels withheld; same retrieval settings and replay as longitudinal-v1', 'scope':'small previously exposed cohort; no superiority or statistical-significance claim'}
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != protocol:
        raise ValueError('Locked protocol changed')
    PROTOCOL.write_text(json.dumps(protocol,indent=2)+'\n')
    OUT.mkdir(parents=True,exist_ok=True)
    raw = (ROOT/'benchmarks/data/locomo10.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4'
    jobs=[]
    for sample in json.loads(raw):
        cases=[c for c in chosen if c['conversation']==sample['sample_id']]
        if not cases:continue
        sessions=list(turns(sample));n=sum(len(s) for _,s in sessions);capacity=16
        while capacity*.7<n*(n-1)//2:capacity*=2
        graph,adjacent=EdgeStore(capacity),EdgeStore(capacity)
        index=LexicalIndex();records={};queries=[]
        for tick,session in sessions:
            previous=None
            for turn in session:
                scores=index.scores(turn['text']);related=[int(i)+1 for i in stable_topk(scores,3) if scores[i]>0]
                text='['+sample['conversation'].get(f'session_{tick}_date_time','')+'] '+turn['speaker']+': '+turn['text']
                if turn.get('blip_caption'):text+=' [Image caption: '+turn['blip_caption']+']'
                nid=index.add(text);records[nid]={'id':turn['dia_id'],'text':text};queries.append(turn['text'])
                context=set(related)
                if previous:
                    context.add(previous);adjacent.reinforce(np.array([nid],np.uint32),np.array([previous],np.uint32),tick)
                if context:graph.reinforce(np.full(len(context),nid,np.uint32),np.array(sorted(context),np.uint32),tick)
                previous=nid
        base=tick;graph.freeze(base);adjacent.freeze(base)
        retrieval=AssociativeRetriever(graph,index);control=AssociativeRetriever(adjacent,index)
        for session in range(1,51):
            pairs=[]
            for query in queries:
                ids=retrieval.search(query,base+session,16,.4);a,b=np.triu_indices(len(ids),1);pairs.append((ids[a],ids[b]))
            graph.reinforce(np.concatenate([p[0] for p in pairs]),np.concatenate([p[1] for p in pairs]),base+session);graph.freeze(base+session)
            assert graph.stats()['rejected_full']==0
        for c in cases:
            qa=sample['qa'][c['question_index']]
            for arm,r,t in [('adjacency',control,base),('hebbian_50',retrieval,base+50)]:
                evidence=[records[int(i)] for i in r.search(qa['question'],t,16,.4)]
                job={'id':c['id']+'-'+arm,'case':c['id'],'arm':arm,'question':qa['question'],'gold':str(qa['answer']),'category':qa['category'],'required':c['gold'],'evidence':evidence}
                job['input_sha256']=hashlib.sha256(json.dumps({'question':job['question'],'evidence':evidence},sort_keys=True).encode()).hexdigest();jobs.append(job)
            if oracle:
                evidence=oracle_packet(c['gold'], records, evidence)
                job={'id':c['id']+'-oracle','case':c['id'],'arm':'oracle','question':qa['question'],'gold':str(qa['answer']),'category':qa['category'],'required':c['gold'],'evidence':evidence}
                job['input_sha256']=hashlib.sha256(json.dumps({'question':job['question'],'evidence':evidence},sort_keys=True).encode()).hexdigest();jobs.append(job)
        print('prepared',sample['sample_id'],flush=True)
    (OUT/'jobs.json').write_text(json.dumps(jobs,indent=2))

def oracle_packet(required, records, distractors, budget=16):
    by_id={r['id']:r for r in records.values()}
    if len(required)>budget:raise ValueError('Oracle exceeds evidence budget')
    packet=[by_id[k] for k in dict.fromkeys(required)]
    present={r['id'] for r in packet}
    for record in distractors+list(records.values()):
        if len(packet)>=budget:break
        if record['id'] not in present:
            packet.append(record);present.add(record['id'])
    return packet

def run(backend):
    protocol=json.loads(PROTOCOL.read_text());provider=CliProvider(backend,protocol['models'][backend],timeout=120,max_budget_usd=.25)
    folder=OUT/backend;folder.mkdir(exist_ok=True)
    jobs=json.loads((OUT/'jobs.json').read_text());random.Random(20261008).shuffle(jobs)
    def one(job):
        path=folder/(job['id']+'.json')
        if path.exists():
            cached=json.loads(path.read_text())
            if cached['input_sha256']!=job['input_sha256'] or cached['requested_model']!=provider.model:
                raise ValueError('Cached input/model mismatch')
            return job['id']+' cached'
        try:
            completion=asdict(provider.answer(job['question'],[Evidence(**e) for e in job['evidence']]))
            row={'status':'success','completion':completion}
        except Exception as exc:
            row={'status':'error','error':str(exc),'raw_output':getattr(exc,'raw_output',None),'completion':{'answer':'','evidence_ids':[]}}
        row.update({'id':job['id'],'case':job['case'],'arm':job['arm'],'input_sha256':job['input_sha256'],'requested_model':provider.model})
        answer=row['completion']['answer'];cites=set(row['completion']['evidence_ids']);available={e['id'] for e in job['evidence']};required=set(job['required'])
        row.update(answer_f1=answer_f1(answer,job['gold'],job['category']),complete_evidence=required<=available,required_citations=required<=cites,invalid_citations=sorted(cites-available),abstained=answer.lower().strip()=='not enough information')
        path.write_text(json.dumps(row,indent=2));return job['id']+' '+row['status']
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,j) for j in jobs]):print(backend,f.result(),flush=True)

def score():
    summary={}
    for backend in ['codex','claude']:
        rows=[json.loads(p.read_text()) for p in sorted((OUT/backend).glob('*.json'))]
        summary[backend]={}
        for arm in json.loads(PROTOCOL.read_text())['arms']:
            group=[r for r in rows if r['arm']==arm]
            summary[backend][arm]={'n':len(group),'success':sum(r['status']=='success' for r in group),'mean_answer_f1':float(np.mean([r['answer_f1'] for r in group])) if group else None,'f1_equals_one':sum(r['answer_f1']==1 for r in group),'correct_with_required_citations':sum(r['answer_f1']==1 and r['complete_evidence'] and r['required_citations'] and not r['invalid_citations'] for r in group),'complete_evidence':sum(r['complete_evidence'] for r in group),'required_citations':sum(r['required_citations'] for r in group),'invalid_citations':sum(bool(r['invalid_citations']) for r in group),'abstentions':sum(r['abstained'] for r in group)}
    result={'protocol':json.loads(PROTOCOL.read_text()),'summary':summary,'per_call':{b:[json.loads(p.read_text()) for p in sorted((OUT/b).glob('*.json'))] for b in ['codex','claude']}}
    # Keep raw text completions local; publish identifiers and metrics only.
    public={'protocol':result['protocol'],'summary':summary,'per_call':{b:[{k:v for k,v in r.items() if k not in ('completion','raw_output','error')} for r in rs] for b,rs in result['per_call'].items()}}
    audit_path=PROTOCOL.with_name(PROTOCOL.name.replace('-v1.json','-audit-v1.json'))
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        public['manual_audit']=audit['description']
        for backend,rows in public['per_call'].items():
            original={r['id']:r for r in result['per_call'][backend]}
            for row in rows:
                label=audit['labels'][backend][row['id']]
                digest=hashlib.sha256(json.dumps(original[row['id']]['completion'],sort_keys=True).encode()).hexdigest()
                if digest!=label['completion_sha256']:raise ValueError('Manual audit belongs to different completions')
                row.update(label)
            for arm in json.loads(PROTOCOL.read_text())['arms']:
                public['summary'][backend][arm]['manual_correct']=sum(r['manual_answer_correct'] for r in rows if r['arm']==arm)
    PROTOCOL.with_name(PROTOCOL.name.replace('-v1.json','-results-v1.json')).write_text(json.dumps(public,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','score']);p.add_argument('--backend',choices=['codex','claude']);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':
        if not a.backend:p.error('backend required')
        run(a.backend)
    else:score()
