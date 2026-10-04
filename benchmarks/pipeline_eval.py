"""Fixed first-attempt answer pilot. Labels never enter provider inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re
import time
from assoc_mem import Memory
from assoc_mem.agent import MemoryAgent, CliProvider, Evidence
from assoc_mem.local_provider import LocalProvider
from assoc_mem.pipeline import EvidencePipeline, evidence_bytes
from reasoning_evidence import make_cases, exact_answer
from agent_answers import answer_f1

OUT = Path('benchmarks/results/pipeline')
SEED = 20261010


def prepare():
    cases=[]
    for c in make_cases(SEED):
        if int(c['id'].split('-')[-1])>=2: continue
        cases.append({'id':'generated-'+c['id'], 'suite':'generated', 'question':c['question'],
            'gold':c['gold'], 'category':0, 'required':[f's{i}' for i in c['required']],
            'observations':[[i//4,k,t] for i,(k,t) in enumerate(c['observations'])],
            'budget':4,'candidates':8, 'context_bytes':3000})
    excluded=set()
    for file,key in [('benchmarks/results/agent-eval/manifest.json','cases'),
                     ('benchmarks/results/reasoning-evidence/manifest.json','natural_selection')]:
        excluded.update(c['case'] for c in json.loads(Path(file).read_text())[key])
    excluded.update(c['case'] for c in json.loads(Path('benchmarks/results/hpc/natural-answer-jobs.json').read_text()))
    dataset=json.loads(Path('benchmarks/data/locomo10.json').read_text())[3:]
    rng=random.Random(SEED)
    for category in [1,2,4]:
        eligible=[(s,i,q) for s in dataset for i,q in enumerate(s['qa'])
                  if q['category']==category and f"{s['sample_id']}-q{i}" not in excluded]
        for sample,i,q in rng.sample(eligible,2):
            obs=[];conv=sample['conversation']
            for tick in sorted(int(k.split('_')[1]) for k in conv if re.fullmatch(r'session_\d+',k)):
                for turn in conv[f'session_{tick}']:
                    text=f"[{conv.get(f'session_{tick}_date_time','')}] {turn['speaker']}: {turn['text']}"
                    if turn.get('blip_caption'):text+=f" [Image caption: {turn['blip_caption']}]"
                    obs.append([tick,turn['dia_id'],text])
            cases.append({'id':f"{sample['sample_id']}-q{i}",'suite':'natural','question':q['question'],
                          'gold':q['answer'],'category':category,'required':q['evidence'],
                          'observations':obs,'budget':8,'candidates':24,'context_bytes':6000})
    OUT.mkdir(exist_ok=True,parents=True)
    encoded=json.dumps(cases,indent=2)
    p=OUT/'cases.json'
    if p.exists() and p.read_text()!=encoded:raise ValueError('Locked cases differ')
    p.write_text(encoded)
    (OUT/'protocol.json').write_text(json.dumps({'seed':SEED,'cases_sha256':hashlib.sha256(encoded.encode()).hexdigest(),
        'design':'6 fresh generated instances from 3 known families; 6 fresh natural questions, 2 per category 1/2/4, prior answer-tested questions excluded.',
        'arms':['single','bounded'],'policy':'combined','rounds':2,
        'budgets':'Same final record count; bounded arm sees more candidates and uses 3-4 model calls. This is NOT compute-matched.',
        'compression':'Whole source records retained; JSON UTF-8 byte counts, not exact tokenizer counts.',
        'failures':'First attempts saved, no retries. Semantic support verifier is fallible.',
        'local_settings':'temperature 0, seed 42, thinking disabled, max output 1024, server context 8192'},indent=2))
    print('Locked',len(cases),'cases')


def execute(args):
    cases=json.loads((OUT/'cases.json').read_text());folder=OUT/args.label;folder.mkdir(exist_ok=True)
    provider=(LocalProvider(args.model,args.url) if args.backend=='local' else CliProvider(args.backend,args.model))
    def one(case,arm):
        path=folder/(case['id']+'-'+arm+'.json')
        if path.exists():
            cached=json.loads(path.read_text())
            if cached['requested_model']!=provider.model:raise ValueError('Cached model mismatch; use a new label')
            return 'cached '+path.name
        started=time.perf_counter();ids={};pipeline=None
        try:
            with Memory(capacity=1<<18) as memory:
                agent=MemoryAgent(memory,provider)
                for tick,key,text in case['observations']:
                    memory.tick=tick;ids[str(agent.observe(text))]=key
                if arm in {'single','wide'}:
                    result=agent.ask(case['question'],budget=case['candidates'] if arm=='wide' else case['budget'])
                    result['metrics']={'model_calls':1,'final_evidence_bytes':evidence_bytes([
                        Evidence(**e) for e in result['evidence']])}
                else:
                    pipeline=EvidencePipeline(agent,rounds=2,candidates=case['candidates'],
                        final_records=case['budget'],context_bytes=case['context_bytes'],review_bytes=20000)
                    result=pipeline.ask(case['question'])
            score=(float(exact_answer(result['response']['answer'],case['gold'])) if case['suite']=='generated'
                   else answer_f1(result['response']['answer'],case['gold'],case['category']))
            selected={ids[e['id']] for e in result['evidence']}
            row={'status':'success','score':score,'complete_evidence':set(case['required'])<=selected,'result':result}
        except Exception as exc:
            row={'status':'error','score':0,'error':str(exc),'raw_output':getattr(exc,'raw_output',None),
                 'completed_stages': getattr(pipeline,'stages',[])}
        row.update({'id':case['id'],'suite':case['suite'],'arm':arm,'requested_model':provider.model,
                    'gold':case['gold'],'seconds':time.perf_counter()-started})
        path.write_text(json.dumps(row,indent=2,ensure_ascii=False))
        return f"{args.label} {path.name}: {row['status']} score={row['score']:.3f}"
    jobs=[(c,a) for c in cases for a in args.arms]
    random.Random(SEED).shuffle(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one,c,a) for c,a in jobs]):print(f.result(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--backend',choices=['codex','claude','local']);p.add_argument('--label');p.add_argument('--model');p.add_argument('--url',default='http://127.0.0.1:8080');p.add_argument('--workers',type=int,default=1);p.add_argument('--arms',nargs='+',choices=['single','bounded','wide'],default=['single','bounded']);args=p.parse_args()
    if args.command=='prepare':prepare()
    else:
        if not args.backend or not args.label or not 1<=args.workers<=2:p.error('run needs backend, label, 1..2 workers')
        execute(args)
