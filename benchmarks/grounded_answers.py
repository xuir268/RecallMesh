"""Paired fixed-evidence answer-profile pilot. First attempts; no gold in prompts."""
import argparse, hashlib, json, random
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.entity import EntityAwareRetriever
from assoc_mem.lexical import LexicalIndex, AssociativeRetriever
from assoc_mem.agent import CliProvider, Evidence, INSTRUCTIONS, GROUNDED_INSTRUCTIONS
from natural_coreference import turns
from agent_answers import answer_f1
from longitudinal import interval
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/grounded-answers'
PROTOCOL=ROOT/'benchmarks/protocols/grounded-answers-v1.json'

def prepare():
    if PROTOCOL.exists(): raise SystemExit('Protocol already locked')
    data=json.loads((ROOT/'benchmarks/data/locomo10.json').read_text()); samples={s['sample_id']:s for s in data}
    pool=json.loads((ROOT/'benchmarks/protocols/entity-gating-remaining-v1.json').read_text())['membership']
    rng=random.Random(20261013); chosen=[]
    for category in (1,2,3,4):
        group=sorted([c for c in pool if samples[c['conversation']]['qa'][c['question_index']]['category']==category],key=lambda c:c['id'])
        chosen+=rng.sample(group,3)
    jobs=[]
    for sample in data:
        cases=[c for c in chosen if c['conversation']==sample['sample_id']]
        if not cases:continue
        sessions=list(turns(sample));names=sorted({t['speaker'] for _,s in sessions for t in s})
        graph=EdgeStore(1<<18);index=LexicalIndex();adj=AssociativeRetriever(graph,index);entity=EntityAwareRetriever(adj,names);records={}
        for tick,session in sessions:
            previous=None
            for turn in session:
                body=turn['text']
                if turn.get('blip_caption'):body+=' [Image caption: '+turn['blip_caption']+']'
                text='['+sample['conversation'].get(f'session_{tick}_date_time','')+'] '+turn['speaker']+': '+body
                nid=index.add(text);records[nid]={'id':turn['dia_id'],'text':text};entity.add(nid,body,turn['speaker'])
                if previous:graph.reinforce(np.array([nid],np.uint32),np.array([previous],np.uint32),tick)
                previous=nid
        graph.freeze(tick)
        for c in cases:
            qa=sample['qa'][c['question_index']];evidence=[records[int(n)] for n in entity.search(qa['question'],tick)]
            jobs.append({'case':c['id'],'question':qa['question'],'gold':str(qa['answer']),'category':qa['category'],'required':c['gold'],'evidence':evidence,'input_sha256':hashlib.sha256(json.dumps({'question':qa['question'],'evidence':evidence},sort_keys=True).encode()).hexdigest()})
    protocol={'date':'2026-10-13','seed':20261013,'case_ids':[c['id'] for c in chosen],'n':12,'calls':48,'arms':['baseline','grounded'],'models':{'codex':'gpt-6-astra','claude':'claude-sonnet-4-6'},'effort':'medium','workers_per_backend':2,'selection':'three per category 1/2/3/4, random from159 residual entity-study cases before answer outcomes; earlier retrieval exposure retained, not pristine holdout','comparison':'Same entity-retrieved16 original records for both arms; contemporary fresh-session answers, only instructions differ. One model call per arm; no gold or arm labels supplied.','scoring':'unchanged token F1; paired deltas, cluster95, W/T/L, abstentions, invalid citations, failure counts. Token overlap is not semantic correctness. Diagnostic pilot, no automatic default promotion.','policy':'No retries, model substitution or prompt tuning after outcomes. Existing default stays baseline.','prompt_sha256':{a:hashlib.sha256(p.encode()).hexdigest() for a,p in [('baseline',INSTRUCTIONS),('grounded',GROUNDED_INSTRUCTIONS)]},'inputs':{j['case']:j['input_sha256'] for j in jobs}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'jobs.json').write_text(json.dumps(jobs,indent=2));PROTOCOL.write_text(json.dumps(protocol,indent=2)+'\n')
    print('Locked12 questions,48 calls')

def run(backend):
    protocol=json.loads(PROTOCOL.read_text());jobs=json.loads((OUT/'jobs.json').read_text());folder=OUT/backend;folder.mkdir(exist_ok=True)
    calls=[(j,a) for j in jobs for a in protocol['arms']];random.Random(20261013).shuffle(calls)
    def one(job,arm):
        path=folder/(job['case']+'-'+arm+'.json')
        if path.exists():return 'cached '+path.name
        provider=CliProvider(backend,protocol['models'][backend],answer_profile=arm)
        instructions=GROUNDED_INSTRUCTIONS if arm=='grounded' else INSTRUCTIONS
        assert hashlib.sha256(instructions.encode()).hexdigest()==protocol['prompt_sha256'][arm]
        assert job['input_sha256']==protocol['inputs'][job['case']]
        try:
            completion=asdict(provider.answer(job['question'],[Evidence(**e) for e in job['evidence']]));row={'status':'success','completion':completion}
        except Exception as exc:row={'status':'error','error':str(exc),'completion':{'answer':'','evidence_ids':[]}}
        answer=row['completion']['answer'];cites=set(row['completion']['evidence_ids']);available={e['id'] for e in job['evidence']}
        row.update(case=job['case'],arm=arm,input_sha256=job['input_sha256'],answer_f1=answer_f1(answer,job['gold'],job['category']),abstained=answer.strip().lower()=='not enough information',invalid_citations=sorted(cites-available),complete_evidence=set(job['required'])<=available)
        path.write_text(json.dumps(row,indent=2));return backend+' '+path.name+' '+row['status']
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,j,a) for j,a in calls]):print(f.result(),flush=True)

def score():
    result={'protocol':'grounded-answers-v1.json','summary':{},'per_call':{}}
    for backend in ('codex','claude'):
        rows=[json.loads(p.read_text()) for p in (OUT/backend).glob('*.json')];assert len(rows)==24
        paired=[]
        for case in json.loads(PROTOCOL.read_text())['case_ids']:
            pair={a:next(r for r in rows if r['case']==case and r['arm']==a) for a in ('baseline','grounded')}
            paired.append({'conversation':case.split(':')[0],**{a:pair[a]['answer_f1'] for a in pair}})
        summary={a:{'mean_f1':float(np.mean([r['answer_f1'] for r in rows if r['arm']==a])),'abstentions':sum(r['abstained'] for r in rows if r['arm']==a),'errors':sum(r['status']=='error' for r in rows if r['arm']==a),'invalid_citations':sum(bool(r['invalid_citations']) for r in rows if r['arm']==a)} for a in ('baseline','grounded')}
        summary.update(delta=float(np.mean([p['grounded']-p['baseline'] for p in paired])),cluster95=interval(paired,'grounded','baseline'),wins=sum(p['grounded']>p['baseline'] for p in paired),ties=sum(p['grounded']==p['baseline'] for p in paired),losses=sum(p['grounded']<p['baseline'] for p in paired))
        result['summary'][backend]=summary;result['per_call'][backend]=[{k:v for k,v in r.items() if k not in ('completion','error')} for r in rows]
    (ROOT/'benchmarks/protocols/grounded-answers-results-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','score']);p.add_argument('--backend',choices=['codex','claude']);args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='run':run(args.backend)
    else:score()
