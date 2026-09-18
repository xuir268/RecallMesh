"""Fresh internal sample for graph-construction answer comparison; no tuning."""
from pathlib import Path
import json,random,re,hashlib,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from assoc_mem import Memory
from assoc_mem.agent import MemoryAgent,CliProvider,Evidence,AnswerFormatError,make_prompt
from agent_answers import answer_f1
OUT=Path('benchmarks/results/hpc')

def prepare():
    data=json.loads(Path('benchmarks/data/locomo10.json').read_text())[3:]
    excluded={c['case'] for c in json.loads(Path('benchmarks/results/agent-eval/manifest.json').read_text())['cases']}
    excluded.update(c['case'] for c in json.loads(Path('benchmarks/results/reasoning-evidence/manifest.json').read_text())['natural_selection'])
    rng=random.Random(20261009);selected=[]
    for category in [1,2,4]:
        candidates=[(s['sample_id'],i) for s in data for i,q in enumerate(s['qa']) if q['category']==category and f"{s['sample_id']}-q{i}" not in excluded]
        selected+=rng.sample(candidates,4)
    jobs=[]
    for sample in data:
        questions=[i for cid,i in selected if cid==sample['sample_id']]
        if not questions:continue
        for policy in ['combined','star']:
            with Memory(capacity=1<<18) as memory:
                agent=MemoryAgent(memory,None,edge_policy=policy);ids={}
                conv=sample['conversation']
                for session in sorted(int(k.split('_')[1]) for k in conv if re.fullmatch(r'session_\d+',k)):
                    memory.tick=session
                    for t in conv[f'session_{session}']:
                        text=f"[{conv.get(f'session_{session}_date_time','')}] {t['speaker']}: {t['text']}"
                        if t.get('blip_caption'):text+=f" [Image caption: {t['blip_caption']}]"
                        ids[str(agent.observe(text))]=t['dia_id']
                memory.flush()
                for i in questions:
                    q=sample['qa'][i];ev,trace=agent.context(q['question'],'associative',16)
                    evidence=[Evidence(ids[e.id],e.text) for e in ev]
                    jobs.append({'id':f"{sample['sample_id']}-q{i}-{policy}",'case':f"{sample['sample_id']}-q{i}",
                        'policy':policy,'question':q['question'],'gold':q['answer'],'category':q['category'],
                        'evidence':[asdict(e) for e in evidence],
                        'prompt_sha256':hashlib.sha256(make_prompt(q['question'],evidence).encode()).hexdigest()})
    file=OUT/'natural-answer-jobs.json';encoded=json.dumps(jobs,indent=2)
    if file.exists() and file.read_text()!=encoded:raise ValueError('Locked input changed')
    file.write_text(encoded)
    print('Locked 12 questions, 24 calls per model; seed 20261009; 4 questions per category 1/2/4; previous answer-tested cases excluded.')

def run(backend):
    jobs=json.loads((OUT/'natural-answer-jobs.json').read_text());random.Random(20261009).shuffle(jobs)
    folder=OUT/backend;folder.mkdir(exist_ok=True);provider=CliProvider(backend)
    def one(job):
        path=folder/(job['id']+'.json')
        if path.exists():
            old=json.loads(path.read_text());assert old['prompt_sha256']==job['prompt_sha256'];return job['id']
        evidence=[Evidence(**e) for e in job['evidence']]
        assert hashlib.sha256(make_prompt(job['question'],evidence).encode()).hexdigest()==job['prompt_sha256']
        try:
            answer=asdict(provider.answer(job['question'],evidence));status='success';raw=None
        except AnswerFormatError as exc:
            answer={'answer':'','evidence_ids':[],'rationale':''};status='format_error';raw=exc.raw_output
        row={**job,'status':status,'completion':answer,'raw_output':raw}
        row['f1']=answer_f1(answer['answer'],job['gold'],job['category'])
        path.write_text(json.dumps(row,indent=2));return job['id']
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0,len(jobs),2):
            for f in as_completed([pool.submit(one,j) for j in jobs[start:start+2]]):print(backend,f.result(),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:run(sys.argv[1])
