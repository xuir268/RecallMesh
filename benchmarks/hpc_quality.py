"""Fixed-policy quality ablations. No model calls or label-driven edge training."""
import json,sys,time,re
from pathlib import Path
from collections import defaultdict
from assoc_mem import Memory
from assoc_mem.agent import MemoryAgent,make_prompt
from reasoning_evidence import make_cases
from locomo_retrieval import measure,evidence_ids

OUT=Path('benchmarks/results/hpc');OUT.mkdir(parents=True,exist_ok=True)

def synthetic():
    results=[];jobs=[]
    for seed in [20261006,20261007,20261008]:
        for case in make_cases(seed):
            for policy in ['combined','star']:
                with Memory(capacity=1024) as memory:
                    agent=MemoryAgent(memory,None,edge_policy=policy);ids={}
                    for i,(key,text) in enumerate(case['observations']):
                        if i and i%4==0:memory.advance()
                        ids[key]=str(agent.observe(text))
                    required={ids[f's{i}'] for i in case['required']}
                    for mode in ['associative','connected']:
                        evidence,trace=agent.context(case['question'],mode,budget=4)
                        rows={e.id for e in evidence}
                        results.append({'seed':seed,'case':case['id'],'policy':policy,'mode':mode,
                            'complete':required<=rows,'recall':len(required&rows)/len(required),
                            'selected':[e.id for e in evidence],'required':sorted(required)})
                        if seed==20261008 and (policy,mode) in [('combined','associative'),('star','connected')]:
                            jobs.append({'id':f"{case['id']}-{policy}-{mode}",'case':case['id'],
                                'policy':policy,'mode':mode,'question':case['question'],'gold':case['gold'],
                                'evidence':[{'id':e.id,'text':e.text} for e in evidence],'trace':trace})
    (OUT/'synthetic-quality.json').write_text(json.dumps(results,indent=2))
    (OUT/'answer-jobs.json').write_text(json.dumps(jobs,indent=2))
    for seed in [20261006,20261007,20261008]:
        print('seed',seed,{(p,m):sum(r['complete'] for r in results if (r['seed'],r['policy'],r['mode'])==(seed,p,m)) for p in ['combined','star'] for m in ['associative','connected']},flush=True)

def natural():
    data=json.loads(Path('benchmarks/data/locomo10.json').read_text());results=[]
    for ci,sample in enumerate(data):
        for policy in ['combined','star']:
            with Memory(capacity=1<<18) as memory:
                agent=MemoryAgent(memory,None,edge_policy=policy);inverse={};sessions=sample['conversation'];start=time.perf_counter()
                for session in sorted(int(k.split('_')[1]) for k in sessions if re.fullmatch(r'session_\d+',k)):
                    memory.tick=session
                    for turn in sessions[f'session_{session}']:
                        text=f"[{sessions.get(f'session_{session}_date_time','')}] {turn['speaker']}: {turn['text']}"
                        if turn.get('blip_caption'):text+=f" [Image caption: {turn['blip_caption']}]"
                        inverse[agent.observe(text)]=turn['dia_id']
                memory.flush();build=time.perf_counter()-start;stats=memory.engine.stats()
                for qi,qa in enumerate(sample['qa']):
                    if qa['category'] not in [1,2,3,4]:continue
                    for mode in ['associative','connected']:
                        # Frozen queries bypass context's publication; no updates during eval.
                        fn=memory.associative.connected if mode=='connected' else memory.associative.explain
                        if mode=='connected':trace=fn(qa['question'],memory.tick,budget=16)
                        else:trace=fn(qa['question'],memory.tick,budget=16,graph_weight=.4)
                        found=[inverse[r['node']] for r in trace['results']]
                        metric=measure(evidence_ids(qa.get('evidence',[])),found)
                        results.append({'conversation':sample['sample_id'],'split':'development' if ci<3 else 'internal_holdout',
                            'question':qi,'policy':policy,'mode':mode,'metric':metric,'returned':len(found)})
                print(sample['sample_id'],policy,'build_seconds',round(build,3),'edges',stats['live_edges'],'rejections',stats['rejected_full'],flush=True)
    (OUT/'natural-quality.json').write_text(json.dumps(results))
    summary={}
    for split in ['development','internal_holdout']:
        for policy in ['combined','star']:
            for mode in ['associative','connected']:
                values=[r['metric']['evidence_recall'] for r in results if (r['split'],r['policy'],r['mode'])==(split,policy,mode) and r['metric'] is not None]
                summary[f'{split}/{policy}/{mode}']={'n':len(values),'mean_evidence_recall':sum(values)/len(values)}
    (OUT/'natural-summary.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':
    synthetic()
    if '--natural' in sys.argv:natural()
