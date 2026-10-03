"""Diagnostic only: generated complete facts, not production retrieval."""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
from assoc_mem.agent import Evidence
from assoc_mem.local_provider import LocalProvider
from reasoning_evidence import exact_answer

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--label',required=True);p.add_argument('--url',default='http://127.0.0.1:8087');args=p.parse_args()
provider=LocalProvider(args.model,args.url)
folder=Path('benchmarks/results/pipeline')/(args.label+'-oracle');folder.mkdir(exist_ok=True)
for case in json.loads(Path('benchmarks/results/pipeline/cases.json').read_text()):
 if case['suite']!='generated':continue
 path=folder/(case['id']+'.json')
 if path.exists():continue
 # All generated task facts, not selected from answer content. Gold is used
 # only for supervisor scoring after the model has completed.
 evidence=[Evidence(key,text) for _,key,text in case['observations'] if key.startswith('s')]
 if len(evidence)<4:
  evidence+= [Evidence(key,text) for _,key,text in case['observations'] if key=='d0']
 started=time.perf_counter()
 try:
  completion=asdict(provider.answer(case['question'],evidence))
  row={'status':'success','completion':completion,'score':float(exact_answer(completion['answer'],case['gold']))}
 except Exception as exc:
  row={'status':'error','score':0,'error':str(exc),'raw_output':getattr(exc,'raw_output',None)}
 row.update({'id':case['id'],'suite':'generated','arm':'oracle','gold':case['gold'],
             'requested_model':args.model,'seconds':time.perf_counter()-started,
             'evidence':[asdict(e) for e in evidence], 'diagnostic_only':True})
 path.write_text(json.dumps(row,indent=2));print(args.label,case['id'],row['score'],flush=True)
