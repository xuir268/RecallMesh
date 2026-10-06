"""Warm-file-cache restart timing in fresh processes, not power-loss durability."""
from pathlib import Path
import json,subprocess,sys,time,tempfile
import numpy as np
from assoc_mem.framework import ManagedMemory


def main():
 out=Path('benchmarks/results/recovery');out.mkdir(parents=True,exist_ok=True)
 config=Path(tempfile.mkdtemp())/'memory.json';config.write_text('{}')
 with ManagedMemory(config) as m:
  ids=[m.remember(f'record {i} has a shared token')['id'] for i in range(64)]
  for _ in range(50):m.associate(ids)
  m.advance(17);m.checkpoint();root=m._checkpoint_root;events=m.stats()['association_events']
  expected=m.recall('record 17');weight=m._memory.engine.peek(1,2,m.tick)
 code='''import json,time,sys,hashlib
from assoc_mem.framework import ManagedMemory
start=time.perf_counter()
with ManagedMemory(sys.argv[1]) as m:
 print(json.dumps({'seconds':time.perf_counter()-start,'recovery':m.stats()['recovery'],'tick':m.tick,'nodes':m.stats()['nodes'],'weight':m._memory.engine.peek(1,2,m.tick),'recall':m.recall('record 17'),'embedding_sha256':hashlib.sha256(m._memory.store.embeddings.tobytes()).hexdigest()}))
'''
 rows=[]
 for repeat in range(5):
  for mode in (['event_replay','mmap_checkpoint'] if repeat%2==0 else ['mmap_checkpoint','event_replay']):
   if mode=='event_replay':(root/'current.json').rename(root/'disabled.json')
   try:
    r=subprocess.run([sys.executable,'-c',code,str(config)],check=True,capture_output=True,text=True);row=json.loads(r.stdout)
   finally:
    if mode=='event_replay':(root/'disabled.json').rename(root/'current.json')
   assert row['weight']==weight and row['recall']==expected and row['tick']==17 and row['nodes']==64 and row['recovery']==mode
   rows.append(row)
 assert len({r['embedding_sha256'] for r in rows})==1
 report={'records':64,'association_events':events,'repeats_per_mode':5,'warm_file_cache':True,'fresh_process_each_restart':True,
         'median_seconds':{mode:float(np.median([r['seconds'] for r in rows if r['recovery']==mode])) for mode in ['event_replay','mmap_checkpoint']},
         'exact_recovery_equal':True,'rows':rows}
 (out/'restart.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
