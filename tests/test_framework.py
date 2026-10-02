from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from assoc_mem.config import MemoryConfig,ConfigError
from assoc_mem.framework import ManagedMemory,QuotaExceeded,FrameworkAgent
from assoc_mem.client import MemoryClient,MemoryClientError
from assoc_mem.agent import Completion
from assoc_mem.pipeline import EvidencePipeline


def cfg(tmp_path,**sections):return MemoryConfig(sections,base_dir=tmp_path)


def test_restart_preserves_associations_ticks_and_recall(tmp_path):
 c=cfg(tmp_path)
 with ManagedMemory(c) as m:
  a=m.remember('Mira calls her token the silver pebble.')['id'];m.advance(3)
  b=m.remember('The silver pebble is in the blue drawer.')['id']
  m.associate([a,b]);expected=m.recall('Mira token');weight=m._memory.engine.peek(m._map[a],m._map[b],m.tick)
 with ManagedMemory(c) as m:
  assert m.tick==3 and m.stats()['nodes']==2
  assert m.recall('Mira token')==expected
  assert m._memory.engine.peek(m._map[a],m._map[b],m.tick)==weight
  assert m.get(a)['text'].startswith('Mira')


def test_reject_rolls_back_and_oldest_retention_preserves_stable_ids(tmp_path):
 c=cfg(tmp_path,limits={'max_nodes':2})
 with ManagedMemory(c) as m:
  m.remember('alpha');m.remember('beta')
  with pytest.raises(QuotaExceeded):m.remember('gamma')
  assert m.stats()['nodes']==2
 c=cfg(tmp_path,limits={'max_nodes':2},retention={'overflow':'oldest'})
 with ManagedMemory(c) as m:
  result=m.remember('gamma');assert result=={'id':3,'tick':0,'evicted_ids':[1]}
  assert m.get(1) is None and m.get(2)['text']=='beta'
  assert set(m._map)=={2,3}
 with ManagedMemory(c) as m:
  assert m.get(3)['text']=='gamma';assert m._memory.store.embeddings.shape[0]==3


def test_ttl_forget_and_unicode_byte_limits(tmp_path):
 c=cfg(tmp_path,limits={'max_text_bytes':12,'max_record_bytes':12},retention={'ttl_ticks':2})
 with ManagedMemory(c) as m:
  a=m.remember('ééé')['id']
  with pytest.raises(QuotaExceeded):m.remember('x'*13)
  m.advance();b=m.remember('ββ')['id']
  assert m.advance()['expired_ids']==[a]
  assert m.get(b) is not None
  assert m.forget(b)['forgotten'];assert not m.forget(b)['forgotten']
  assert m.stats()['association_events']==0


def test_growth_is_durable_and_no_silent_rejection(tmp_path):
 c=cfg(tmp_path,graph={'initial_capacity':2,'max_capacity':32})
 with ManagedMemory(c) as m:
  for i in range(8):m.remember(f'shared token {i}')
  stats=m.stats();assert 2<stats['capacity']<=32 and stats['rejected_full']==0
 with ManagedMemory(c) as m:assert m.stats()['capacity']==stats['capacity']
 c2=cfg(tmp_path/'fixed',graph={'initial_capacity':2,'max_capacity':2,'allow_growth':False})
 with ManagedMemory(c2) as m:
  m.remember('shared a');m.remember('shared b')
  with pytest.raises(QuotaExceeded):m.remember('shared c')
  assert m.stats()['nodes']==2 and m.stats()['rejected_full']==0


def test_two_instance_caches_and_threaded_writers(tmp_path):
 c=cfg(tmp_path)
 with ManagedMemory(c) as a,ManagedMemory(c) as b:
  with ThreadPoolExecutor(max_workers=2) as pool:
   list(pool.map(lambda m:[m.remember(f'writer memory {i}') for i in range(10)], [a,b]))
  assert a.stats()['nodes']==20 and b.stats()['nodes']==20
  assert len(a._map)==20
  a.forget(1);assert b.get(1) is None


def test_database_quota_rolls_back_and_config_change_rejects_old_clients(tmp_path):
 c=cfg(tmp_path,storage={'max_database_bytes':65536},limits={'max_record_bytes':20000})
 with ManagedMemory(c) as m:
  accepted=0
  for i in range(20):
   try:m.remember('x'*16000+str(i));accepted+=1
   except QuotaExceeded:break
  assert 0<accepted<20
  assert m.stats()['nodes']==accepted
  assert m.stats()['database_bytes']<=65536
  with ManagedMemory(cfg(tmp_path,storage={'max_database_bytes':131072},limits={'max_record_bytes':20000})) as newer:
   with pytest.raises(ConfigError):m.stats()
   assert newer.stats()['nodes']==accepted


def test_stdio_sdk_and_abrupt_exit_preserve_data(tmp_path):
 config=tmp_path/'memory.json';config.write_text('{}')
 with MemoryClient(config) as client:
  first=client.remember('Astra can use these observations')['id']
  assert client.recall('Astra')['memories'][0]['id']==str(first)
  with pytest.raises(MemoryClientError):client.call('unknown')
  assert client.stats()['nodes']==1
 # Kill after the successful write; persistence must not depend on close().
 process=subprocess.Popen([sys.executable,'-m','assoc_mem.cli','serve','--config',str(config)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
 process.stdin.write(json.dumps({'id':1,'method':'remember','params':{'text':'second fact'}})+'\n');process.stdin.flush()
 assert json.loads(process.stdout.readline())['ok']
 process.kill();process.wait();process.stdin.close();process.stdout.close()
 with MemoryClient(config) as client:assert client.stats()['nodes']==2


def test_bad_and_oversized_stdio_requests_recover(tmp_path):
 p=tmp_path/'cfg.json';p.write_text(json.dumps({'limits':{'max_request_bytes':1024}}))
 inputs='bad json\n'+json.dumps({'method':'remember','params':{'text':'x'*2000}})+'\n'+json.dumps({'id':3,'method':'stats'})+'\n'
 r=subprocess.run([sys.executable,'-m','assoc_mem.cli','serve','--config',str(p)],input=inputs,text=True,capture_output=True,check=True)
 rows=[json.loads(s) for s in r.stdout.splitlines()]
 assert [r.get('error',{}).get('code') for r in rows[:2]]==['invalid_json','request_too_large']
 assert rows[2]['ok'] and rows[2]['id']==3


def test_config_validation_context_bound_and_pipeline_adapter(tmp_path):
 for bad in [{'unknown':1},{'graph':{'initial_capacity':3}},{'graph':{'allow_growth':'yes'}},{'retention':{'overflow':'oops'}}]:
  with pytest.raises(ConfigError):cfg(tmp_path,**bad)
 with ManagedMemory(cfg(tmp_path,retrieval={'max_context_bytes':80})) as memory:
  memory.remember('x'*200)
  assert not memory.recall('x')['memories']
 class Provider:
  def structured(self,instructions,data,schema):
   if 'selected_ids' in schema['properties']:
    return {'data':{'selected_ids':[r['id'] for r in data['records']], 'queries':[],'missing':''},'metadata':{}}
   return {'data':{'supported':True,'reason':'explicit'},'metadata':{}}
  def answer(self,question,evidence):return Completion('blue',[e.id for e in evidence],'facts','custom',0)
 with ManagedMemory(cfg(tmp_path/'adapter')) as memory:
  agent=FrameworkAgent(memory,Provider());agent.observe('The token is blue.')
  result=EvidencePipeline(agent).ask('token color?')
  assert result['response']['answer']=='blue'


def test_independent_cli_processes_share_durable_database(tmp_path):
 config=tmp_path/'memory.json';config.write_text('{}')
 with MemoryClient(config) as a,MemoryClient(config) as b:
  with ThreadPoolExecutor(max_workers=2) as pool:
   list(pool.map(lambda client:[client.remember(f'independent process record {i}') for i in range(6)],[a,b]))
  assert a.stats()['nodes']==12 and b.stats()['nodes']==12
  a.forget(1);assert b.call('get',id=1) is None
 with MemoryClient(config) as m:assert m.stats()['nodes']==11


def test_config_rejects_nonobject_and_invalid_enum(tmp_path):
 for data in [[],0,False,{'graph':{'edge_policy':[]}}, {'retention':{'overflow':[]}}]:
  with pytest.raises(ConfigError):cfg(tmp_path,**data) if isinstance(data,dict) else MemoryConfig(data)
