import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import numpy as np
import pytest
from assoc_mem import EdgeStore
from assoc_mem.client import MemoryClient
from assoc_mem.config import MemoryConfig
from assoc_mem.framework import ManagedMemory
from assoc_mem.embed import HashEmbedder
pytestmark=pytest.mark.skipif(sys.platform=='win32',reason='native mmap checkpoints currently require POSIX')


def test_native_mapped_roundtrip_preserves_packed_ticks_and_future_updates(tmp_path):
 s=EdgeStore(128,lambda_=.05)
 a=np.array([1,1,2],np.uint32);b=np.array([2,3,4],np.uint32)
 s.reinforce(a,b,3);s.reinforce(a,b,9);s.freeze(12)
 path=str(tmp_path/'graph.csr');s.save_checkpoint(path)
 r=EdgeStore(128,lambda_=.05);r.restore_checkpoint(path)
 seeds=np.array([1],np.uint32);w=np.array([1],np.float32)
 for tick in [12,20]:
  expected=s.activate(seeds,w,tick=tick);actual=r.activate(seeds,w,tick=tick)
  np.testing.assert_array_equal(expected[0],actual[0]);np.testing.assert_array_equal(expected[1],actual[1])
  for x,y in zip(a,b):assert r.peek(int(x),int(y),tick)==s.peek(int(x),int(y),tick)
 for store in [s,r]:store.reinforce(a,b,20);store.freeze(20)
 np.testing.assert_array_equal(s.weights_for_pairs(a,b,tick=20),r.weights_for_pairs(a,b,tick=20))
 with pytest.raises(RuntimeError):r.restore_checkpoint(path)


def test_dead_delta_edges_survive_checkpoint_for_exact_resurrection(tmp_path):
 a=np.array([1],np.uint32);b=np.array([2],np.uint32);s=EdgeStore(128,lambda_=.5)
 s.reinforce(a,b,0);s.freeze(11);assert s.edge_count==0
 path=str(tmp_path/'dead');s.save_checkpoint(path);r=EdgeStore(128,lambda_=.5);r.restore_checkpoint(path)
 assert r.peek(1,2,11)==s.peek(1,2,11)>0
 for engine in [s,r]:engine.reinforce(a,b,11)
 assert r.peek(1,2,11)==s.peek(1,2,11)


def test_managed_mmap_restart_and_copy_on_write(tmp_path,monkeypatch):
 c=MemoryConfig({},tmp_path)
 with ManagedMemory(c) as m:
  a=m.remember('Mira calls her token the silver pebble.')['id'];m.advance(2)
  b=m.remember('The silver pebble is in the blue drawer.')['id'];m.associate([a,b])
  before=m.recall('Mira token');emb=m._memory.store.embeddings.copy();weight=m._memory.engine.peek(1,2,m.tick)
  m.checkpoint()
 with monkeypatch.context() as patch:
  patch.setattr(HashEmbedder,'encode',lambda *args: (_ for _ in ()).throw(AssertionError('restart must not regenerate embeddings')))
  with ManagedMemory(c) as m:
   assert m.stats()['recovery']=='mmap_checkpoint'
   assert isinstance(m._memory.store._emb,np.memmap)
   assert m.tick==2 and m._memory.engine.peek(1,2,m.tick)==weight
   np.testing.assert_array_equal(emb,m._memory.store.embeddings)
   assert m.recall('Mira token')==before
 with ManagedMemory(c) as m:
  m.remember('Mira uses the blue drawer daily.')
  assert m._memory.store._emb.flags.writeable
 with ManagedMemory(c) as m:assert m.stats()['recovery']=='event_replay' and m.stats()['nodes']==3


def test_corrupt_checkpoint_falls_back_to_durable_log(tmp_path):
 c=MemoryConfig({},tmp_path)
 with ManagedMemory(c) as m:
  m.remember('shared alpha');m.remember('shared beta');expected=m.recall('alpha');saved=m.checkpoint()
 root=Path(saved['path']);(root/saved['generation']/'graph.csr').write_bytes(b'truncated')
 with ManagedMemory(c) as m:
  assert m.stats()['recovery']=='event_replay' and m.stats()['checkpoint_error']
  assert m.recall('alpha')==expected


def test_checkpoint_commit_survives_abrupt_client_exit(tmp_path):
 config=tmp_path/'memory.json';config.write_text('{}')
 client=MemoryClient(config);client.remember('durable token');client.advance(5);client.call('checkpoint')
 client._process.kill();client._process.wait();client.close()
 with ManagedMemory(config) as m:assert m.tick==5 and m.stats()['recovery']=='mmap_checkpoint' and m.stats()['nodes']==1


def test_v1_database_migrates_embeddings_and_ticks(tmp_path):
 c=MemoryConfig({},tmp_path);c.path.parent.mkdir(parents=True)
 db=sqlite3.connect(c.path)
 db.executescript('CREATE TABLE memories(id INTEGER PRIMARY KEY AUTOINCREMENT,text TEXT NOT NULL,writer TEXT NOT NULL,tick INTEGER NOT NULL,bytes INTEGER NOT NULL); CREATE TABLE framework_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);')
 db.execute('INSERT INTO memories(text,writer,tick,bytes) VALUES (?,?,?,?)',('old memory','',4,10))
 db.executemany('INSERT INTO framework_meta VALUES (?,?)',[('schema_version','1'),('tick','7')]);db.commit();db.close()
 with ManagedMemory(c) as m:
  assert m.tick==7 and m.get(1)['text']=='old memory'
  assert len(m.db.execute('SELECT embedding FROM memories').fetchone()[0])==128*4
  m.checkpoint()
 with ManagedMemory(c) as m:assert m.stats()['recovery']=='mmap_checkpoint'


def test_native_rejects_corruption_and_learning_mismatch(tmp_path):
 s=EdgeStore(128);s.freeze(0);path=str(tmp_path/'empty');s.save_checkpoint(path)
 with pytest.raises(RuntimeError):EdgeStore(128,lambda_=.2).restore_checkpoint(path)
 f=Path(path+'.delta');raw=bytearray(f.read_bytes());raw[-1]^=1;f.write_bytes(raw)
 with pytest.raises(RuntimeError):EdgeStore(128).restore_checkpoint(path)


def test_native_forged_offset_with_valid_checksum_is_rejected(tmp_path):
 import struct
 s=EdgeStore(128);s.reinforce(np.array([1],np.uint32),np.array([2],np.uint32),0);s.freeze(0)
 path=str(tmp_path/'offset');s.save_checkpoint(path);f=Path(path+'.delta');raw=bytearray(f.read_bytes())
 # Last offset must equal two directed entries. Keep the checksum valid so
 # the structural parser, not only the checksum, must reject the corruption.
 header=list(struct.unpack_from('<11Q',raw));struct.pack_into('<Q',raw,88+header[2]*8,1000)
 h=14695981039346656037
 for v in raw[88:]:h=((h^v)*1099511628211)&((1<<64)-1)
 struct.pack_into('<Q',raw,80,h);f.write_bytes(raw)
 with pytest.raises(RuntimeError):EdgeStore(128).restore_checkpoint(path)


def test_checkpoint_quota_rejection_preserves_previous_manifest_and_database(tmp_path):
 from assoc_mem.framework import QuotaExceeded
 c=MemoryConfig({'storage':{'max_checkpoint_bytes':65536}},tmp_path)
 with ManagedMemory(c) as m:
  m.remember('original record');m.checkpoint();manifest=(m._checkpoint_root/'current.json').read_bytes()
  ids=[1]+[m.remember(f'additional shared record {i}')['id'] for i in range(63)]
  m.associate(ids)
  with pytest.raises(QuotaExceeded):m.checkpoint()
  assert (m._checkpoint_root/'current.json').read_bytes()==manifest
  assert m.stats()['nodes']==64 and len(list(m._checkpoint_root.glob('generation-*')))==1
 with ManagedMemory(c) as m:assert m.stats()['nodes']==64 and m.stats()['recovery']=='event_replay'
