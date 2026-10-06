"""Model-independent durable memory. SQLite is authoritative; native indexes
are rebuildable caches. Stable public IDs are mapped to compact engine IDs.
"""
from contextlib import contextmanager
from dataclasses import asdict
import json
import sqlite3
import threading
import hashlib
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
from . import Memory
from .agent import Evidence
from .config import MemoryConfig, ConfigError
from .lexical import stable_topk


class QuotaExceeded(RuntimeError):
    pass


SCHEMA = '''
CREATE TABLE IF NOT EXISTS framework_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
 id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, writer TEXT NOT NULL,
 tick INTEGER NOT NULL, bytes INTEGER NOT NULL, embedding BLOB);
CREATE TABLE IF NOT EXISTS associations (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 a INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
 b INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
 tick INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS associations_a ON associations(a);
CREATE INDEX IF NOT EXISTS associations_b ON associations(b);
'''


class ManagedMemory:
    """Persistent, bounded memory API. Calls are serialized per instance and
    across processes through SQLite transactions. Reads never train the graph.
    Graph growth rebuilds a replacement from committed association history;
    native lock-free slot migration is not involved.
    """
    def __init__(self,config='memory.json'):
        self.config=config if isinstance(config,MemoryConfig) else MemoryConfig.load(config)
        self.config.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.config.path,timeout=30,check_same_thread=False,isolation_level=None)
        self._lock=threading.RLock();self._memory=None;self._revision=-1;self._graph_dirty=False;self._recovery="event_replay";self._checkpoint_error=None
        try:
            self.db.execute('PRAGMA foreign_keys=ON')
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('PRAGMA auto_vacuum=FULL')
            self.db.execute('PRAGMA secure_delete=ON')
            size=self.db.execute('PRAGMA page_size').fetchone()[0]
            pages=self.config['storage']['max_database_bytes']//size
            if self.db.execute('PRAGMA page_count').fetchone()[0]>pages:
                raise QuotaExceeded('Existing database exceeds new limit; compact under the prior limit first')
            actual=self.db.execute(f'PRAGMA max_page_count={pages}').fetchone()[0]
            if actual>pages:raise QuotaExceeded('Cannot apply database page limit')
            self.db.executescript(SCHEMA)
            with self._lock:
                self.db.execute('BEGIN IMMEDIATE')
                try:
                    self._set_default('schema_version','2');self._set_default('tick','0')
                    self._set_default('revision','0');self._set_default('capacity',str(self.config['graph']['initial_capacity']))
                    self._set_default('learning',self.config.learning_fingerprint)
                    if self._get('schema_version') not in {'1','2'}:raise ConfigError('unsupported database schema')
                    columns={r[1] for r in self.db.execute('PRAGMA table_info(memories)')}
                    if 'embedding' not in columns:self.db.execute('ALTER TABLE memories ADD COLUMN embedding BLOB')
                    from .embed import HashEmbedder
                    for nid,text in self.db.execute('SELECT id,text FROM memories WHERE embedding IS NULL').fetchall():
                        blob=HashEmbedder().encode([text])[0].astype('<f4').tobytes()
                        self.db.execute('UPDATE memories SET embedding=? WHERE id=?',(blob,nid))
                    self._set('schema_version','2')
                    if self._get('learning')!=self.config.learning_fingerprint:
                        raise ConfigError('Learning parameters are fixed for this database; use a new storage path')
                    if self._get('config')!=self.config.fingerprint:
                        self._set('config',self.config.fingerprint);self._bump_revision()
                    self._enforce_limits();self._reload();self.db.commit()
                except Exception:self.db.rollback();raise
        except Exception:
            if self._memory:self._memory.close()
            self.db.close();raise

    def _get(self,key):
        row=self.db.execute('SELECT value FROM framework_meta WHERE key=?',(key,)).fetchone()
        return row[0] if row else None
    def _set(self,key,value):
        self.db.execute('INSERT INTO framework_meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,str(value)))
    def _set_default(self,key,value):
        self.db.execute('INSERT OR IGNORE INTO framework_meta VALUES (?,?)',(key,value))
    def _bump_revision(self):self._set('revision',int(self._get('revision'))+1)

    @property
    def tick(self):return int(self._get('tick'))

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                if self._get('config')!=self.config.fingerprint:
                    raise ConfigError('Configuration changed in another process; reopen with the current config')
                if self._revision!=int(self._get('revision')):self._reload()
                yield
                self.db.commit()
            except sqlite3.Error as exc:
                self.db.rollback();self._revision=-1
                if getattr(exc,'sqlite_errorcode',None)==sqlite3.SQLITE_FULL or 'full' in str(exc).lower():
                    raise QuotaExceeded('Database page budget reached; forget/compact or raise the configured limit') from exc
                raise
            except Exception:
                self.db.rollback();self._revision=-1;raise

    def _reload(self,exclude=None):
        graph=self.config['graph'];capacity=max(graph['initial_capacity'],min(int(self._get('capacity')),graph['max_capacity']))
        rows=self.db.execute('SELECT id,text,writer,tick,embedding FROM memories WHERE id!=? ORDER BY id',(exclude or -1,)).fetchall()
        while True:
            memory=Memory(capacity=capacity,lambda_=graph['lambda'],eta=graph['eta'],floor=graph['floor'],graph_weight=self.config['retrieval']['graph_weight'])
            mapping={};reverse={}
            try:
                for public,text,writer,tick,blob in rows:
                    emb=np.frombuffer(blob,dtype='<f4')
                    if emb.shape!=(memory.embedder.dim,) or not np.isfinite(emb).all():raise ConfigError('Invalid persisted embedding')
                    memory.tick=tick
                    internal=memory.store.add(text,emb,tick,writer=writer)
                    memory.lexical_index.add(text)
                    mapping[public]=internal;reverse[internal]=public
                restored=False
                if exclude is None:
                    try:
                        manifest_path=self._checkpoint_root/'current.json'
                        if manifest_path.exists():
                            manifest=json.loads(manifest_path.read_text())
                            if (manifest.get('version')==1 and manifest['revision']==int(self._get('revision')) and manifest['learning']==self.config.learning_fingerprint
                                and manifest['capacity']==capacity and manifest['ids']==list(mapping)):
                                directory=self._checkpoint_root/manifest['generation']
                                if directory.parent!=self._checkpoint_root or not directory.name.startswith('generation-'):raise ValueError('invalid checkpoint path')
                                for name,expected in manifest['hashes'].items():
                                    if name not in {'graph.csr','graph.csr.delta','embeddings.npy'}:raise ValueError('invalid checkpoint file')
                                    if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:raise ValueError('checkpoint checksum mismatch')
                                if set(manifest['hashes'])!={'graph.csr','graph.csr.delta','embeddings.npy'}:raise ValueError('incomplete checkpoint')
                                embedding=np.load(directory/'embeddings.npy',mmap_mode='r',allow_pickle=False)
                                if embedding.dtype!=np.float32 or embedding.shape!=(len(rows)+1,memory.embedder.dim) or not np.isfinite(embedding).all():raise ValueError('invalid checkpoint embeddings')
                                if not np.array_equal(embedding,memory.store.embeddings):raise ValueError('checkpoint embeddings mismatch')
                                memory.engine.restore_checkpoint(str(directory/'graph.csr'))
                                memory.store._emb=embedding;memory.store._used_rows=len(rows)+1
                                memory.engine.attach_embeddings(memory.store.embeddings)
                                self._recovery='mmap_checkpoint';restored=True
                    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
                        self._checkpoint_error=str(exc)
                        # A failed native restore can have filled some slots. Recreate it.
                        from . import EdgeStore
                        memory.engine=EdgeStore(capacity,graph['lambda'],graph['eta'],graph['floor'])
                        memory.associative.engine=memory.engine;memory.queue.engine=memory.engine
                if not restored:
                    batch=[];last_tick=None
                    def apply():
                        if batch:
                            ab=np.asarray(batch,np.uint32);memory.engine.reinforce(ab[:,0].copy(),ab[:,1].copy(),last_tick)
                    for a,b,tick in self.db.execute('SELECT a,b,tick FROM associations ORDER BY id'):
                        if a not in mapping or b not in mapping:continue
                        if last_tick is not None and (tick!=last_tick or len(batch)>=256):apply();batch=[]
                        last_tick=tick;batch.append((mapping[a],mapping[b]))
                    apply();memory.tick=self.tick;memory.flush()
                    self._recovery='event_replay'
                else:memory.tick=self.tick
                if memory.engine.stats()['rejected_full']:
                    memory.close()
                    if not graph['allow_growth'] or capacity>=graph['max_capacity']:
                        raise QuotaExceeded('Graph capacity/probe limit reached; raise max_capacity or forget memories')
                    capacity=min(capacity*2,graph['max_capacity']);continue
            except Exception:
                try:memory.close()
                except sqlite3.Error:pass
                raise
            if self._memory:self._memory.close()
            self._memory=memory;self._map=mapping;self._reverse=reverse
            self._set('capacity',capacity);self._revision=int(self._get('revision'));self._graph_dirty=False;return

    def _enforce_limits(self,protect=None):
        limits=self.config['limits'];ttl=self.config['retention']['ttl_ticks'];deleted=[]
        if ttl:
            old=[r[0] for r in self.db.execute('SELECT id FROM memories WHERE tick<=?',(self.tick-ttl,))]
            for nid in old:self.db.execute('DELETE FROM memories WHERE id=?',(nid,))
            deleted.extend(old)
        while True:
            count,size=self.db.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM memories').fetchone()
            if count<=limits['max_nodes'] and size<=limits['max_text_bytes']:break
            if self.config['retention']['overflow']=='reject':raise QuotaExceeded('Node or text quota reached')
            row=self.db.execute('SELECT id FROM memories WHERE id!=? ORDER BY id LIMIT 1',(protect or -1,)).fetchone()
            if not row:raise QuotaExceeded('New record cannot fit the configured quota')
            deleted.append(row[0]);self.db.execute('DELETE FROM memories WHERE id=?',(row[0],))
        if deleted:self._bump_revision()
        return deleted

    def remember(self,text,writer=''):
        if not isinstance(text,str) or not text.strip():raise ValueError('text must be a nonempty string')
        if not isinstance(writer,str):raise ValueError('writer must be a string')
        size=len(text.encode('utf-8'))+len(writer.encode('utf-8'))
        if size>self.config['limits']['max_record_bytes']:raise QuotaExceeded('Record byte budget exceeded')
        with self._transaction():
            embedding=self._memory.embedder.encode([text])[0].astype('<f4')
            nid=int(self.db.execute('INSERT INTO memories(text,writer,tick,bytes,embedding) VALUES (?,?,?,?,?)',(text,writer,self.tick,size,embedding.tobytes())).lastrowid)
            evicted=self._enforce_limits(protect=nid)
            # On eviction rebuild first without the just-added record, so its
            # neighbor selection sees exactly the retained chronological prefix.
            if evicted:
                self._reload(exclude=nid)
            memory=self._memory;scores=memory.lexical_index.scores(text)
            related=stable_topk(scores,3);related=[self._reverse[int(i)+1] for i in related if scores[i]>0]
            previous=self.db.execute('SELECT id FROM memories WHERE id<? AND tick=? ORDER BY id DESC LIMIT 1',(nid,self.tick)).fetchone()
            active={nid}
            if previous:active.add(previous[0])
            if self.config['graph']['edge_policy']!='adjacent':active.update(related)
            active=sorted(active)
            pairs=([(a,b) for i,a in enumerate(active) for b in active[i+1:]] if self.config['graph']['edge_policy']=='combined' else
                   [(min(nid,n),max(nid,n)) for n in active if n!=nid])
            self.db.executemany('INSERT INTO associations(a,b,tick) VALUES (?,?,?)',[(a,b,self.tick) for a,b in pairs])
            internal=memory.store.add(text,embedding,self.tick,writer=writer)
            memory.lexical_index.add(text);memory._dirty=True
            self._map[nid]=internal;self._reverse[internal]=nid
            if pairs:
                ab=np.asarray([(self._map[a],self._map[b]) for a,b in pairs],np.uint32)
                memory.engine.reinforce(ab[:,0].copy(),ab[:,1].copy(),self.tick)
            self._graph_dirty=True
            self._bump_revision()
            if memory.engine.stats()['rejected_full']:self._reload()
            self._revision=int(self._get('revision'))
            return {'id':nid,'tick':self.tick,'evicted_ids':evicted}

    def context(self,query,mode='associative',limit=None,hops=None):
        if not isinstance(query,str) or not query.strip():raise ValueError('query must be nonempty')
        if mode not in {'none','lexical','associative'}:raise ValueError('unsupported retrieval mode')
        policy=self.config['retrieval'];limit=policy['default_limit'] if limit is None else limit
        hops=policy['hops'] if hops is None else hops
        if type(limit) is not int or not 1<=limit<=policy['max_limit']:raise ValueError('invalid retrieval limit')
        if type(hops) is not int or not 0<=hops<=8:raise ValueError('hops must be 0..8')
        with self._transaction():
            if mode=='none':return [],{}
            self._publish()
            explanation=self._memory.associative.explain(query,self.tick,limit,0 if mode=='lexical' else policy['graph_weight'],hops=hops)
            records=[];results=[]
            for hit in explanation['results']:
                internal=hit['node'];public=self._reverse[internal]
                records.append(Evidence(str(public),self._memory.store.get(internal).content))
                hit['node']=public;hit['path']=[self._reverse[n] for n in hit['path']];results.append(hit)
            explanation['seeds']=[self._reverse[n] for n in explanation['seeds']]
            explanation['results']=results
            from .pipeline import fit_records
            kept=fit_records(records,policy['max_context_bytes'],len(records))
            explanation['omitted_ids']=[e.id for e in records if e not in kept]
            return kept,explanation

    def recall(self,query,limit=None,hops=None,mode='associative'):
        from .pipeline import fit_records
        records,trace=self.context(query,mode,limit,hops)
        kept=fit_records(records,self.config['retrieval']['max_context_bytes'],len(records))
        return {'memories':[asdict(e) for e in kept], 'retrieval':trace,
                'context_bytes_limit':self.config['retrieval']['max_context_bytes'],
                'omitted_ids':trace.get('omitted_ids',[])+[e.id for e in records if e not in kept]}

    def get(self,nid):
        if type(nid) is not int or nid<=0:raise ValueError('id must be positive')
        with self._transaction():
            row=self.db.execute('SELECT id,text,writer,tick FROM memories WHERE id=?',(nid,)).fetchone()
            return dict(zip(['id','text','writer','tick'],row)) if row else None

    def associate(self,ids):
        if (not isinstance(ids,list) or not 2<=len(ids)<=64 or
            any(type(i) is not int or i<=0 for i in ids) or len(set(ids))!=len(ids)):
            raise ValueError('ids must contain 2..64 distinct positive integers')
        with self._transaction():
            if any(i not in self._map for i in ids):raise ValueError('unknown memory id')
            ordered=sorted(ids);pairs=[(a,b) for i,a in enumerate(ordered) for b in ordered[i+1:]]
            self.db.executemany('INSERT INTO associations(a,b,tick) VALUES (?,?,?)',[(a,b,self.tick) for a,b in pairs])
            ab=np.asarray([(self._map[a],self._map[b]) for a,b in pairs],np.uint32)
            self._memory.engine.reinforce(ab[:,0].copy(),ab[:,1].copy(),self.tick)
            self._graph_dirty=True
            self._bump_revision()
            if self._memory.engine.stats()['rejected_full']:self._reload()
            self._revision=int(self._get('revision'))
            return {'pairs_reinforced':len(pairs),'tick':self.tick}

    def forget(self,nid):
        if type(nid) is not int or nid<=0:raise ValueError('id must be a positive integer')
        with self._transaction():
            changed=self.db.execute('DELETE FROM memories WHERE id=?',(nid,)).rowcount
            if changed:self._bump_revision();self._reload()
            return {'forgotten':bool(changed),'id':nid}

    def advance(self,steps=1):
        if type(steps) is not int or steps<=0:raise ValueError('steps must be positive')
        with self._transaction():
            tick=self.tick+steps
            if tick>=(1<<32):raise ValueError('tick limit reached')
            self._set('tick',tick);expired=self._enforce_limits();self._bump_revision()
            self._memory.tick=tick;self._graph_dirty=True
            if expired:self._reload()
            self._revision=int(self._get('revision'))
            return {'tick':tick,'expired_ids':expired}

    def stats(self):
        with self._transaction():
            s=self._memory.stats();count,text=self.db.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM memories').fetchone()
            pages=self.db.execute('PRAGMA page_count').fetchone()[0];page_size=self.db.execute('PRAGMA page_size').fetchone()[0]
            s.update({'nodes':count,'text_bytes':text,'database_bytes':pages*page_size,
                      'association_events':self.db.execute('SELECT COUNT(*) FROM associations').fetchone()[0],
                      'graph_slot_bytes':s['capacity']*16,'limits':self.config.data,
                      'storage_path':str(self.config.path),'revision':self._revision,
                      'recovery':self._recovery,'checkpoint_error':self._checkpoint_error})
            return s

    @property
    def _checkpoint_root(self):return self.config.path.parent/(self.config.path.name+'.checkpoints')

    def _publish(self):
        dirty=self._memory._dirty
        self._memory._sync()
        if self._graph_dirty and not dirty:self._memory.engine.freeze(self.tick)
        self._graph_dirty=False

    def checkpoint(self):
        """Optional immutable cache; SQLite remains authoritative on stale/corrupt cache."""
        with self._transaction():
            self._publish()
            root=self._checkpoint_root;root.mkdir(exist_ok=True)
            generation='generation-'+uuid.uuid4().hex;directory=root/generation;directory.mkdir()
            try:
                self._memory.engine.save_checkpoint(str(directory/'graph.csr'))
                np.save(directory/'embeddings.npy',self._memory.store.embeddings,allow_pickle=False)
                size=sum(f.stat().st_size for f in directory.iterdir())
                if size>self.config['storage']['max_checkpoint_bytes']:raise QuotaExceeded('Checkpoint byte budget exceeded')
                hashes={}
                for f in directory.iterdir():
                    with f.open('rb') as stream:os.fsync(stream.fileno())
                    hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
                manifest={'version':1,'generation':generation,'revision':int(self._get('revision')),
                          'capacity':self._memory.engine.stats()['capacity'],'learning':self.config.learning_fingerprint,
                          'tick':self.tick,'ids':list(self._map),'hashes':hashes}
                fd=os.open(directory,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
                temp=root/('manifest-'+uuid.uuid4().hex+'.tmp')
                with temp.open('w') as stream:
                    json.dump(manifest,stream);stream.flush();os.fsync(stream.fileno())
                os.replace(temp,root/'current.json')
                fd=os.open(root,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
            except Exception:
                shutil.rmtree(directory,ignore_errors=True);raise
            # Previous mappings remain valid on POSIX after unlink. Only CLI-owned generations.
            for old in root.glob('generation-*'):
                if old!=directory:shutil.rmtree(old)
            return {'revision':manifest['revision'],'bytes':size,'generation':generation,'path':str(root)}

    def compact(self):
        # FULL auto-vacuum already reclaims trailing free pages after deletions.
        # Exclusive SQLite locking also prevents another CLI from writing during VACUUM.
        with self._lock:
            with self._transaction():pass
            self.db.execute('VACUUM')
            return self.stats()

    def close(self):
        with self._lock:
            if self._memory:self._memory.close();self._memory=None
            self.db.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


class FrameworkAgent:
    """Duck-typed adapter for EvidencePipeline; any AnswerProvider is supported."""
    def __init__(self,memory,provider):self.memory=memory;self.provider=provider
    def observe(self,text,writer=''):return self.memory.remember(text,writer)['id']
    def context(self,question,mode='associative',budget=8,hops=2):
        return self.memory.context(question,mode,budget,hops)
    def ask(self,question,mode='associative',budget=8,hops=2):
        records,trace=self.context(question,mode,budget,hops)
        return {'question':question,'response':asdict(self.provider.answer(question,records)),
                'evidence':[asdict(e) for e in records],'retrieval':trace}
