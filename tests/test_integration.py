from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from assoc_mem import EdgeStore, Memory, HashEmbedder
from assoc_mem.coactivate import CoActivation
from assoc_mem.retrieve import unexplored, seed

def u(xs): return np.asarray(xs, np.uint32)
def f(xs): return np.asarray(xs, np.float32)

def test_snapshot_queries_and_custom_decay():
    e = EdgeStore(64, lambda_=0.5)
    e.reinforce(u([1, 1, 2]), u([2, 3, 4]), 10)
    assert e.neighbors(1)[0].size == 0
    e.freeze(10)
    np.testing.assert_allclose(e.weights_for_pairs(u([1, 2, 1]), u([2, 1, 4]), 11), [0.5, 0.5, 0])
    assert e.edge_mass_along(u([3, 1, 2, 4]), 10) == 3
    assert e.neighbors(1, tick=10)[0].tolist() == [2, 3]
    ids, scores = e.activate(u([1]), f([1]), hops=1, cutoff=0, tick=10)
    got=dict(zip(ids.tolist(), scores.tolist()))
    assert got == {1: 1, 2: 0.5, 3: 0.5}
    e.freeze(100)
    assert e.edge_count == 0 and e.node_count == 0

def test_stale_tick_and_long_age():
    e=EdgeStore(64)
    e.reinforce(u([1]),u([2]),101)
    e.reinforce(u([1]),u([2]),100)
    assert e.peek(1,2,101)==2
    e.reinforce(u([1]*20),u([3]*20),0)
    e.freeze(136)
    assert e.weights_for_pairs(u([1]),u([3]),136)[0] > 0.001

def test_concurrent_reinforcement():
    e=EdgeStore(64,lambda_=0)
    a=u([1]*50000); b=u([2]*50000)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _:e.reinforce(a,b,0),range(8)))
    assert e.peek(1,2)==400000
    assert e.stats()['inserts']==1
    assert e.stats()['rejected_full']==0

def test_walk_and_binding_validation():
    e=EdgeStore(64,lambda_=0)
    e.reinforce(u([1,2]),u([2,3]),0)
    e.attach_embeddings(f([[0,0],[0,0],[1,0],[2,0]]))
    e.freeze(0)
    assert e.walk_directional(1,f([1,0])).tolist()==[1,2,3]
    with pytest.raises(ValueError): e.walk_directional(1,f([1]))
    with pytest.raises(ValueError): e.activate(u([1,2]),f([1]))
    with pytest.raises(ValueError): e.reinforce(u([1]),u([]),0)

def test_queue_preserves_event_ticks():
    e=EdgeStore(64,lambda_=0.5)
    q=CoActivation(e)
    q.record([1,2],0); q.record([1,2],2)
    q.flush(2)
    assert e.peek(1,2,2)==1.25

def test_unexplored_subset_and_empty_inputs():
    e=EdgeStore(64)
    emb=f([[0,0],[1,0],[1,0],[0,1],[1,0]])
    pairs=unexplored(e,emb,u([1,2]),k=2)
    assert pairs.ndim==2 and pairs.shape[1]==2
    assert seed(emb,f([1,0]),0)[0].size==0

def test_memory_pipeline_and_supersede():
    with Memory(capacity=128,embedder=HashEmbedder(32)) as m:
        a=m.remember('cat food')
        b=m.remember('cat sleep')
        m.remember('dog walk')
        assert m.store.embeddings.shape[0]==4
        assert m.recall('cat',k_seed=3)
        m.flush()
        assert m.engine.edge_count>0
        assert m.unexplored('cat',k=2).shape[1]==2
        assert len(m.axis([a,b])[1])==2
        assert len(m.sectors(a)[0])==12
        m.advance()
        new=m.supersede(a,'cat dinner')
        assert m.store.get(a).valid_to==1
        assert new in [n.id for n in m.recall('cat dinner',k_seed=4)]
        assert a not in [n.id for n in m.recall('cat dinner',k_seed=4)]

def test_background_flush():
    with Memory(capacity=128,background=True) as m:
        m.remember('shared cat');m.remember('shared dog')
        for _ in range(10): m.recall('shared')
        m.flush()
        assert m.engine.peek(1,2,m.tick)==10
