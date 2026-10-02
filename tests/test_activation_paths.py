import numpy as np
import pytest
from assoc_mem import EdgeStore
from assoc_mem.lexical import LexicalIndex, AssociativeRetriever

def arr(x,dtype=np.uint32):return np.array(x,dtype)

def test_merged_frontier_matches_linear_oracle_and_caches_rows():
    e=EdgeStore(128,lambda_=0)
    edges=[(1,2),(1,3),(2,4),(3,4)]
    e.reinforce(arr([a for a,b in edges]),arr([b for a,b in edges]),0);e.freeze(0)
    seed=arr([1]);weights=arr([1],np.float32)
    ids,scores,new,traces=e.activate_profile(seed,weights,hops=4,cutoff=0,cap=10,trace=True)
    old_ids,old_scores,old,_=e.activate_profile(seed,weights,hops=4,cutoff=0,cap=10,optimized=False)
    matrix=np.zeros((5,5));
    for a,b in edges:matrix[a,b]=matrix[b,a]=1
    matrix[1:]/=matrix[1:].sum(axis=1,keepdims=True)
    current=np.array([0,1,0,0,0],float);total=current.copy()
    for _ in range(4):current=current@matrix;total+=current
    np.testing.assert_allclose(scores,total[ids],atol=1e-6)
    np.testing.assert_allclose(scores,[dict(zip(old_ids,old_scores))[i] for i in ids],atol=1e-6)
    assert new['row_visits']<old['row_visits']
    assert new['decay_evaluations']==8
    assert new['merged_entries']>0
    assert any(a==2 and b==4 and hop==2 for a,b,hop,w in traces)

def test_query_cache_resets_after_tick_and_snapshot_change():
    e=EdgeStore(64,lambda_=.5)
    e.reinforce(arr([1]),arr([2]),0)
    e.reinforce(arr([1]),arr([3]),2);e.freeze(2)
    ids,scores,_,_=e.activate_profile(arr([1]),arr([1],np.float32),hops=1,tick=2,cutoff=0)
    assert dict(zip(ids,scores))[2]==pytest.approx(.2)
    e.reinforce(arr([1]),arr([2]),2);e.freeze(2)
    ids,scores,_,_=e.activate_profile(arr([1]),arr([1],np.float32),hops=1,tick=2,cutoff=0)
    assert dict(zip(ids,scores))[2]==pytest.approx(1.25/2.25)

def test_actual_two_hop_path_disappears_when_bridge_removed():
    def retrieve(with_bridge):
        e=EdgeStore(64,lambda_=0)
        e.reinforce(arr([1]),arr([2]),0)
        if with_bridge:e.reinforce(arr([2]),arr([3]),0)
        e.freeze(0)
        return e.activate_profile(arr([1]),arr([1],np.float32),hops=2,cutoff=0,trace=True)
    ids,_,_,trace=retrieve(True)
    assert 3 in ids and (1,2,1) in [tuple(t[:3]) for t in trace] and (2,3,2) in [tuple(t[:3]) for t in trace]
    assert 3 not in retrieve(False)[0]

def test_nonzero_cutoff_uses_aggregated_hop_mass():
    e=EdgeStore(128,lambda_=0)
    e.reinforce(arr([1,1,2,3,4]),arr([2,3,4,4,5]),0);e.freeze(0)
    ids,_,_,_=e.activate_profile(arr([1]),arr([1],np.float32),hops=3,cutoff=.1)
    old,_,_,_=e.activate_profile(arr([1]),arr([1],np.float32),hops=3,cutoff=.1,optimized=False)
    assert 5 in ids and 5 not in old
