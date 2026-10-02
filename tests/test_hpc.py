import numpy as np
from assoc_mem import Memory
from assoc_mem.agent import MemoryAgent
from assoc_mem.lexical import stable_topk,LexicalIndex


def test_partition_topk_preserves_full_sort_and_ties():
    rng=np.random.default_rng(41)
    for n in [0,1,2,17,1000,10000]:
        for scale in [1,10,100000]:
            scores=rng.integers(-scale,scale+1,n).astype(np.float32)
            for k in [0,1,4,16,n,n+3]:
                np.testing.assert_array_equal(stable_topk(scores,k),np.argsort(-scores,kind='stable')[:k])


def test_length_cache_growth_and_reference_bm25():
    index=LexicalIndex()
    for i in range(77):index.add('apple '*((i%5)+1)+' train station')
    assert len(index.lengths)==77
    np.testing.assert_array_equal(index._length_array[:77],index.lengths)
    scores=index.scores('apple');n=77;avg=sum(index.lengths)/n
    expected=[]
    for i in range(n):
        tf=i%5+1;idf=np.log(1+.5/(n+.5))
        expected.append(idf*tf*2.2/(tf+1.2*(.25+.75*index.lengths[i]/avg)))
    np.testing.assert_allclose(scores,expected,rtol=2e-6)


def test_star_does_not_link_unrelated_retrieved_memories():
    with Memory(capacity=128) as m:
        # Prior nodes have no edge. A new observation matches both independently.
        m.remember('alpha amber');m.remember('beta blue')
        a=MemoryAgent(m,None,edge_policy='star')
        current=a.observe('alpha beta amber blue');m.flush()
        assert m.engine.peek(1,2,0)==0
        assert m.engine.peek(1,current,0)>0 and m.engine.peek(2,current,0)>0


def test_connected_selection_contains_actual_witness_paths_and_budget():
    for seed in range(10):
        rng=np.random.default_rng(seed)
        with Memory(capacity=1024) as m:
            a=MemoryAgent(m,None,edge_policy='star')
            for i in range(25):
                a.observe('record '+str(i)+' '+' '.join('term'+str(v) for v in rng.integers(0,8,4)))
            for budget in [0,1,2,4,8]:
                evidence,trace=a.context('term3 term4','connected',budget=budget)
                selected={r['node'] for r in trace.get('results',[])}
                assert len(selected)<=budget and len(evidence)==len(selected)
                for r in trace.get('results',[]):
                    assert r['path'][-1]==r['node'] and set(r['path'])<=selected
                    for x,y in zip(r['path'],r['path'][1:]):assert m.engine.peek(x,y,m.tick)>0
