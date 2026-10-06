"""Controlled mechanism test inspired by HeLa-Mem Figure 5.
Not a real LoCoMo QA result; seed scores and graph history are supplied.
"""
import json
from pathlib import Path
import numpy as np
from assoc_mem import EdgeStore

def run_case():
    # Explicit illustrative memory labels, not fetched benchmark evidence.
    texts={15:'Met Dr. Sarah at the adoption support conference.',
           89:'Dr. Sarah encouraged me to pursue counseling as a career.'}
    engine=EdgeStore(128,lambda_=0.05,eta=1.0)
    # Three historical co-retrieval events, all before evaluation.
    for tick in [1,39,61]:
        engine.reinforce(np.array([15],np.uint32),np.array([89],np.uint32),tick)
    engine.freeze(61)
    seeds=np.array([89],np.uint32); weights=np.array([.82],np.float32)
    baseline,_=engine.activate(seeds,weights,hops=0,tick=61,cap=2)
    nodes,scores=engine.activate(seeds,weights,hops=1,tick=61,cap=2)
    assert 15 not in baseline and 15 in nodes
    return {'kind':'synthetic mechanism check, not benchmark score',
        'question':'Where did you first meet the person who influenced your career choice?',
        'memory_texts':texts,'manually_supplied_seed':89,
        'without_spreading':baseline.tolist(),
        'with_spreading':[{'node':int(n),'score':float(w)} for n,w in zip(nodes,scores)],
        'location_evidence_retrieved':True,
        'generated_answer':None,
        'note':'Current engine uses normalized propagation; this does not reproduce the paper numerical score.'}

if __name__=='__main__':
    result=run_case()
    Path('benchmarks/results/paper_case.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
