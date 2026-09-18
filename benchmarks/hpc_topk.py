"""Reproduce the top-16 selection microbenchmark; includes allocation cost."""
import json,time,statistics
from pathlib import Path
import numpy as np
from assoc_mem.lexical import stable_topk
rows=[];rng=np.random.default_rng(42)
for n in [10000,100000,1000000]:
    scores=rng.random(n,dtype=np.float32);timings={'full_sort':[],'partition_topk':[]}
    for repeat in range(9):
        for method in (['full_sort','partition_topk'] if repeat%2 else ['partition_topk','full_sort']):
            start=time.perf_counter_ns()
            result=np.argsort(-scores,kind='stable')[:16] if method=='full_sort' else stable_topk(scores,16)
            if repeat:timings[method].append((time.perf_counter_ns()-start)/1e6)
    np.testing.assert_array_equal(stable_topk(scores,16),np.argsort(-scores,kind='stable')[:16])
    rows.append({'n':n,**{k:statistics.median(v) for k,v in timings.items()}})
Path('benchmarks/results/hpc/topk.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
