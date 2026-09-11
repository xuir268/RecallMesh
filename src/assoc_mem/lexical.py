"""Inspectable lexical seeding plus optional associative reranking.

No answer generation. Graph construction is controlled by the caller; search
never reinforces edges. The same lexical index supports graph-off ablations.
"""
from collections import Counter, defaultdict
import re
import numpy as np

STOP = set('a an the is are was were be been to of and or in on at for with it its that this did does do what when where who how which would could should'.split())

def tokenize(text):
    return [t for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in STOP]

def stable_topk(scores, k):
    """Descending score, ascending ID ties, without sorting the entire array."""
    k=min(max(0,k),len(scores))
    if k==0:return np.empty(0,np.intp)
    if k==len(scores):return np.argsort(-scores,kind='stable')
    boundary=np.partition(scores,len(scores)-k)[len(scores)-k]
    above=np.flatnonzero(scores>boundary)
    tied=np.flatnonzero(scores==boundary)[:k-len(above)]
    selected=np.concatenate((above,tied))
    return selected[np.lexsort((selected,-scores[selected]))]

class LexicalIndex:
    """Incremental BM25; returned IDs start at 1 to match NodeStore."""
    def __init__(self):
        self.postings=defaultdict(list)
        self.lengths=[]
        self.total_length=0
        self._length_array=np.empty(16,np.float32)
    def add(self,text):
        counts=Counter(tokenize(text)); i=len(self.lengths)
        length=sum(counts.values());self.lengths.append(length);self.total_length+=length
        if i==len(self._length_array):
            grown=np.empty(max(16,i*2),np.float32)
            grown[:i]=self._length_array
            self._length_array=grown
        self._length_array[i]=length
        for t,n in counts.items():self.postings[t].append((i,n))
        return i+1
    def scores(self,query):
        n=len(self.lengths);out=np.zeros(n,np.float32)
        if not n:return out
        lengths=self._length_array[:n]
        avg=max(self.total_length/n,1)
        for term in sorted(set(tokenize(query))):
            posting=self.postings.get(term,[])
            if not posting:continue
            ids,tf=np.asarray(posting).T
            idf=np.log(1+(n-len(ids)+.5)/(len(ids)+.5))
            out[ids]+=idf*tf*2.2/(tf+1.2*(.25+.75*lengths[ids]/avg))
        return out

class AssociativeRetriever:
    def __init__(self,engine,index):
        self.engine=engine;self.index=index
    def components(self,query,tick,hops=2,profile=False,optimized=True,cutoff=.001,seed_count=8):
        lexical=self.index.scores(query)
        order=stable_topk(lexical,max(0,seed_count))
        chosen=order[lexical[order]>0][:max(0,seed_count)]
        seeds=(chosen+1).astype(np.uint32)
        weights=lexical[chosen].astype(np.float32)
        if weights.size:weights/=weights.sum()
        assoc=np.zeros_like(lexical)
        metrics={};transitions=[]
        if seeds.size:
            if profile:
                ids,values,metrics,transitions=self.engine.activate_profile(
                    seeds,weights,hops=hops,cutoff=cutoff,tick=tick,cap=len(lexical),
                    optimized=optimized,trace=True)
            else:
                ids,values=self.engine.activate(seeds,weights,hops=hops,cutoff=cutoff,
                    tick=tick,cap=len(lexical))
            valid=(ids>0)&(ids<=len(lexical));assoc[ids[valid]-1]=values[valid]
            # Remove starting scores, retaining only graph-propagated activation.
            assoc[chosen]=np.maximum(assoc[chosen]-weights,0)
        return lexical,assoc,seeds,metrics,transitions
    @staticmethod
    def rank(lexical,assoc,budget=16,graph_weight=.1):
        if budget<=0:return np.empty(0,np.uint32)
        base=lexical/max(float(lexical.max(initial=0)),1e-12)
        graph=assoc/max(float(assoc.max(initial=0)),1e-12)
        combined=base+graph_weight*graph
        return (stable_topk(combined,budget)+1).astype(np.uint32)
    def search(self,query,tick,budget=16,graph_weight=.4,hops=2,seed_count=8):
        if graph_weight==0:
            lexical=self.index.scores(query)
            return self.rank(lexical,np.zeros_like(lexical),budget,0)
        lexical,assoc,_,_,_=self.components(query,tick,hops=hops,seed_count=seed_count)
        return self.rank(lexical,assoc,budget,graph_weight)

    def explain(self,query,tick,budget=16,graph_weight=.4,hops=2,seed_count=8):
        lexical,assoc,seeds,metrics,transitions=self.components(
            query,tick,hops=hops,seed_count=seed_count,profile=True)
        ids=self.rank(lexical,assoc,budget,graph_weight)
        baseline=set(self.rank(lexical,assoc,budget,0).tolist())
        layer={int(n):[int(n)] for n in seeds};paths={int(n):[int(n)] for n in seeds}
        by_hop=defaultdict(list)
        for u,v,h,w in transitions:by_hop[h].append((u,v,w))
        for _,arcs in sorted(by_hop.items()):
            next_layer={};strongest={}
            for u,v,w in arcs:
                if u in layer and w>strongest.get(v,-1):
                    next_layer[v]=layer[u]+[v];strongest[v]=w
            for v,path in next_layer.items():paths.setdefault(v,path)
            layer=next_layer
        return {'seeds':seeds.tolist(),'operations':metrics,'results':[
            {'node':int(n),'lexical_score':float(lexical[n-1]),
             'association_score':float(assoc[n-1]),'added_by_graph':int(n) not in baseline,
             'path':paths.get(int(n),[])} for n in ids]}

    def connected(self, query, tick, budget=16, hops=2, graph_weight=.4):
        """Experimental: one lexical anchor, whole witness paths within budget.

        Every included non-anchor has a traversed path included with it. This
        preserves graph connectivity, not semantic entailment. Multi-anchor
        questions can need the existing ranked policy instead.
        """
        if budget<=0:return {'seeds':[],'results':[],'operations':{}}
        lex,assoc,seeds,metrics,arcs=self.components(query,tick,hops=hops,seed_count=1,profile=True)
        if not len(seeds):return {'seeds':[],'results':[],'operations':metrics}
        root=int(seeds[0]);layer={root:[root]};paths={root:[root]}
        by_hop=defaultdict(list)
        for u,v,h,w in arcs:by_hop[h].append((u,v,w))
        for _,edges in sorted(by_hop.items()):
            nxt={};strength={}
            for u,v,w in edges:
                if u in layer and v not in layer[u] and w>strength.get(v,-1):
                    nxt[v]=layer[u]+[v];strength[v]=w
            for v,path in nxt.items():paths.setdefault(v,path)
            layer=nxt
        order=self.rank(lex,assoc,len(lex),graph_weight)
        selected=[root];present={root};selected_paths={root:[root]}
        for target in order:
            path=paths.get(int(target))
            if path is None:continue
            extra=[v for v in path if v not in present]
            if len(selected)+len(extra)>budget:continue
            for i,v in enumerate(path):
                if v not in present:selected_paths[v]=path[:i+1]
            selected.extend(extra);present.update(extra)
        return {'seeds':[root],'operations':metrics,'results':[
            {'node':n,'lexical_score':float(lex[n-1]),'association_score':float(assoc[n-1]),
             'path':selected_paths[n]} for n in selected]}
