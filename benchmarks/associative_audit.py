"""BM25 vs associative reranking: dev-only selection, held-out conversations.
Performance compares both traversal implementations on identical frozen graphs.
"""
from collections import defaultdict
import hashlib, json, re, time
from pathlib import Path
import numpy as np
from assoc_mem import Memory, EdgeStore
from assoc_mem.lexical import LexicalIndex, AssociativeRetriever
from locomo_retrieval import evidence_ids, measure, CATEGORIES

WEIGHTS=[0.,.05,.1,.2,.4]
OUT=Path('benchmarks/results/associative-audit')

def pct(rows,method):
    x=[r[method]['evidence_recall'] for r in rows if r[method] is not None]
    return float(np.mean(x)) if x else None

def paths_from_trace(seeds,transitions):
    # One actually traversed witness path per reached node, not an explanation
    # for every summed contribution. Reconstruct in hop order from seed states.
    layer={int(n):[int(n)] for n in seeds}; witnesses={}
    by_hop=defaultdict(list)
    for u,v,h,w in transitions:by_hop[h].append((u,v,w))
    for hop,arcs in sorted(by_hop.items()):
        nxt={};strength={}
        for u,v,w in arcs:
            if u not in layer:continue
            if w>strength.get(v,-1):
                nxt[v]=layer[u]+[v];strength[v]=w
        for v,path in nxt.items():witnesses.setdefault(v,path)
        layer=nxt
    return witnesses

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=Path('benchmarks/data/locomo10.json').read_bytes();data=json.loads(source)
    dev=[];test=[];all_trace=[];perf=[];chosen=None;examples=[];graph_stats=[]
    started=time.perf_counter()
    for ci,sample in enumerate(data):
        split='development' if ci<3 else 'held_out'
        if ci==3:
            dev_scores={str(w):pct(dev,str(w)) for w in WEIGHTS}
            chosen=max(WEIGHTS,key=lambda w:(dev_scores[str(w)],-w))
            print('Locked graph weight from development only:',chosen,dev_scores,flush=True)
        with Memory(capacity=1<<18,background=False) as mem:
            index=LexicalIndex();retriever=AssociativeRetriever(mem.engine,index)
            adjacent=EdgeStore(1<<18);learned=EdgeStore(1<<18)
            adjacent_retriever=AssociativeRetriever(adjacent,index)
            learned_retriever=AssociativeRetriever(learned,index)
            provenance=defaultdict(set)
            inverse={};content={};sessions=sample['conversation'];last_tick=0
            for session in sorted(int(k.split('_')[1]) for k in sessions if re.fullmatch(r'session_\d+',k)):
                previous=None;mem.tick=session;last_tick=session
                for turn in sessions[f'session_{session}']:
                    date=sessions.get(f'session_{session}_date_time','')
                    text=f"[{date}] {turn['speaker']}: {turn['text']}"
                    if turn.get('blip_caption'):text+=f" [Image caption: {turn['blip_caption']}]"
                    # Retrieve only already seen turns. No QA/evidence or future text.
                    scores=index.scores(turn['text']);order=np.argsort(-scores,kind='stable')
                    related=(order[scores[order]>0][:3]+1).tolist()
                    nid=mem.remember(text,writer=turn['speaker']);assert index.add(text)==nid
                    inverse[nid]=turn['dia_id'];content[nid]=text
                    active=sorted(set([nid]+related+([previous] if previous is not None else [])))
                    if len(active)>1:
                        pairs=np.asarray([(a,b) for i,a in enumerate(active) for b in active[i+1:]],np.uint32)
                        mem.engine.reinforce(pairs[:,0].copy(),pairs[:,1].copy(),session)
                    if previous is not None:
                        adjacent.reinforce(np.array([previous],np.uint32),np.array([nid],np.uint32),session)
                        provenance[tuple(sorted((previous,nid)))].add('same-session adjacency')
                    group=sorted(set([nid]+related))
                    lp=[(a,b) for i,a in enumerate(group) for b in group[i+1:]]
                    if lp:
                        pairs=np.array(lp,np.uint32)
                        learned.reinforce(pairs[:,0].copy(),pairs[:,1].copy(),session)
                        for pair in lp:provenance[pair].add('turn-driven co-retrieval')
                    # Combined graph also co-activates previous with retrieved memories.
                    for i,a in enumerate(active):
                        for b in active[i+1:]:
                            if (a,b) not in provenance:provenance[(a,b)].add('combined context co-activation')
                    previous=nid
            adjacent.freeze(last_tick);learned.freeze(last_tick)
            mem.flush();before=mem.engine.stats();graph_stats.append(before)
            for qi,qa in enumerate(sample['qa']):
                if qa['category'] not in CATEGORIES:continue
                query=qa['question'];gold=evidence_ids(qa.get('evidence',[]))
                lex,assoc,seeds,counts,transitions=retriever.components(query,last_tick,profile=True)
                base=AssociativeRetriever.rank(lex,assoc,16,0)
                row={'conversation':sample.get('sample_id',ci),'split':split,'question_index':qi,
                     'category':CATEGORIES[qa['category']],'question':query,'gold':gold}
                weights=WEIGHTS if ci<3 else [0.,chosen]
                for weight in weights:
                    ids=retriever.rank(lex,assoc,16,weight)
                    row[str(weight)]=measure(gold,[inverse[int(n)] for n in ids])
                (dev if ci<3 else test).append(row)
                if ci<3:continue
                selected=retriever.rank(lex,assoc,16,chosen)
                for label,ablation in [('adjacency_only',adjacent_retriever),('learned_only',learned_retriever)]:
                    abl_ids=ablation.search(query,last_tick,16,chosen)
                    row[label]=measure(gold,[inverse[int(n)] for n in abl_ids])
                # Profile exact same seeds/graph/tick with cutoff=0 to isolate
                # optimization from the intentionally changed pruning semantics.
                chosen_idx=seeds-1;w=lex[chosen_idx].copy().astype(np.float32)
                if w.size:w/=w.sum()
                timing={True:[],False:[]};counter={};values={}
                if len(seeds):
                    for repeat in range(4):
                        for optimized in ([False,True] if repeat%2==0 else [True,False]):
                            t=time.perf_counter_ns()
                            ids,sc,metrics,_=mem.engine.activate_profile(seeds,w,hops=2,cutoff=0,
                                tick=last_tick,cap=len(lex),optimized=optimized)
                            elapsed=(time.perf_counter_ns()-t)/1e6
                            if repeat:timing[optimized].append(elapsed)
                            counter[optimized]=metrics;values[optimized]=dict(zip(ids.tolist(),sc.tolist()))
                    assert values[True].keys()==values[False].keys()
                    for n,value in values[True].items():
                        assert np.isclose(value,values[False][n],rtol=2e-5,atol=2e-6)
                    perf.append({'old_ms':float(np.median(timing[False])),'new_ms':float(np.median(timing[True])),
                                 'old':counter[False],'new':counter[True]})
                # Equal output budget AND same record loading for end-to-end latency.
                end_times={}
                for method,weight in [('baseline',0.),('associative',chosen)]:
                    ts=[]
                    for _ in range(3):
                        t=time.perf_counter_ns();ids=retriever.search(query,last_tick,16,weight)
                        records=mem.store.many(ids)
                        ts.append((time.perf_counter_ns()-t)/1e6)
                    end_times[method]=float(np.median(ts))
                witnesses=paths_from_trace(seeds,transitions)
                new_ids=[int(n) for n in selected if n not in set(base.tolist())]
                introduced=[]
                for n in new_ids:
                    path=witnesses.get(n,[])
                    # Check actual retained snapshot edges in each reported path.
                    if path:
                        weights_path=mem.engine.weights_for_pairs(np.array(path[:-1],np.uint32),np.array(path[1:],np.uint32),last_tick)
                        assert np.all(weights_path>0)
                    introduced.append({'id':inverse[n],'gold':inverse[n] in gold,
                        'path':[inverse[v] for v in path], 'text':content[n],
                        'path_texts':[content[v] for v in path],
                        'edge_origins':[sorted(provenance[tuple(sorted((a,b)))]) for a,b in zip(path,path[1:])],
                        'graph_score':float(assoc[n-1]),'lexical_score':float(lex[n-1])})
                base_ev=set(inverse[int(n)] for n in base)&set(gold)
                assoc_ev=set(inverse[int(n)] for n in selected)&set(gold)
                record={**row,'baseline':[inverse[int(n)] for n in base],
                    'associative':[inverse[int(n)] for n in selected],
                    'seeds':[inverse[int(n)] for n in seeds],
                    'gained_evidence':sorted(assoc_ev-base_ev),'lost_evidence':sorted(base_ev-assoc_ev),
                    'introduced':introduced,'latency_ms':end_times,'operations':counts}
                all_trace.append(record)
                if record['gained_evidence'] and len(examples)<8:examples.append(record)
            after=mem.engine.stats();assert before['inserts']==after['inserts'] and before['updates']==after['updates']
            print(f'{ci+1}/10 {split}: {len(inverse)} turns, {mem.engine.edge_count} graph edges',flush=True)
    summary={'dataset_sha256':hashlib.sha256(source).hexdigest(),
        'split':'first 3 conversations development; last 7 held out; earlier work already evaluated full dataset, so this is an internal holdout, not an unseen external test',
        'graph_weight_candidates':WEIGHTS,'development_scores':dev_scores,'selected_graph_weight':chosen,
        'settings':{'seed_count':8,'budget_turns':16,'hops':2,'cutoff':.001,
         'lambda':.05,'eta':1.,'graph_training':'per-turn clique of current turn, previous same-session turn, top 3 BM25 prior turns; chronological prefix only',
         'ranking':'normalized BM25 + development-selected weight * normalized propagated score (seed score removed)'},
        'held_out_questions':len(test),'scored_questions':sum(r['0.0'] is not None for r in test),
        'baseline_evidence_recall':pct(test,'0.0'),'associative_evidence_recall':pct(test,str(chosen)),
        'questions_gaining_evidence':sum(bool(r['gained_evidence']) for r in all_trace),
        'questions_losing_evidence':sum(bool(r['lost_evidence']) for r in all_trace),
        'introduced_results':sum(len(r['introduced']) for r in all_trace),
        'introduced_with_verified_path':sum(bool(x['path']) for r in all_trace for x in r['introduced']),
        'rejected_updates':sum(s['rejected_full'] for s in graph_stats),
        'categories':{},'performance':{},'answer_quality':'Not evaluated: no answer model configured',
        'ablations':{label:pct(test,label) for label in ['adjacency_only','learned_only']},
        'seconds':time.perf_counter()-started}
    grouped=defaultdict(list)
    for r in test:
        if r['0.0'] is not None:
            grouped[r['conversation']].append(r[str(chosen)]['evidence_recall']-r['0.0']['evidence_recall'])
    groups=list(grouped.values());rng=np.random.default_rng(42)
    bootstrap=[]
    for _ in range(10000):
        selected_groups=[groups[i] for i in rng.integers(0,len(groups),len(groups))]
        bootstrap.append(sum(map(sum,selected_groups))/sum(map(len,selected_groups)))
    summary['paired_gain_cluster_bootstrap_95_interval']=np.quantile(bootstrap,[.025,.975]).tolist()
    summary['conversation_gains']={k:float(np.mean(v)) for k,v in grouped.items()}
    for c in CATEGORIES.values():
        rows=[r for r in test if r['category']==c]
        summary['categories'][c]={'n':len(rows),'baseline':pct(rows,'0.0'),'associative':pct(rows,str(chosen))}
    for method in ['old','new']:
        arr=[p[method+'_ms'] for p in perf]
        summary['performance'][method]={'median_ms':float(np.median(arr)),'p95_ms':float(np.percentile(arr,95)),
            **{key:sum(p[method][key] for p in perf) for key in ['row_visits','edge_visits','decay_evaluations','merged_entries']}}
    for method in ['baseline','associative']:
        arr=[r['latency_ms'][method] for r in all_trace]
        summary['performance'][method+'_end_to_end']={'median_ms':float(np.median(arr)),'p95_ms':float(np.percentile(arr,95))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'development.json').write_text(json.dumps(dev,indent=2))
    (OUT/'traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_trace))
    (OUT/'performance.json').write_text(json.dumps(perf))
    (OUT/'examples.json').write_text(json.dumps(examples,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
