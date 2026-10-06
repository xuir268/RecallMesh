"""LoCoMo retrieval evaluation, NOT answer F1/BLEU or HeLa-Mem reproduction.

Graph training replays dialogue turns as recall queries after remembering them.
No benchmark questions, answers, evidence labels, or supplied summaries enter
training. Evaluation calls read-only retrieval, never Memory.recall().
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import re
import time
import numpy as np
from assoc_mem import Memory, HashEmbedder
from assoc_mem.retrieve import recall, seed

CATEGORIES = {1:'multi-hop', 2:'temporal', 3:'open-domain', 4:'single-hop'}
STOP = set('a an the is are was were be been to of and or in on at for with it its that this did does do what when where who how which would could should'.split())

def tokenize(text):
    return [x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in STOP]

class BM25:
    def __init__(self, texts):
        counts = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.asarray([sum(c.values()) for c in counts], float)
        self.avg = max(float(self.lengths.mean()), 1)
        self.postings = defaultdict(list)
        for i, c in enumerate(counts):
            for term, freq in c.items(): self.postings[term].append((i,freq))
    def rank(self, query, k):
        n = len(self.lengths)
        scores = np.zeros(n)
        for term in set(tokenize(query)):
            posting = self.postings.get(term, [])
            if not posting: continue
            ids, tf = np.asarray(posting).T
            idf = np.log(1 + (n - len(ids) + .5)/(len(ids)+.5))
            denom = tf + 1.2*(.25 + .75*self.lengths[ids]/self.avg)
            scores[ids] += idf * tf * 2.2 / denom
        return np.argsort(-scores, kind='stable')[:k] + 1

def evidence_ids(evidence):
    return sorted({m for item in evidence for m in re.findall(r'D\d+:\d+', str(item))})

def measure(gold, retrieved):
    if not gold: return None
    found = set(retrieved) & set(gold)
    return {'evidence_recall':len(found)/len(gold),
            'any_evidence':int(bool(found)), 'all_evidence':int(len(found)==len(gold))}

def aggregate(rows):
    scored = [r for r in rows if r['metrics'] is not None]
    out = {'questions':len(rows),'scored_questions':len(scored)}
    for key in ['evidence_recall','any_evidence','all_evidence']:
        out[key] = float(np.mean([r['metrics'][key] for r in scored])) if scored else None
    times=[r['latency_ms'] for r in rows]
    out['latency_ms_median']=float(np.median(times)) if times else None
    out['latency_ms_p95']=float(np.percentile(times,95)) if times else None
    out['mean_returned_turns']=float(np.mean([len(r['retrieved']) for r in rows])) if rows else None
    return out

def run(args):
    started=time.perf_counter()
    raw=args.data.read_bytes(); data=json.loads(raw)
    all_rows=[]; conv_reports=[]
    for ci, sample in enumerate(data):
        ingest_start=time.perf_counter()
        conv=sample['conversation']
        sessions=sorted((int(k.split('_')[1]),v) for k,v in conv.items()
                        if re.fullmatch(r'session_\d+',k))
        records=[]; mapping={}; inverse={}; texts=[]
        with Memory(capacity=1<<18,embedder=HashEmbedder(128),background=False,retrieval="embedding") as mem:
            for session, turns in sessions:
                mem.tick=session
                date=conv.get(f'session_{session}_date_time','')
                for turn in turns:
                    # Text-only dataset; captions are supplied annotations, not image inference.
                    text=f"[{date}] {turn['speaker']}: {turn['text']}"
                    if turn.get('blip_caption'): text+=f" [Image caption: {turn['blip_caption']}]"
                    nid=mem.remember(text,writer=turn['speaker'])
                    mapping[turn['dia_id']]=nid;inverse[nid]=turn['dia_id'];texts.append(text)
                    # Controlled usage simulation: turn-driven recall, not held-out QA.
                    mem.recall(turn['text'],budget=8,k_seed=8,hops=0)
                mem.flush()
            stats=mem.stats();ingest_seconds=time.perf_counter()-ingest_start
            bm25=BM25(texts)
            engine_lat=[]
            for qi, qa in enumerate(sample['qa']):
                if qa['category'] not in CATEGORIES: continue
                question=qa['question'];gold=evidence_ids(qa.get('evidence',[]))
                q=mem.embedder.encode([question])[0]
                for method in ['hash_only','hash_graph','bm25_only']:
                    t=time.perf_counter()
                    if method=='hash_only': ids,_=seed(mem.store.embeddings,q,args.k)
                    elif method=='hash_graph':
                        found,_=recall(mem.engine,mem.store,q,tick=mem.tick,k_seed=8,hops=2,budget=args.k)
                        ids=[node.id for node in found]
                    else: ids=bm25.rank(question,args.k)
                    elapsed=(time.perf_counter()-t)*1000
                    retrieved=[inverse[int(i)] for i in ids]
                    all_rows.append({'conversation':sample.get('sample_id',ci),'question_index':qi,
                        'category':CATEGORIES[qa['category']],'method':method,'question':question,
                        'answer':qa.get('answer'),'gold_evidence':gold,
                        'unresolved_evidence':[g for g in gold if g not in mapping],
                        'retrieved':retrieved,'metrics':measure(gold,retrieved),'latency_ms':elapsed})
                # Isolate C++ spreading latency from embedding and SQLite.
                ids,w=seed(mem.store.embeddings,q,8)
                t=time.perf_counter()
                mem.engine.activate(ids,w,hops=2,tick=mem.tick,cap=args.k*4)
                engine_lat.append((time.perf_counter()-t)*1000)
            after=mem.engine.stats()
            assert after['inserts']==stats['inserts'] and after['updates']==stats['updates'], 'Evaluation mutated graph'
            cr={'conversation':sample.get('sample_id',ci),'turns':len(texts),'sessions':len(sessions),
                'ingest_seconds':ingest_seconds,'stats':stats,'snapshot_edges':mem.engine.edge_count,
                'activation_ms_median':float(np.median(engine_lat)),
                'activation_ms_p95':float(np.percentile(engine_lat,95))}
            conv_reports.append(cr)
            print(f"{ci+1}/10: {len(texts)} turns, {stats['live_edges']} edges, {stats['rejected_full']} rejected; {ingest_seconds:.2f}s ingest",flush=True)
    report={'evaluation':'LoCoMo evidence retrieval only; not answer F1/BLEU',
        'dataset':{'source':'https://github.com/snap-research/locomo/blob/main/data/locomo10.json',
        'sha256':hashlib.sha256(raw).hexdigest(),'conversations':len(data),
        'turns':sum(c['turns'] for c in conv_reports),
        'questions_categories_1_4':len(all_rows)//3,
        'excluded_category_5':sum(q['category']==5 for c in data for q in c['qa'])},
        'settings':{'k':args.k,'embedding':'HashEmbedder(dim=128), deterministic token hash, not semantic',
        'graph_seed_count':8,'graph_hops':2,'lambda':.05,'eta':1.,'floor':.001,
        'tick':'one per dataset session','training':'After each turn, recall that turn against prefix; 8 seeds, hops=0; flush at session end',
        'leakage_control':'No QA or summaries used in ingestion; no reinforcement during evaluation',
        'bm25':'lexical diagnostic comparator, k1=1.2,b=0.75; no semantic model',
        'metrics':'macro evidence recall, any/all annotated evidence retrieved; empty gold omitted from denominator; unresolved IDs retained as misses'},
        'platform':platform.platform(),'seconds':time.perf_counter()-started,
        'methods':{},'conversations':conv_reports}
    for method in ['hash_only','hash_graph','bm25_only']:
        rows=[r for r in all_rows if r['method']==method]
        report['methods'][method]={'overall':aggregate(rows),'categories':{
            c:aggregate([r for r in rows if r['category']==c]) for c in CATEGORIES.values()}}
    report['unresolved_evidence_questions']=sum(bool(r['unresolved_evidence']) for r in all_rows if r['method']=='hash_only')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
    lines=['# LoCoMo retrieval evaluation','', '**This is not a reproduction of HeLa-Mem answer F1/BLEU.**','',
        f"Dataset: {report['dataset']['conversations']} conversations, {report['dataset']['turns']} turns, {len(all_rows)//3} non-adversarial questions.",
        '',f'Budget: {args.k} dialogue turns. Four questions without evidence are excluded from evidence-score denominators.',
        '', '| Method | Evidence recall | Any evidence | All evidence | Median query ms | P95 query ms |',
        '|---|---:|---:|---:|---:|---:|']
    for method,obj in report['methods'].items():
        r=obj['overall'];lines.append(f"| {method} | {r['evidence_recall']:.2%} | {r['any_evidence']:.2%} | {r['all_evidence']:.2%} | {r['latency_ms_median']:.3f} | {r['latency_ms_p95']:.3f} |")
    lines+=['','## By category (macro evidence recall)','', '| Category | Questions scored | Hash | Hash + graph | BM25 |','|---|---:|---:|---:|---:|']
    for c in CATEGORIES.values():
        r=[report['methods'][m]['categories'][c] for m in report['methods']]
        lines.append(f"| {c} | {r[0]['scored_questions']} | {r[0]['evidence_recall']:.2%} | {r[1]['evidence_recall']:.2%} | {r[2]['evidence_recall']:.2%} |")
    lines+=['','## Protocol and limits','',
        '- Graph construction replays conversation text only, before reading held-out questions. Graph is frozen for QA.',
        '- Uses the project’s token-hash embedding stub, not a learned semantic encoder.',
        '- No answer generator, reflective distillation, semantic memory store, or paper-exact dual-path ranking.',
        '- Evidence recall measures retrieved source turns, not whether an LLM can answer correctly.',
        '- Budgets are dialogue-turn counts, not matched token budgets; no tokenizer token-cost claims.',
        '- Latencies are a single local run, exclude query embedding, and include retrieval/SQLite for graph; separately measured C++ activation latency is in summary.json.',
        '- Image captions included when present; images not downloaded. No dataset summaries supplied to retrieval.',
        f"- {report['unresolved_evidence_questions']} questions contain evidence IDs absent from their conversation; retained as misses.",
        '- All per-question retrieved IDs and gold evidence are recorded in predictions.jsonl for audit.',
        '',f"Dataset SHA256: `{report['dataset']['sha256']}`",f"Elapsed: {report['seconds']:.2f} seconds."]
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['methods'],indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('benchmarks/data/locomo10.json'))
    parser.add_argument('--output',type=Path,default=Path('benchmarks/results/locomo'))
    parser.add_argument('--k',type=int,default=16)
    args=parser.parse_args()
    if args.k<=0: parser.error('--k must be positive')
    run(args)
