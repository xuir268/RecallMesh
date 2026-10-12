"""Post-hoc mechanism diagnostics; no tuning, model calls, or fresh-holdout claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.entity import EntityAwareRetriever
from assoc_mem.lexical import LexicalIndex, AssociativeRetriever, stable_topk
from natural_coreference import turns
from locomo_retrieval import measure, evidence_ids
from longitudinal import interval
ROOT = Path(__file__).resolve().parents[1]

def main():
    raw = (ROOT/'benchmarks/data/locomo10.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4'
    membership = json.loads((ROOT/'benchmarks/protocols/natural-reference-qa-v1.json').read_text())['membership']
    rows, fallback = [], []
    for sample in json.loads(raw):
        selected = {c['question_index']: c for c in membership if c['conversation'] == sample['sample_id']}
        if not selected: continue
        sessions = list(turns(sample))
        names = sorted({t['speaker'] for _, session in sessions for t in session})
        graph = EdgeStore(1 << 18)
        index = LexicalIndex(); adj = AssociativeRetriever(graph, index)
        entity = EntityAwareRetriever(adj, names); inverse = {}
        for tick, session in sessions:
            previous = None
            for turn in session:
                body = turn['text']
                if turn.get('blip_caption'): body += ' [Image caption: '+turn['blip_caption']+']'
                text = '['+sample['conversation'].get(f'session_{tick}_date_time','')+'] '+turn['speaker']+': '+body
                nid = index.add(text); inverse[nid] = turn['dia_id']; entity.add(nid, body, turn['speaker'])
                if previous: graph.reinforce(np.array([nid],np.uint32),np.array([previous],np.uint32),tick)
                previous = nid
        graph.freeze(tick); before = graph.stats()
        for qi, qa in enumerate(sample['qa']):
            q = qa['question']; matches = entity._matches(q)
            entities = {m[2] for m in matches}
            if qi not in selected:
                gold = evidence_ids(qa.get('evidence', []))
                if len(entities) == 1 or qa.get('category') not in (1,2,3,4) or not gold or not set(gold) <= set(inverse.values()): continue
                base = adj.search(q,tick); routed = entity.search(q,tick)
                assert np.array_equal(base,routed)
                fallback.append({'id':f"{sample['sample_id']}:q{qi}",'conversation':sample['sample_id'],'group':'zero_names' if not entities else 'multiple_names','identical_packet':True,'metrics':measure(gold,[inverse[int(n)] for n in base])})
                continue
            c = selected[qi]; stripped = q
            for start,end,_ in sorted(matches,reverse=True): stripped = stripped[:start]+' '+stripped[end:]
            lexical, activation, _, _, _ = adj.components(q,tick)
            packets = {'adjacency':adj.rank(lexical,activation,16,.4),
                       'adjacency_stripped_query':adj.search(stripped,tick),
                       'strip_lexical_keep_original_seeds':adj.rank(index.scores(stripped),activation,16,.4)}
            candidates = np.array(sorted(set().union(*entity.roles[next(iter(entities))].values())),np.intp)-1
            for rewrite, query in [('original',q),('stripped',stripped)]:
                scores = entity.topic.scores(query)
                packets[f'body_{rewrite}_ungated'] = adj.rank(scores,activation,16,.4)
                local = scores[candidates]/max(float(scores[candidates].max(initial=0)),1e-12)+.4*activation[candidates]/max(float(activation[candidates].max(initial=0)),1e-12)
                packets[f'body_{rewrite}_gated'] = (candidates[stable_topk(local,16)]+1).astype(np.uint32)
            assert np.array_equal(packets['body_stripped_gated'],entity.search(q,tick))
            rows.append({'id':c['id'],'conversation':sample['sample_id'],'metrics':{a:measure(c['gold'],[inverse[int(n)] for n in ids]) for a,ids in packets.items()}})
        assert graph.stats() == before
    summary = {a:{m:float(np.mean([r['metrics'][a][m] for r in rows])) for m in ['evidence_recall','all_evidence']} for a in rows[0]['metrics']}
    paired = [{'conversation':r['conversation'],**{a:r['metrics'][a]['evidence_recall'] for a in summary}} for r in rows]
    contrasts = {}
    for left,right in [('adjacency_stripped_query','adjacency'),('body_stripped_gated','body_stripped_ungated'),('body_stripped_ungated','body_original_ungated')]:
        contrasts[left+' minus '+right] = {'delta':summary[left]['evidence_recall']-summary[right]['evidence_recall'],'cluster95':interval(paired,left,right)}
    groups = {g:{'n':sum(r['group']==g for r in fallback),'recall':float(np.mean([r['metrics']['evidence_recall'] for r in fallback if r['group']==g])),'identical_packets':sum(r['group']==g and r['identical_packet'] for r in fallback)} for g in sorted({r['group'] for r in fallback})}
    result = {'design':'Post-hoc fixed diagnostics; existing 100 exposed cases, same seven conversations. Fallback includes all resolvable category 1–4 questions outside cohort with zero/multiple recognized speaker names. Exact packet equality is a delegation check, not active-route generalization. Original graph seeds held fixed in body-index factorial. Gating includes candidate-local normalization. No model calls.','n':len(rows),'summary':summary,'contrasts':contrasts,'fallback':groups,'per_question':rows,'fallback_per_question':fallback}
    (ROOT/'benchmarks/protocols/entity-ablation-results-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_question','fallback_per_question')},indent=2))
if __name__ == '__main__': main()
