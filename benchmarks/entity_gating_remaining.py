"""One-shot registered gating contrast on remaining entity-study questions."""
import hashlib
import json
from pathlib import Path
import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.entity import EntityAwareRetriever
from assoc_mem.lexical import LexicalIndex, AssociativeRetriever
from natural_coreference import turns
from locomo_retrieval import measure
from longitudinal import interval
ROOT = Path(__file__).resolve().parents[1]

def main():
    target = ROOT/'benchmarks/protocols/entity-gating-remaining-results-v1.json'
    if target.exists(): raise SystemExit('Results already exist; refusing another execution.')
    protocol = json.loads((ROOT/'benchmarks/protocols/entity-gating-remaining-v1.json').read_text())
    raw = (ROOT/'benchmarks/data/locomo10.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == protocol['source_sha256']
    old = json.loads((ROOT/'benchmarks/protocols/natural-reference-qa-v1.json').read_text())['membership']
    assert not {c['id'] for c in old} & {c['id'] for c in protocol['membership']}
    rows = []
    for sample in json.loads(raw):
        cases = [c for c in protocol['membership'] if c['conversation']==sample['sample_id']]
        if not cases: continue
        sessions = list(turns(sample)); names = sorted({t['speaker'] for _,s in sessions for t in s})
        graph = EdgeStore(protocol['settings']['capacity']); index = LexicalIndex()
        adjacency = AssociativeRetriever(graph,index); entity = EntityAwareRetriever(adjacency,names)
        inverse = {}
        for tick,session in sessions:
            previous = None
            for turn in session:
                body = turn['text']
                if turn.get('blip_caption'): body += ' [Image caption: '+turn['blip_caption']+']'
                text = '['+sample['conversation'].get(f'session_{tick}_date_time','')+'] '+turn['speaker']+': '+body
                nid = index.add(text); inverse[nid] = turn['dia_id']; entity.add(nid,body,turn['speaker'])
                if previous: graph.reinforce(np.array([nid],np.uint32),np.array([previous],np.uint32),tick)
                previous = nid
        graph.freeze(tick); before = graph.stats()
        for c in cases:
            query = sample['qa'][c['question_index']]['question']
            explanation = entity.explain(query,tick)
            assert explanation['mode']=='entity_topic_plus_adjacency' and len(explanation['entities'])==1
            _,activation,_,_,_ = adjacency.components(query,tick,hops=2,seed_count=8)
            ungated = adjacency.rank(entity.topic.scores(explanation['topic_query']),activation,16,.4)
            gated = [r['node'] for r in explanation['results']]
            metrics = {arm:measure(c['gold'],[inverse[int(n)] for n in ids]) for arm,ids in [('ungated',ungated),('gated',gated)]}
            rows.append({'id':c['id'],'conversation':sample['sample_id'],'metrics':metrics,'delta':metrics['gated']['evidence_recall']-metrics['ungated']['evidence_recall']})
        assert before==graph.stats()
    assert len(rows)==protocol['n']
    summary = {arm:{m:float(np.mean([r['metrics'][arm][m] for r in rows])) for m in ['evidence_recall','all_evidence','any_evidence']} for arm in ['ungated','gated']}
    paired = [{'conversation':r['conversation'],**{a:r['metrics'][a]['evidence_recall'] for a in summary}} for r in rows]
    delta = float(np.mean([r['delta'] for r in rows])); ci = interval(paired,'gated','ungated')
    result = {'protocol':'entity-gating-remaining-v1.json','n':len(rows),'exposure':protocol['exposure'],'summary':summary,'delta':delta,'cluster95':ci,'meets_2pp':delta>=.02,'positive_lower_bound':ci[0]>0,'criterion_passed':delta>=.02 and ci[0]>0,'wins':sum(r['delta']>0 for r in rows),'ties':sum(r['delta']==0 for r in rows),'losses':sum(r['delta']<0 for r in rows),'conversation_deltas':{g:{'n':sum(r['conversation']==g for r in rows),'mean_delta':float(np.mean([r['delta'] for r in rows if r['conversation']==g]))} for g in sorted({r['conversation'] for r in rows})},'per_question':rows,'model_calls':0}
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_question'},indent=2))
if __name__=='__main__': main()
