"""Preregistered entity-routing retrieval test. No model calls or answer labels."""
import hashlib,json
from pathlib import Path
import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.entity import EntityAwareRetriever
from assoc_mem.lexical import LexicalIndex,AssociativeRetriever
from natural_coreference import turns
from locomo_retrieval import measure
from longitudinal import interval
ROOT=Path(__file__).resolve().parents[1]

def main():
    raw=(ROOT/'benchmarks/data/locomo10.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4'
    cases=json.loads((ROOT/'benchmarks/protocols/natural-reference-qa-v1.json').read_text())['membership']
    rows=[];traces=[]
    for sample in json.loads(raw):
        selected=[c for c in cases if c['conversation']==sample['sample_id']]
        if not selected:continue
        sessions=list(turns(sample));names=sorted({t['speaker'] for _,s in sessions for t in s})
        graph=EdgeStore(1<<18);index=LexicalIndex();adj=AssociativeRetriever(graph,index);entity=EntityAwareRetriever(adj,names);inverse={}
        for tick,session in sessions:
            previous=None
            for turn in session:
                body=turn['text']
                if turn.get('blip_caption'):body+=' [Image caption: '+turn['blip_caption']+']'
                text='['+sample['conversation'].get(f'session_{tick}_date_time','')+'] '+turn['speaker']+': '+body
                nid=index.add(text);inverse[nid]=turn['dia_id'];entity.add(nid,body,turn['speaker'])
                if previous:graph.reinforce(np.array([nid],np.uint32),np.array([previous],np.uint32),tick)
                previous=nid
        graph.freeze(tick);before=graph.stats()
        for c in selected:
            q=sample['qa'][c['question_index']]['question']
            packets={'bm25':adj.search(q,tick,16,0),'adjacency':adj.search(q,tick,16,.4),'entity_topic_only':entity.search(q,tick,use_adjacency=False),'entity_topic_plus_adjacency':entity.search(q,tick)}
            metrics={arm:measure(c['gold'],[inverse[int(n)] for n in ids]) for arm,ids in packets.items()}
            rows.append({'id':c['id'],'conversation':sample['sample_id'],'metrics':metrics})
            traces.append({'id':c['id'],'explanation':entity.explain(q,tick),'packets':{a:[inverse[int(n)] for n in ids] for a,ids in packets.items()}})
        assert graph.stats()==before
    summary={a:{m:float(np.mean([r['metrics'][a][m] for r in rows])) for m in ['evidence_recall','any_evidence','all_evidence']} for a in rows[0]['metrics']}
    paired=[]
    for row in rows:paired.append({'conversation':row['conversation'],**{a:row['metrics'][a]['evidence_recall'] for a in summary}})
    delta=summary['entity_topic_plus_adjacency']['evidence_recall']-summary['adjacency']['evidence_recall'];ci=interval(paired,'entity_topic_plus_adjacency','adjacency')
    result={'protocol':json.loads((ROOT/'benchmarks/protocols/entity-aware-v1.json').read_text()),'n':len(rows),'summary':summary,'entity_minus_adjacency':delta,'cluster95':ci,'positive_signal':delta>=.02 and ci[0]>0,'per_question':rows,'wins':sum(r['entity_topic_plus_adjacency']>r['adjacency'] for r in paired),'losses':sum(r['entity_topic_plus_adjacency']<r['adjacency'] for r in paired)}
    (ROOT/'benchmarks/protocols/entity-aware-results-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    out=ROOT/'benchmarks/results/entity-aware';out.mkdir(parents=True,exist_ok=True);(out/'traces.json').write_text(json.dumps(traces,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['protocol','per_question']},indent=2))

if __name__=='__main__':main()
