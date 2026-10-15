"""Read-only source/packet audit; human interpretations are explicitly separate."""
import json,hashlib
from pathlib import Path
import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.lexical import LexicalIndex,AssociativeRetriever
from assoc_mem.entity import EntityAwareRetriever
from natural_coreference import turns
ROOT=Path(__file__).resolve().parents[1]

def rank(scores,node):
    order=np.lexsort((np.arange(len(scores)),-scores))
    return int(np.flatnonzero(order==node-1)[0])+1

def main():
    raw=(ROOT/'benchmarks/data/locomo10.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4'
    jobs=json.loads((ROOT/'benchmarks/results/grounded-answers/jobs.json').read_text())
    wide={j['case']:j for j in json.loads((ROOT/'benchmarks/results/wide-answers/jobs.json').read_text())}
    missing_cases=[j for j in jobs if not set(j['required'])<={e['id'] for e in j['evidence']}]
    rows=[]
    for sample in json.loads(raw):
        selected=[j for j in missing_cases if j['case'].split(':')[0]==sample['sample_id']]
        if not selected:continue
        sessions=list(turns(sample));names=sorted({t['speaker'] for _,s in sessions for t in s})
        graph=EdgeStore(1<<18);index=LexicalIndex();adj=AssociativeRetriever(graph,index);entity=EntityAwareRetriever(adj,names)
        lookup={};node_by_id={};position={};by_session={}
        for tick,session in sessions:
            previous=None;by_session[tick]=[]
            for pos,turn in enumerate(session):
                body=turn['text']
                if turn.get('blip_caption'):body+=' [Image caption: '+turn['blip_caption']+']'
                nid=index.add('['+sample['conversation'].get(f'session_{tick}_date_time','')+'] '+turn['speaker']+': '+body)
                entity.add(nid,body,turn['speaker']);lookup[turn['dia_id']]=turn;node_by_id[turn['dia_id']]=nid;position[turn['dia_id']]=(tick,pos);by_session[tick].append(turn['dia_id'])
                if previous:graph.reinforce(np.array([nid],np.uint32),np.array([previous],np.uint32),tick)
                previous=nid
        graph.freeze(tick)
        for j in selected:
            packet={e['id'] for e in j['evidence']};large={e['id'] for e in wide[j['case']]['evidence']}
            explanation=entity.explain(j['question'],tick);scores=entity.topic.scores(explanation['topic_query'])
            _,activation,_,_,_=adj.components(j['question'],tick)
            candidates=set().union(*entity.roles[explanation['entities'][0]].values())
            sources=[]
            for source in sorted(set(j['required'])-packet):
                if source not in lookup:
                    sources.append({'source':source,'in_corpus':False});continue
                node=node_by_id[source];session,pos=position[source]
                distances=[abs(position[s][1]-pos) for s in packet if position[s][0]==session]
                sources.append({'source':source,'in_corpus':True,'speaker':lookup[source]['speaker'],'passes_entity_gate':node in candidates,'stripped_body_bm25_score':float(scores[node-1]),'stripped_body_bm25_global_rank':rank(scores,node),'original_graph_activation':float(activation[node-1]),'nearest_packet_turn_same_session':min(distances) if distances else None,'in64':source in large,'previous_turn':by_session[session][pos-1] if pos else None,'next_turn':by_session[session][pos+1] if pos+1<len(by_session[session]) else None})
            rows.append({'case':j['case'],'annotated_source_count':len(j['required']),'all_gold_possible_at16':len(j['required'])<=16,'missing16':len(sources),'missing64':len(set(j['required'])-large),'sources':sources})
    behavior={}
    for backend in ('codex','claude'):
        entries=[]
        for j in jobs:
            a,b=[json.loads((ROOT/'benchmarks/results/grounded-answers'/backend/(j['case']+'-'+arm+'.json')).read_text()) for arm in ('baseline','grounded')]
            entries.append({'case':j['case'],'complete':a['complete_evidence'],'baseline_abstained':a['abstained'],'grounded_abstained':b['abstained'],'baseline_f1':a['answer_f1'],'grounded_f1':b['answer_f1']})
        behavior[backend]={'incomplete_baseline_answered':sum(not e['complete'] and not e['baseline_abstained'] for e in entries),'incomplete_grounded_answered':sum(not e['complete'] and not e['grounded_abstained'] for e in entries),'new_abstentions':[e['case'] for e in entries if not e['baseline_abstained'] and e['grounded_abstained']],'per_question':entries}
    sources=[s for r in rows for s in r['sources']]
    result={'date':'2026-10-15','scope':'Six incomplete cases selected post-hoc; original public dataset, no model calls. Lexical rank is over full body index. Distance is undirected within-session turn distance, not proof of membership in current seed set. Candidate inclusion and positive activation are separate from final top-k.','cases':rows,'summary':{'cases':len(rows),'missing16_sources':len(sources),'missing_sources_found_in_corpus':sum(s['in_corpus'] for s in sources),'missing16_sources_recovered_in64':sum(s.get('in64',False) for s in sources),'missing_sources_passing_gate':sum(s.get('passes_entity_gate',False) for s in sources),'zero_body_bm25':sum(s.get('stripped_body_bm25_score')==0 for s in sources),'positive_original_activation':sum(s.get('original_graph_activation',0)>0 for s in sources)},'behavior':behavior,'model_calls':0}
    (ROOT/'benchmarks/protocols/missing-evidence-audit-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
    for r in rows:print(r['case'],'missing16/64',r['missing16'],r['missing64'],'min distance',[s.get('nearest_packet_turn_same_session') for s in r['sources']])
if __name__=='__main__':main()
