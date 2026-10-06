"""Explain which required facts are reached but excluded by current ranking."""
import json
from assoc_mem import Memory
from assoc_mem.agent import MemoryAgent
from reasoning_evidence import OUT, make_cases


def main():
    rows=[]
    for case in make_cases():
        with Memory(capacity=1024,graph_weight=.4) as memory:
            agent=MemoryAgent(memory,None);ids={}
            for i,(key,text) in enumerate(case['observations']):
                if i and i%4==0:memory.advance()
                ids[key]=agent.observe(text)
            memory.flush()
            lex,assoc,seeds,counts,arcs=memory.associative.components(case['question'],memory.tick,profile=True)
            base=memory.associative.rank(lex,assoc,len(lex),0).tolist()
            ranked=memory.associative.rank(lex,assoc,len(lex),.4).tolist()
            first_hop={}
            for a,b,hop,weight in arcs:first_hop.setdefault(b,hop)
            required=[]
            for i in case['required']:
                node=ids[f's{i}']
                required.append({'id':node,'text':case['facts'][i],'seed':node in seeds,
                    'lexical_rank':base.index(node)+1,'graph_rank':ranked.index(node)+1,
                    'propagated_score':float(assoc[node-1]),'first_reached_hop':first_hop.get(node),
                    'inside_budget':node in ranked[:4]})
            rows.append({'case':case['id'],'question':case['question'],'required':required,'operations':counts})
    (OUT/'retrieval-diagnosis.json').write_text(json.dumps(rows,indent=2))
    missing=[r for row in rows for r in row['required'] if not r['inside_budget']]
    print('Required facts outside graph budget:',len(missing))
    print('Of these, reached by propagation:',sum(r['first_reached_hop'] is not None for r in missing))

if __name__=='__main__':main()
