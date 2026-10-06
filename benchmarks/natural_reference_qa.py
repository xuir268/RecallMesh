"""100 original LoCoMo QA with speaker-grounded first-person references.

Uses existing questions/evidence IDs, never rewritten questions or answers.
This is a narrow structural coreference cohort, not validated multi-turn aliases.
"""
from collections import defaultdict
import hashlib,json,random,re
from pathlib import Path
import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.lexical import LexicalIndex,AssociativeRetriever,stable_topk
from natural_coreference import turns,SEED
from locomo_retrieval import evidence_ids,measure


def mine(data,count=100):
 candidates=[]
 for ci,sample in enumerate(data):
  if ci<3:continue  # Earlier parameter-development conversations are excluded.
  ordered=[t for _,session in turns(sample) for t in session];lookup={t['dia_id']:t for t in ordered};names={t['speaker'] for t in ordered}
  for qi,qa in enumerate(sample['qa']):
   gold=evidence_ids(qa.get('evidence',[]))
   if qa['category'] not in [1,2,3,4] or not gold or any(g not in lookup for g in gold):continue
   subjects=[name for name in sorted(names) if re.search(r'\b'+re.escape(name)+r'\b',qa['question'],re.I)]
   if len(subjects)!=1:continue
   speaker=subjects[0]
   refs=[g for g in gold if lookup[g]['speaker']==speaker and re.match(r"^(?:I\b|I'm\b|I've\b|I'd\b|My\b)",lookup[g]['text']) and not re.search(r'\b'+re.escape(speaker)+r'\b',lookup[g]['text'],re.I)]
   if not refs:continue
   # A structural I/my -> annotated speaker identity link, not an inferred semantic relation.
   candidates.append({'id':sample['sample_id']+':q'+str(qi),'conversation':sample['sample_id'],'question_index':qi,
                     'question':qa['question'],'gold':gold,'speaker':speaker,'references':refs,
                     'reference_kind':'first-person pronoun to dataset speaker annotation',
                     'multi_turn_alias_validated':False})
 random.Random(SEED).shuffle(candidates)
 return candidates[:count],len(candidates)


def main():
 source=Path('benchmarks/data/locomo10.json');raw=source.read_bytes();data=json.loads(raw);cases,pool=mine(data)
 out=Path('benchmarks/results/natural-reference-qa');out.mkdir(parents=True,exist_ok=True)
 protocol={'source_sha256':hashlib.sha256(raw).hexdigest(),'seed':SEED,'candidate_pool':pool,'cases':len(cases),
  'selection':'original QA, one named speaker, original evidence starts I/my and omits that speaker name; fixed random sample before outcomes; conversations 0–2 excluded',
  'gold':'unchanged upstream evidence IDs; identity is grounded by dataset speaker annotation',
  'scope':'speaker coreference QA; NOT 100 independently annotated alias/multi-turn identity chains',
  'query':'original unedited benchmark question','metadata':'date, speaker name, original text and supplied caption are included equally in all production arms',
  'limit':16,'graph_weight':.4,'hops':2,'seed_count':8,'policy':'star current/prior-top-three/previous-session-turn',
  'conditions':['lexical','associative','oracle','counterfactual','bridge_removed'],
  'ablations':['adjacency_only','learned_only','shuffled'],
  'controls':'oracle is supplied gold; counterfactual replaces source with another speaker source; bridge_removed drops a traversed non-target intermediate from the evidence packet, if any, with no backfill. Neither is an answer-score test without model calls.',
  'answer_quality':'not evaluated','prior_exposure':'these conversations appeared in earlier project evaluation; not a new unseen corpus'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'cases.json').write_text(json.dumps(cases,indent=2))
 rows=[]
 for sample in data:
  selected=[c for c in cases if c['conversation']==sample['sample_id']]
  if not selected:continue
  index=LexicalIndex();graphs={name:EdgeStore(1<<18) for name in ['associative','adjacency_only','learned_only','shuffled']};lookup={};inverse={};history=[];tick=0
  for tick,session in turns(sample):
   previous=None
   for turn in session:
    scores=index.scores(turn['text']);top=stable_topk(scores,3);related=[int(i)+1 for i in top if scores[i]>0]
    text='['+sample['conversation'].get(f'session_{tick}_date_time','')+'] '+turn['speaker']+': '+turn['text']
    if turn.get('blip_caption'):text+=' [Image caption: '+turn['blip_caption']+']'
    nid=index.add(text);inverse[nid]=turn['dia_id'];lookup[turn['dia_id']]=turn
    context=set(related)
    if previous:context.add(previous)
    for name,neighbors in [('associative',sorted(context)),('adjacency_only',[previous] if previous else []),('learned_only',related)]:
     if neighbors:graphs[name].reinforce(np.full(len(neighbors),nid,np.uint32),np.array(neighbors,np.uint32),tick)
    history.extend((nid,n,tick) for n in sorted(context));previous=nid
  perm=np.random.default_rng(SEED).permutation(len(inverse))+1
  for a,b,when in history:graphs['shuffled'].reinforce(np.array([perm[a-1]],np.uint32),np.array([perm[b-1]],np.uint32),when)
  for g in graphs.values():g.freeze(tick)
  retrievers={name:AssociativeRetriever(g,index) for name,g in graphs.items()};before={n:g.stats() for n,g in graphs.items()}
  for c in selected:
   trace=retrievers['associative'].explain(c['question'],tick,16,.4,hops=2)
   packets={name:[inverse[int(n)] for n in r.search(c['question'],tick,16,.4)] for name,r in retrievers.items()}
   packets['lexical']=[inverse[int(n)] for n in retrievers['associative'].search(c['question'],tick,16,0)]
   packets['oracle']=c['gold'].copy()
   wrong=next(d for d,t in lookup.items() if t['speaker']!=c['speaker'] and d not in c['gold'])
   packets['counterfactual']=list(dict.fromkeys(wrong if n in c['references'] else n for n in packets['associative']))
   intermediates=[inverse[n] for hit in trace['results'] if inverse[hit['node']] in c['gold'] for n in hit['path'][:-1] if inverse[n] not in c['gold']]
   bridge=intermediates[-1] if intermediates else None
   packets['bridge_removed']=[n for n in packets['associative'] if n!=bridge]
   metrics={name:measure(c['gold'],ids) for name,ids in packets.items()}
   rows.append({'id':c['id'],'conversation':c['conversation'],'metrics':metrics,'packets':packets,'gold':c['gold'],
                'removed_intermediate':bridge,'coreference_ids':c['references'],'trace':trace})
  for n,g in graphs.items():assert before[n]['inserts']==g.stats()['inserts'] and before[n]['updates']==g.stats()['updates']
 summary={name:{key:float(np.mean([r['metrics'][name][key] for r in rows])) for key in ['evidence_recall','any_evidence','all_evidence']} for name in protocol['conditions']+protocol['ablations']}
 by=defaultdict(list)
 for r in rows:by[r['conversation']].append(r['metrics']['associative']['evidence_recall']-r['metrics']['lexical']['evidence_recall'])
 groups=list(by.values());rng=np.random.default_rng(SEED);boot=[]
 for _ in range(3000):boot.append(np.mean([x for i in rng.integers(0,len(groups),len(groups)) for x in groups[i]]))
 result={'protocol':protocol,'summary':summary,'cluster_bootstrap_delta_95':np.percentile(boot,[2.5,97.5]).tolist(),
   'graph_gains':sum(r['metrics']['associative']['evidence_recall']>r['metrics']['lexical']['evidence_recall'] for r in rows),
   'graph_losses':sum(r['metrics']['associative']['evidence_recall']<r['metrics']['lexical']['evidence_recall'] for r in rows)}
 (out/'results.json').write_text(json.dumps(result,indent=2));(out/'per-case.json').write_text(json.dumps(rows,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
