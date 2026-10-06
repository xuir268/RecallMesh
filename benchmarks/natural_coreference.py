"""Pre-outcome mining of unedited LoCoMo pronoun/antecedent candidates.

This is an exploratory screen, NOT a human-validated alias benchmark. Query text
is the original preceding turn, not a generated QA. Annotation status is retained.
No QA labels, mined chain IDs, or evaluation queries enter graph construction.
"""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.lexical import LexicalIndex,AssociativeRetriever,stable_topk

SEED=20261007
OPENING=re.compile(r"^(?:It(?:'s| is| was| has| had| really| helped| makes| made| feels| felt| brings| brought| means| meant| turned| gives| gave| took)|They(?:'re| are| were| have| had)|He(?:'s| is| was| has| had)|She(?:'s| is| was| has| had))\b",re.I)


def turns(sample):
 for key in sorted((k for k in sample['conversation'] if re.fullmatch(r'session_\d+',k)),key=lambda k:int(k.split('_')[1])):
  yield int(key.split('_')[1]),sample['conversation'][key]


def mine(data,count=100):
 candidates=[]
 for sample in data:
  for tick,session in turns(sample):
   for i,target in enumerate(session):
    if i==0 or not OPENING.search(target['text']) or len(target['text'].split())<12:continue
    anchor=session[i-1]
    if len(anchor['text'].split())<10:continue
    candidates.append({'id':sample['sample_id']+':'+target['dia_id'],'conversation':sample['sample_id'],
      'tick':tick,'anchor':anchor['dia_id'],'target':target['dia_id'],
      'query':anchor['text'],'anchor_text':anchor['text'],'target_text':target['text'],
      'annotation':{'status':'unreviewed','valid_chain':None,'referent':None,'kind':None},
      'warning':'Pronoun opening is a heuristic; expletive it, vague affect and ambiguous antecedents may be false positives.'})
 random.Random(SEED).shuffle(candidates)
 return candidates[:count],len(candidates)


def cluster_interval(rows,key):
 by=defaultdict(list)
 for row in rows:by[row['conversation']].append(row['scores'][key]['target_recall']-row['scores']['lexical']['target_recall'])
 groups=list(by.values());rng=np.random.default_rng(SEED);means=[]
 for _ in range(3000):
  sampled=rng.integers(0,len(groups),len(groups));values=[x for i in sampled for x in groups[i]];means.append(np.mean(values))
 return np.percentile(means,[2.5,97.5]).tolist()


def run(data_path,out,count=100,limit=16):
 raw=data_path.read_bytes();data=json.loads(raw);cases,total=mine(data,count)
 out.mkdir(parents=True,exist_ok=True)
 # Lock source, deterministic selection and settings BEFORE computing any retrieval outcome.
 protocol={'version':1,'source_sha256':hashlib.sha256(raw).hexdigest(),'seed':SEED,'requested':count,'mined':len(cases),
  'candidate_pool':total,'cases_sha256':hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
  'status':'exploratory unreviewed candidate screen; not a verified natural identity benchmark',
  'query':'unedited preceding dialogue turn','limit':limit,'seed_count':8,'hops':2,'graph_weight':.4,
  'edge_policy':'star: current plus previous session turn and top-three lexical prior turns',
  'selection':'pronoun opening/length rules, fixed-seed shuffle; no outcome selection',
  'conditions':['lexical','associative','oracle','counterfactual','bridge_removed'],
  'controls':'oracle supplies annotated candidate pair; counterfactual replaces candidate target with a different original source; bridge_removed removes candidate anchor from associative packet. These are diagnostic evidence controls, not production methods.',
  'answer_quality':'not measured; no model answer calls',
  'ablations':['adjacency_only','learned_only','shuffled'],
  'shuffled':'fixed permutation of node IDs retains topology and degree distribution, not each labeled node degree',
  'holdout':'settings inherited from earlier work; same public corpus previously inspected; not a new independent held-out dataset'}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'cases.json').write_text(json.dumps(cases,indent=2))
 rows=[]
 for sample in data:
  selected=[c for c in cases if c['conversation']==sample['sample_id']]
  if not selected:continue
  index=LexicalIndex();engine=EdgeStore(1<<18);adjacent=EdgeStore(1<<18);learned=EdgeStore(1<<18);shuffled=EdgeStore(1<<18);history=[];mapping={};inverse={};tick=0
  for tick,session in turns(sample):
   previous=None
   for turn in session:
    # Preserve the current framework's speaker metadata and captions in every arm.
    text=turn['speaker']+': '+turn['text']
    if turn.get('blip_caption'):text+=' [Image caption: '+turn['blip_caption']+']'
    scores=index.scores(turn['text']);top=stable_topk(scores,3);related=[int(i)+1 for i in top if scores[i]>0]
    nid=index.add(text);mapping[turn['dia_id']]=nid;inverse[nid]=turn['dia_id']
    context=set(related)
    if previous:context.add(previous)
    if context:
     src=np.full(len(context),nid,np.uint32);dst=np.array(sorted(context),np.uint32);engine.reinforce(src,dst,tick)
    for neighbor in sorted(context):history.append((nid,neighbor,tick))
    if previous:adjacent.reinforce(np.array([nid],np.uint32),np.array([previous],np.uint32),tick)
    if related:learned.reinforce(np.full(len(related),nid,np.uint32),np.array(related,np.uint32),tick)
    previous=nid
  permutation=np.random.default_rng(SEED).permutation(len(inverse))+1
  for a,b,when in history:shuffled.reinforce(np.array([permutation[a-1]],np.uint32),np.array([permutation[b-1]],np.uint32),when)
  for graph in [adjacent,learned,shuffled]:graph.freeze(tick)
  ablations={name:AssociativeRetriever(graph,index) for name,graph in [('adjacency_only',adjacent),('learned_only',learned),('shuffled',shuffled)]}
  engine.freeze(tick);retriever=AssociativeRetriever(engine,index);before=engine.stats()
  for case in selected:
   trace=retriever.explain(case['query'],tick,limit,.4,hops=2)
   associative=[inverse[r['node']] for r in trace['results']]
   lexical=[inverse[int(n)] for n in retriever.search(case['query'],tick,limit,0)]
   alternatives=[inverse[n] for n in sorted(inverse) if inverse[n] not in {case['anchor'],case['target']}]
   alternate=alternatives[int(hashlib.sha256(case['id'].encode()).hexdigest(),16)%len(alternatives)]
   packets={'lexical':lexical,'associative':associative,'oracle':[case['anchor'],case['target']],
      'counterfactual':[alternate if n==case['target'] else n for n in associative],
      'bridge_removed':[n for n in associative if n!=case['anchor']]}
   for name,other in ablations.items():packets[name]=[inverse[int(n)] for n in other.search(case['query'],tick,limit,.4,hops=2)]
   scores={name:{'target_recall':int(case['target'] in ids),
                 'candidate_pair_recall':len({case['anchor'],case['target']}&set(ids))/2,
                 'all_candidate_pair':int({case['anchor'],case['target']}<=set(ids))} for name,ids in packets.items()}
   witnesses={inverse[r['node']]:[inverse[n] for n in r['path']] for r in trace['results']}
   rows.append({'id':case['id'],'conversation':case['conversation'],'anchor':case['anchor'],'target':case['target'],
                'annotation':case['annotation'],'packets':packets,'scores':scores,'witnesses':witnesses,
                'target_lexical_score':float(index.scores(case['query'])[mapping[case['target']]-1])})
  after=engine.stats();assert before['inserts']==after['inserts'] and before['updates']==after['updates']
 summary={name:{metric:float(np.mean([r['scores'][name][metric] for r in rows])) for metric in ['target_recall','candidate_pair_recall','all_candidate_pair']} for name in protocol['conditions']+['adjacency_only','learned_only','shuffled']}
 gains=sum(r['scores']['associative']['target_recall']>r['scores']['lexical']['target_recall'] for r in rows)
 losses=sum(r['scores']['associative']['target_recall']<r['scores']['lexical']['target_recall'] for r in rows)
 result={'protocol':protocol,'summary':summary,'graph_target_gains':gains,'graph_target_losses':losses,
  'cluster_bootstrap_delta_95':cluster_interval(rows,'associative'),'human_validated_cases':0,
  'zero_lexical_score_targets':sum(r['target_lexical_score']==0 for r in rows),'conversations':len(set(r['conversation'] for r in rows))}
 (out/'results.json').write_text(json.dumps(result,indent=2));(out/'per-case.json').write_text(json.dumps(rows,indent=2))
 print(json.dumps(result,indent=2))
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('benchmarks/data/locomo10.json'));parser.add_argument('--out',type=Path,default=Path('benchmarks/results/natural-coreference'));parser.add_argument('--count',type=int,default=100);parser.add_argument('--limit',type=int,default=16)
 args=parser.parse_args();run(args.data,args.out,args.count,args.limit)
