"""Prepare an explicitly exploratory adjacency-only arm on the locked pilot.

This arm was added after inspecting the primary run. Weight .4, two hops and
budget16 stay fixed; no answer labels select edges or ranking parameters.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.agent import Evidence, make_prompt
from assoc_mem.lexical import AssociativeRetriever, LexicalIndex
from agent_answers import ROOT, OUT


def main():
    source=json.loads((ROOT/'benchmarks/data/locomo10.json').read_text())
    locked=json.loads((OUT/'jobs.json').read_text())
    requests=[j for j in locked if j['cohort']=='primary' and j['mode']=='lexical']
    jobs=[]
    for sample in source[3:]:
        relevant=[j for j in requests if j['case'].startswith(sample['sample_id']+'-q')]
        if not relevant:continue
        index=LexicalIndex();engine=EdgeStore(1<<18);texts={};dialog_ids={}
        sessions=sample['conversation'];tick=0
        for session in sorted(int(k.split('_')[1]) for k in sessions if re.fullmatch(r'session_\d+',k)):
            previous=None;tick=session
            for turn in sessions[f'session_{session}']:
                date=sessions.get(f'session_{session}_date_time','')
                text=f"[{date}] {turn['speaker']}: {turn['text']}"
                if turn.get('blip_caption'):text+=f" [Image caption: {turn['blip_caption']}]"
                nid=index.add(text);texts[nid]=text;dialog_ids[nid]=turn['dia_id']
                if previous is not None:
                    engine.reinforce(np.array([previous],np.uint32),np.array([nid],np.uint32),tick)
                previous=nid
        engine.freeze(tick);retriever=AssociativeRetriever(engine,index)
        for job in relevant:
            explanation=retriever.explain(job['question'],tick,budget=16,graph_weight=.4,hops=2)
            evidence=[Evidence(dialog_ids[r['node']],texts[r['node']]) for r in explanation['results']]
            prompt=make_prompt(job['question'],evidence)
            jobs.append({**job,'id':job['case']+'-adjacent','mode':'adjacent',
                         'exploratory':True,'evidence':[asdict(e) for e in evidence],
                         'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                         'context_words':sum(len(e.text.split()) for e in evidence),
                         'retrieval':explanation})
    (OUT/'adjacent-jobs.json').write_text(json.dumps(jobs,indent=2))
    (OUT/'adjacent-protocol.json').write_text(json.dumps({
        'status':'exploratory follow-up, prepared after initial answer results',
        'selection':'same 12 locked primary questions', 'calls_per_provider':len(jobs),
        'edge_policy':'previous/current turn within session only',
        'graph_weight':.4,'hops':2,'cutoff':.001,'budget_turns':16},indent=2))
    print(f'Prepared {len(jobs)} exploratory adjacency calls per model')

if __name__=='__main__':main()
