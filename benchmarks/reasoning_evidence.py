"""Locked causal evidence tests plus stratified natural multi-evidence QA.

Synthetic facts are newly generated. Ground truth is computed independently.
Natural cases are explicitly sampled by prior retrieval outcome, not population
representative. Oracle/control evidence is never described as production retrieval.
"""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

from assoc_mem import Memory
from assoc_mem.agent import AnswerFormatError, CliProvider, Evidence, MemoryAgent, make_prompt
from agent_answers import ROOT, answer_f1

OUT = ROOT / 'benchmarks/results/reasoning-evidence'
SEED = 20261006


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def make_cases(seed=SEED):
    rng = random.Random(seed)
    cases = []
    for family in ['alias', 'arithmetic', 'temporal']:
        for index in range(4):
            codes = rng.sample(range(1000, 9999), 20)
            root, middle, tail = (f'ID{n}' for n in codes[:3])
            rooms = [f'RM{n}' for n in codes[3:6]]
            if family == 'alias':
                facts = [f'Package {root} uses the tracking handle {middle}.',
                         f'Tracking handle {middle} identifies container {tail}.',
                         f'Container {tail} is stored in room {rooms[0]}.']
                question = f'In which room is package {root} stored? Return only the room code.'
                gold, changed = rooms[:2]
                alternate = facts.copy(); alternate[2] = facts[2].replace(gold, changed)
                required = [0, 1, 2]; changed_required = required
            elif family == 'arithmetic':
                a, b, c = rng.randint(3, 9), rng.randint(4, 11), rng.randint(3, 8)
                d = c + rng.randint(2, 5)
                facts = [f'Order {root} requests {a} crates of product {middle}.',
                         f'Each crate of product {middle} contains {b} packets of component {tail}.',
                         f'Each packet of component {tail} contains {c} cells.']
                question = f'How many cells in total are requested by order {root}? Return only the integer.'
                gold, changed = str(a*b*c), str(a*b*d)
                alternate = facts.copy(); alternate[2] = facts[2].replace(f'{c} cells', f'{d} cells')
                required = [0, 1, 2]; changed_required = required
            else:
                date = f'2031-07-{index+10:02d}'
                facts = [f'Parcel {root} is tracked under handle {middle}.',
                         f'At 09:00 on {date}, handle {middle} moved to room {rooms[0]}.',
                         f'At 12:00 on {date}, handle {middle} moved to room {rooms[1]}.',
                         f'At 18:00 on {date}, handle {middle} moved to room {rooms[2]}.']
                question = f'At 14:00 on {date}, in which room was parcel {root}? Return only the room code.'
                gold, changed = rooms[1], rooms[0]
                alternate = facts.copy(); alternate[2] = facts[2].replace('12:00', '16:00')
                required = [0, 2]; changed_required = [0, 1]
            distractors = []
            for j in range(16):
                code = f'ZZ{rng.randrange(100000, 999999)}'
                place = f'RM{rng.randrange(10000, 99999)}'
                distractors.append(rng.choice([
                    f'Package {code} is stored in room {place}.',
                    f'Order {code} requests {rng.randint(2,9)} crates for delivery next week.',
                    f'Container {code} has {rng.randint(12,80)} cells and a damaged label.',
                    f'Parcel {code} uses a paper tracking tag stored in room {place}.',
                ]))
            # A shuffled observation stream, in four-turn sessions. Neither the
            # question nor the labels are used by the production graph builder.
            observations = [(f's{i}', t) for i,t in enumerate(facts)] + [(f'd{i}',t) for i,t in enumerate(distractors)]
            rng.shuffle(observations)
            cases.append({'id':f'{family}-{index}', 'family':family, 'question':question,
                          'facts':facts, 'alternate':alternate, 'gold':gold,
                          'counterfactual_gold':changed, 'required':required,
                          'counterfactual_required':changed_required,
                          'observations':observations})
    return cases


def record_job(case, suite, arm, question, evidence, gold, required, metadata=None):
    prompt = make_prompt(question, evidence)
    return {'id':f'{suite}-{case}-{arm}', 'case':case, 'suite':suite, 'arm':arm,
            'question':question, 'evidence':[asdict(e) for e in evidence],
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            # Labels are retained in the supervisor's file only, never make_prompt.
            'gold':gold, 'required_evidence':required, 'metadata':metadata or {}}


def prepare(out: Path):
    jobs=[]; cases=make_cases(); case_details=[]
    for case in cases:
        with Memory(capacity=1024,graph_weight=.4) as memory:
            agent=MemoryAgent(memory, None)
            id_map={}
            for i,(key,text) in enumerate(case['observations']):
                if i and i%4==0:memory.advance()
                id_map[key]=str(agent.observe(text))
            required=[id_map[f's{i}'] for i in case['required']]
            before=None
            for mode in ['lexical','associative']:
                evidence,trace=agent.context(case['question'],mode,budget=4,hops=2)
                current=memory.engine.stats()
                if before is not None:
                    assert current['inserts']==before['inserts'] and current['updates']==before['updates']
                before=current
                jobs.append(record_job(case['id'],'synthetic',mode,case['question'],evidence,
                    case['gold'],required,{'family':case['family'],'trace':trace}))
            oracle=[Evidence(id_map[f's{i}'],text) for i,text in enumerate(case['facts'])]
            # Pad three-fact oracle contexts to the same four-record budget.
            if len(oracle)<4:
                oracle.append(Evidence(id_map['d0'],next(t for k,t in case['observations'] if k=='d0')))
            rng=random.Random(SEED+len(jobs));rng.shuffle(oracle)
            jobs.append(record_job(case['id'],'synthetic','oracle',case['question'],oracle,case['gold'],required,{'family':case['family']}))
            missing=[e for e in oracle if e.id!=id_map['s0']]
            missing.append(Evidence(id_map['d1'],next(t for k,t in case['observations'] if k=='d1')))
            jobs.append(record_job(case['id'],'synthetic','bridge_removed',case['question'],missing,
                'Not enough information',[],{'family':case['family'],'removed_id':id_map['s0']}))
            replacements={id_map[f's{i}']:text for i,text in enumerate(case['alternate'])}
            changed=[Evidence(e.id,replacements.get(e.id,e.text)) for e in oracle]
            jobs.append(record_job(case['id'],'synthetic','counterfactual',case['question'],changed,
                case['counterfactual_gold'],[id_map[f's{i}'] for i in case['counterfactual_required']],{'family':case['family']}))
            case_details.append({'case':case['id'],'family':case['family'],'node_ids':id_map,
                                 'oracle_gold':case['gold'],'counterfactual_gold':case['counterfactual_gold']})
    # Natural cases: multi-evidence questions from existing held-out conversations.
    # Select equally by graph retrieval gain, loss, or unchanged evidence. Do not
    # pool this selected sample into a representative LoCoMo score.
    dataset=json.loads((ROOT/'benchmarks/data/locomo10.json').read_text())
    samples={s['sample_id']:s for s in dataset[3:]}
    traces=[json.loads(x) for x in (ROOT/'benchmarks/results/associative-audit/traces.jsonl').read_text().splitlines()]
    prior=json.loads((ROOT/'benchmarks/results/agent-eval/manifest.json').read_text())
    prior_cases={c['case'] for c in prior['cases']}
    rng=random.Random(SEED)
    natural=[]
    for stratum in ['gain','loss','unchanged']:
        eligible=[]
        for r in traces:
            cid=r['conversation']; qid=r['question_index'];qa=samples[cid]['qa'][qid]
            if f'{cid}-q{qid}' in prior_cases or qa['category']!=1 or not 2<=len(set(r['gold']))<=8:continue
            if stratum=='gain' and not (r['gained_evidence'] and not r['lost_evidence']):continue
            if stratum=='loss' and not (r['lost_evidence'] and not r['gained_evidence']):continue
            if stratum=='unchanged' and (r['gained_evidence'] or r['lost_evidence']):continue
            eligible.append(r)
        if len(eligible)<4:raise ValueError(f'Not enough natural cases in {stratum}: {len(eligible)}')
        for r in rng.sample(eligible,4):
            sample=samples[r['conversation']];qa=sample['qa'][r['question_index']]
            case=f"{r['conversation']}-q{r['question_index']}";texts={}
            for key,turns in sample['conversation'].items():
                if not re.fullmatch(r'session_\d+',key):continue
                date=sample['conversation'].get(key+'_date_time','')
                for turn in turns:
                    text=f"[{date}] {turn['speaker']}: {turn['text']}"
                    if turn.get('blip_caption'):text+=f" [Image caption: {turn['blip_caption']}]"
                    texts[turn['dia_id']]=text
            if any(e not in texts for e in r['gold']):raise ValueError('Unresolvable gold evidence')
            for arm,ids in [('lexical',r['baseline']),('associative',r['associative']),('oracle',list(dict.fromkeys(r['gold'])))]:
                # Oracle is a diagnostic upper bound, not an equal-budget retriever.
                evidence=[Evidence(e,texts[e]) for e in ids]
                jobs.append(record_job(case,'natural',arm,qa['question'],evidence,qa['answer'],r['gold'],
                    {'stratum':stratum,'category':qa['category'],'gained_evidence':r['gained_evidence'],
                     'lost_evidence':r['lost_evidence'],'paths':r['introduced'] if arm=='associative' else []}))
            natural.append({'case':case,'stratum':stratum})
    manifest={'seed':SEED,'synthetic_cases':12,'natural_cases':12,'calls_per_provider':len(jobs),
              'synthetic_design':'4 seeded instances each of alias chaining, three-factor arithmetic, temporal state selection; 4 evidence records in every arm.',
              'natural_design':'4 gain, 4 loss, 4 unchanged multi-evidence category-1 cases, excluding earlier pilot questions. Outcome-stratified diagnostic sample, not population representative.',
              'fixed_retrieval':{'budget_synthetic':4,'budget_natural':16,'hops':2,'graph_weight':.4,'seed_count':8},
              'controls':'Oracle supplied facts; remove identity bridge; change terminal fact or timestamp with independently recomputed target.',
              'claims_boundary':'Tests evidence-dependent answering, not access to private reasoning or proof of improved intrinsic model intelligence.',
              'cases':case_details,'natural_selection':natural,'jobs_sha256':digest(jobs)}
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('jobs',jobs),('manifest',manifest),('synthetic-generator-output',cases)]:
        path=out/(name+'.json');encoded=json.dumps(value,indent=2,ensure_ascii=False)
        if path.exists() and path.read_text()!=encoded:raise ValueError(f'Refusing to change locked inputs: {path}')
        path.write_text(encoded)
    print(f'Locked {len(jobs)} calls per provider; {len(cases)} synthetic + {len(natural)} natural cases')


def execute(out,backend,workers):
    jobs=json.loads((out/'jobs.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    assert digest(jobs)==manifest['jobs_sha256']
    provider=CliProvider(backend);folder=out/backend;folder.mkdir(exist_ok=True)
    def one(job):
        dest=folder/(job['id']+'.json')
        if dest.exists():
            row=json.loads(dest.read_text())
            if row['prompt_sha256']!=job['prompt_sha256'] or row['requested_model']!=provider.model:raise ValueError('Cache mismatch')
            return job['id'],'cached'
        evidence=[Evidence(**e) for e in job['evidence']]
        prompt=make_prompt(job['question'],evidence)
        assert hashlib.sha256(prompt.encode()).hexdigest()==job['prompt_sha256']
        try:
            answer=provider.answer(job['question'],evidence)
            row={**job,'backend':backend,'requested_model':provider.model,'effort':provider.effort,'status':'success','completion':asdict(answer)}
        except AnswerFormatError as exc:
            # Keep the first attempt as a failed output. Do not sample again until
            # a desired format/answer appears; score reliability conservatively.
            row={**job,'backend':backend,'requested_model':provider.model,'effort':provider.effort,
                 'status':'format_error','error':str(exc),'raw_output':exc.raw_output,
                 'completion':{'answer':'','evidence_ids':[],'rationale':'','model':provider.model,
                               'seconds':None,'usage':{},'estimated_cost_usd':None}}

        temporary=dest.with_suffix('.tmp');temporary.write_text(json.dumps(row,indent=2,ensure_ascii=False));temporary.replace(dest)
        return job['id'],'completed'
    # Interleave suites and arms deterministically to limit ordering effects.
    pending=jobs.copy();random.Random(SEED+1).shuffle(pending)
    done=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0,len(pending),workers):
            for future in as_completed([pool.submit(one,j) for j in pending[start:start+workers]]):
                name,status=future.result();done+=1;print(f'{backend} {done}/{len(jobs)} {name}: {status}',flush=True)
    score(out)


def exact_answer(answer,gold):
    # The prompt asks for a room code or integer. Accept harmless surrounding
    # punctuation and the unit label, but never search a long response for gold.
    text=answer.strip().lower().strip(' .`\"')
    expected=gold.lower()
    if expected=='not enough information':return text==expected
    if expected.startswith('rm'):
        return re.fullmatch(r'(?:room\s+)?'+re.escape(expected),text) is not None
    return re.fullmatch(re.escape(expected)+r'(?:\s+cells)?',text) is not None


def score(out):
    rows=[];summary={}
    for backend in ['codex','claude']:
        group=[]
        for path in sorted((out/backend).glob('*.json')):
            row=json.loads(path.read_text());completion=row['completion']
            ids={e['id'] for e in row['evidence']};required=set(row['required_evidence']);cites=set(completion['evidence_ids'])
            row['answer_score']=float(exact_answer(completion['answer'],row['gold'])) if row['suite']=='synthetic' else answer_f1(completion['answer'],row['gold'],1)
            row['complete_required_evidence']=required<=ids
            row['full_required_citations']=required<=cites
            row['invalid_citations']=sorted(cites-ids)
            row['abstained']=completion['answer'].strip().lower()=='not enough information'
            row['grounded_correct']=(row['answer_score']==1 and row['complete_required_evidence'] and row['full_required_citations'])
            # Secondary, explicitly post-hoc content audit: an integer or list
            # answer can be correct despite violating the string schema. Keep
            # the first-attempt strict score unchanged and preserve raw output.
            row['audited_answer']=completion['answer']
            row['content_score']=row['answer_score']
            row['format_only_recovery']=False
            if row.get('status')=='format_error':
                row['content_score']=None
                raw=row.get('raw_output')
                if raw:
                    try:
                        payload=json.loads(raw)
                        value=payload.get('answer')
                        if type(value) is int:
                            recovered=str(value)
                        elif isinstance(value,list) and all(isinstance(v,str) for v in value):
                            recovered=', '.join(value)
                        else:
                            recovered=None
                        if recovered is not None:
                            row['audited_answer']=recovered
                            row['format_only_recovery']=True
                            row['content_score']=float(exact_answer(recovered,row['gold'])) if row['suite']=='synthetic' else answer_f1(recovered,row['gold'],1)
                    except (ValueError,AttributeError):
                        pass
            group.append(row)
        if not group:continue
        entry={'completed':len(group),'expected':96,'synthetic':{},'natural':{},'causal_pairs':{}}
        for suite in ['synthetic','natural']:
            strata=['all'] if suite=='synthetic' else ['gain','loss','unchanged']
            for stratum in strata:
                selected=[r for r in group if r['suite']==suite and (stratum=='all' or r['metadata']['stratum']==stratum)]
                arms=defaultdict(list)
                for r in selected:arms[r['arm']].append(r)
                result={arm:{'n':len(rs),'mean_answer_score':statistics.mean(r['answer_score'] for r in rs),
                             'complete_required_evidence':sum(r['complete_required_evidence'] for r in rs),
                             'full_required_citations':sum(r['full_required_citations'] for r in rs),
                             'abstentions':sum(r['abstained'] for r in rs),
                             'grounded_exact_successes':sum(r['grounded_correct'] for r in rs) if suite=='synthetic' else None} for arm,rs in arms.items()}
                if suite=='synthetic':entry[suite]=result
                else:entry[suite][stratum]=result
        indexed={(r['case'],r['arm']):r for r in group if r['suite']=='synthetic'}
        cases=sorted({case for case,arm in indexed})
        complete=[c for c in cases if all((c,a) in indexed for a in ['oracle','bridge_removed','counterfactual'])]
        entry['causal_pairs']={'n':len(complete),
            'oracle_and_counterfactual_correct':sum(indexed[c,'oracle']['answer_score']==1 and indexed[c,'counterfactual']['answer_score']==1 for c in complete),
            'oracle_correct_and_abstains_without_bridge':sum(indexed[c,'oracle']['answer_score']==1 and indexed[c,'bridge_removed']['answer_score']==1 for c in complete),
            'all_three_pass':sum(all(indexed[c,a]['answer_score']==1 for a in ['oracle','bridge_removed','counterfactual']) for c in complete)}
        entry['synthetic_by_family']={}
        for family in ['alias','arithmetic','temporal']:
            selected=[r for r in group if r['suite']=='synthetic' and r['metadata']['family']==family]
            entry['synthetic_by_family'][family]={arm:{'n':len(rs),'correct':sum(r['answer_score']==1 for r in rs)} for arm in sorted({r['arm'] for r in selected}) if (rs:=[r for r in selected if r['arm']==arm])}
        entry['content_audit']={}
        for arm in ['lexical','associative','oracle','counterfactual','bridge_removed']:
            rs=[r for r in group if r['suite']=='synthetic' and r['arm']==arm]
            entry['content_audit'][arm]={'n':len(rs),'correct':sum(r['content_score']==1 for r in rs),
                                         'unscorable':sum(r['content_score'] is None for r in rs)}
        entry['causal_pairs']['content_all_three_pass']=sum(all(indexed[c,a]['content_score']==1 for a in ['oracle','bridge_removed','counterfactual']) for c in complete)
        entry['format_failures']=sum(r.get('status')=='format_error' for r in group)
        entry['invalid_citation_count']=sum(len(r['invalid_citations']) for r in group)
        entry['estimated_cost_usd']=sum(r['completion']['estimated_cost_usd'] or 0 for r in group) if backend=='claude' else None
        summary[backend]=entry;rows.extend(group)
    (out/'scored.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','score'])
    p.add_argument('--backend',choices=['codex','claude']);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--out',type=Path,default=OUT);args=p.parse_args()
    if args.command=='prepare':prepare(args.out)
    elif args.command=='score':score(args.out)
    else:
        if not args.backend or not 1<=args.workers<=4:p.error('run needs --backend and 1-4 workers')
        execute(args.out,args.backend,args.workers)
