"""Fresh-session answer pilot using saved, audited LoCoMo retrieval contexts.

Prepare locks selection before model calls. Labels are never sent to models.
Primary sample is category-stratified; known gains/losses are diagnostics only.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import string
import sys
import time

from nltk.stem import PorterStemmer
from assoc_mem.agent import CliProvider, Evidence, make_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/results/agent-eval'
STEM = PorterStemmer()
MODES = ['none', 'lexical', 'associative']
REPORT_MODES = MODES + ['adjacent']


def normalize(text):
    text = str(text).lower().replace(',', '')
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(w for w in text.split() if w not in {'a', 'an', 'the', 'and'})


def token_f1(prediction, gold):
    pred = [STEM.stem(w) for w in normalize(prediction).split()]
    target = [STEM.stem(w) for w in normalize(gold).split()]
    overlap = sum((Counter(pred) & Counter(target)).values())
    if not overlap: return 0.
    return 2 * overlap / (len(pred) + len(target))


def answer_f1(prediction, gold, category):
    # Same normalization, Porter stemming and category handling as the saved
    # upstream evaluation.py. No heavy bert_score/model import or downloads.
    gold = str(gold)
    if category == 3: gold = gold.split(';')[0].strip()
    if category == 1:
        return statistics.mean(max(token_f1(p.strip(), g.strip())
                                   for p in prediction.split(',')) for g in gold.split(','))
    return token_f1(prediction, gold)


def prepare(out: Path, seed=20261005):
    out.mkdir(parents=True, exist_ok=True)
    source = (ROOT / 'benchmarks/data/locomo10.json').read_bytes()
    dataset = {s['sample_id']: s for s in json.loads(source)}
    traces_path = ROOT / 'benchmarks/results/associative-audit/traces.jsonl'
    traces = [json.loads(line) for line in traces_path.read_text().splitlines()]
    rng = random.Random(seed)
    primary = []
    for category in ['multi-hop', 'temporal', 'open-domain', 'single-hop']:
        group = [r for r in traces if r['category'] == category]
        primary += rng.sample(group, 3)
    selected = {(r['conversation'], r['question_index']) for r in primary}
    diagnostic = []
    for field in ['gained_evidence', 'lost_evidence']:
        group = [r for r in traces if r[field] and (r['conversation'], r['question_index']) not in selected]
        picked = rng.sample(group, 2)
        diagnostic += picked
        selected.update((r['conversation'], r['question_index']) for r in picked)
    jobs = []; labels = []; cases = []
    for cohort, rows in [('primary', primary), ('diagnostic', diagnostic)]:
        for row in rows:
            cid = row['conversation']; qi = row['question_index']; sample = dataset[cid]
            qa = sample['qa'][qi]; case = f'{cid}-q{qi}'
            texts = {}
            for key, turns in sample['conversation'].items():
                if not key.startswith('session_') or not isinstance(turns, list): continue
                date = sample['conversation'].get(key + '_date_time', '')
                for turn in turns:
                    text = f"[{date}] {turn['speaker']}: {turn['text']}"
                    if turn.get('blip_caption'): text += f" [Image caption: {turn['blip_caption']}]"
                    texts[turn['dia_id']] = text
            labels.append({'case': case, 'cohort': cohort, 'question': qa['question'],
                           'category': qa['category'], 'answer': qa['answer'], 'gold_evidence': row['gold']})
            cases.append({'case': case, 'cohort': cohort, 'category': qa['category'],
                          'gained_evidence': row['gained_evidence'], 'lost_evidence': row['lost_evidence'],
                          'paths': row['introduced']})
            modes = MODES if cohort == 'primary' else ['lexical', 'associative']
            for mode in modes:
                ids = [] if mode == 'none' else row['baseline' if mode == 'lexical' else 'associative']
                evidence = [Evidence(i, texts[i]) for i in ids]
                prompt = make_prompt(qa['question'], evidence)
                jobs.append({'id': case + '-' + mode, 'case': case, 'cohort': cohort,
                             'mode': mode, 'question': qa['question'],
                             'evidence': [asdict(e) for e in evidence],
                             'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                             'context_words': sum(len(e.text.split()) for e in evidence)})
    manifest = {'seed': seed, 'primary_questions': 12, 'diagnostic_questions': 4,
                'calls_per_provider': len(jobs), 'budget_turns': 16,
                'dataset_sha256': hashlib.sha256(source).hexdigest(),
                'traces_sha256': hashlib.sha256(traces_path.read_bytes()).hexdigest(),
                'primary_selection': 'Three seeded-random questions per category 1-4 from the seven internal-holdout conversations; no selection by retrieval success.',
                'diagnostic_selection': 'Two known retrieval gains and two losses; excluded from primary means.',
                'limitations': 'Small one-draw pilot, not full LoCoMo. Internal holdout was previously inspected. CLI harnesses differ across providers; compare memory arms within each model.',
                'cases': cases}
    for name, value in [('manifest', manifest), ('jobs', jobs), ('labels', labels)]:
        target = out / (name + '.json')
        encoded = json.dumps(value, indent=2, ensure_ascii=False)
        if target.exists() and target.read_text() != encoded:
            raise ValueError(f'Refusing to replace changed evaluation inputs: {target}')
        target.write_text(encoded)
    print(f'Prepared {len(jobs)} isolated calls per provider in {out}', flush=True)


def execute(out: Path, backend: str, model: str | None, workers: int):
    jobs = json.loads((out / 'jobs.json').read_text())
    if (out / 'adjacent-jobs.json').exists():
        jobs += json.loads((out / 'adjacent-jobs.json').read_text())
    provider = CliProvider(backend, model)
    directory = out / backend; directory.mkdir(exist_ok=True)
    def one(job):
        path = directory / (job['id'] + '.json')
        if path.exists():
            existing = json.loads(path.read_text())
            if existing['prompt_sha256'] != job['prompt_sha256'] or existing['requested_model'] != provider.model:
                raise ValueError('Cached completion configuration mismatch')
            return job['id'], 'cached'
        evidence = [Evidence(**e) for e in job['evidence']]
        actual = hashlib.sha256(make_prompt(job['question'], evidence).encode()).hexdigest()
        if actual != job['prompt_sha256']: raise ValueError('Prompt changed after preparation')
        response = provider.answer(job['question'], evidence)
        record = {**job, 'backend': backend, 'requested_model': provider.model,
                  'effort': provider.effort, 'completion': asdict(response)}
        temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(record, indent=2, ensure_ascii=False));temp.replace(path)
        return job['id'], 'completed'
    # Bounded windows stop further dispatch when a connection/model request fails.
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(jobs), workers):
            futures = [executor.submit(one, j) for j in jobs[start:start+workers]]
            for future in as_completed(futures):
                name, status = future.result();done += 1
                print(f'{backend} {done}/{len(jobs)} {name}: {status}', flush=True)
    summarize(out)


def summarize(out: Path):
    labels = {r['case']: r for r in json.loads((out / 'labels.json').read_text())}
    jobs = json.loads((out / 'jobs.json').read_text())
    if (out / 'adjacent-jobs.json').exists():
        jobs += json.loads((out / 'adjacent-jobs.json').read_text())
    all_rows = []; summaries = {}
    for backend in ['codex', 'claude']:
        rows = []
        for path in sorted((out / backend).glob('*.json')):
            record = json.loads(path.read_text()); label = labels[record['case']]; result = record['completion']
            citations = set(result['evidence_ids']); allowed = {e['id'] for e in record['evidence']}
            gold = set(label['gold_evidence'])
            row = {'backend': backend, 'case': record['case'], 'cohort': record['cohort'],
                   'mode': record['mode'], 'model': result['model'], 'question': record['question'],
                   'gold_answer': label['answer'], 'answer': result['answer'], 'rationale': result['rationale'],
                   'f1': answer_f1(result['answer'], label['answer'], label['category']),
                   'citation_ids': sorted(citations), 'invalid_citations': sorted(citations-allowed),
                   'gold_evidence_recall': len(allowed & gold)/len(gold) if gold else None,
                   'gold_citation_precision': len(citations & gold)/len(citations) if citations else None,
                   'abstained': result['answer'].strip().lower() == 'not enough information',
                   'seconds': result['seconds'], 'context_words': record['context_words'],
                   'estimated_cost_usd': result['estimated_cost_usd']}
            rows.append(row)
        if not rows: continue
        primary = [r for r in rows if r['cohort']=='primary']; summary = {}
        for mode in REPORT_MODES:
            selected = [r for r in primary if r['mode']==mode]
            if selected:
                summary[mode] = {'n':len(selected), 'mean_answer_f1':statistics.mean(r['f1'] for r in selected),
                    'abstentions':sum(r['abstained'] for r in selected),
                    'invalid_citation_count':sum(len(r['invalid_citations']) for r in selected),
                    'median_seconds':statistics.median(r['seconds'] for r in selected),
                    'mean_context_words':statistics.mean(r['context_words'] for r in selected)}
        indexed = {(r['case'],r['mode']):r for r in primary}
        paired = [(indexed[c,'lexical'], indexed[c,'associative']) for c in labels
                  if (c,'lexical') in indexed and (c,'associative') in indexed]
        summary['paired'] = {'n':len(paired), 'better':sum(b['f1']>a['f1'] for a,b in paired),
                             'worse':sum(b['f1']<a['f1'] for a,b in paired),
                             'tied':sum(b['f1']==a['f1'] for a,b in paired)}
        summary['models'] = sorted({r['model'] for r in rows})
        summary['completed_calls'] = len(rows); summary['expected_calls'] = len(jobs)
        summary['estimated_cost_usd'] = sum(r['estimated_cost_usd'] or 0 for r in rows) if backend=='claude' else None
        summaries[backend] = summary;all_rows += rows
    (out / 'scored.json').write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','score'])
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--backend',choices=['codex','claude'])
    parser.add_argument('--model')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.out)
    elif args.command=='score':summarize(args.out)
    else:
        if not args.backend:parser.error('--backend required for run')
        if not 1<=args.workers<=4:parser.error('--workers must be between 1 and 4')
        execute(args.out,args.backend,args.model,args.workers)
