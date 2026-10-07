"""Run the preregistered repeated-use experiment, without question training."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.lexical import LexicalIndex, AssociativeRetriever, stable_topk
from natural_coreference import turns
from locomo_retrieval import measure

ROOT = Path(__file__).resolve().parents[1]

def interval(rows, left, right):
    groups = {}
    for row in rows:
        groups.setdefault(row['conversation'], []).append(row[left] - row[right])
    values = list(groups.values())
    rng = np.random.default_rng(20261007)
    boot = [np.mean([x for i in rng.integers(0, len(values), len(values)) for x in values[i]]) for _ in range(3000)]
    return np.percentile(boot, [2.5, 97.5]).tolist()

def main():
    protocol = json.loads((ROOT/'benchmarks/protocols/longitudinal-v1.json').read_text())
    raw = (ROOT/'benchmarks/data/locomo10.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == protocol['source_sha256']
    membership = json.loads((ROOT/protocol['membership']).read_text())['membership']
    out = ROOT/'benchmarks/results/longitudinal'
    out.mkdir(parents=True, exist_ok=True)
    rows, diagnostics = [], []
    started = time.monotonic()
    for sample in json.loads(raw):
        selected = [c for c in membership if c['conversation'] == sample['sample_id']]
        if not selected:
            continue
        sessions = list(turns(sample))
        n = sum(len(s) for _, s in sessions)
        capacity = 16
        while capacity*.70 < n*(n-1)//2:
            capacity *= 2
        graph, adjacent = EdgeStore(capacity), EdgeStore(capacity)
        index, inverse, queries = LexicalIndex(), {}, []
        for tick, session in sessions:
            previous = None
            for turn in session:
                scores = index.scores(turn['text'])
                related = [int(i)+1 for i in stable_topk(scores, 3) if scores[i] > 0]
                text = '['+sample['conversation'].get(f'session_{tick}_date_time', '')+'] '+turn['speaker']+': '+turn['text']
                if turn.get('blip_caption'):
                    text += ' [Image caption: '+turn['blip_caption']+']'
                nid = index.add(text)
                inverse[nid] = turn['dia_id']
                queries.append(turn['text'])
                context = set(related)
                if previous:
                    context.add(previous)
                    adjacent.reinforce(np.array([nid], np.uint32), np.array([previous], np.uint32), tick)
                if context:
                    graph.reinforce(np.full(len(context), nid, np.uint32), np.array(sorted(context), np.uint32), tick)
                previous = nid
        base_tick = tick
        graph.freeze(tick)
        adjacent.freeze(tick)
        retrieval = AssociativeRetriever(graph, index)
        control = AssociativeRetriever(adjacent, index)
        case_rows = []
        for c in selected:
            q = sample['qa'][c['question_index']]['question']
            packet = control.search(q, base_tick, 16, .4).tolist()
            lexical = retrieval.search(q, base_tick, 16, 0).tolist()
            case_rows.append({'id':c['id'], 'conversation':sample['sample_id'], 'adjacency':measure(c['gold'], [inverse[i] for i in packet])['evidence_recall'], 'lexical':measure(c['gold'], [inverse[i] for i in lexical])['evidence_recall'], 'control_packet':packet})
        for session in range(51):
            tick = base_tick + session
            if session:
                # Snapshot stays unchanged during selection; all events use one tick.
                pairs = []
                for query in queries:
                    ids = retrieval.search(query, tick, 16, .4)
                    a, b = np.triu_indices(len(ids), 1)
                    pairs.append((ids[a], ids[b]))
                graph.reinforce(np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs]), tick)
                graph.freeze(tick)
                assert graph.stats()['rejected_full'] == 0, graph.stats()
            if session in protocol['checkpoints']:
                before = graph.stats()
                for c, row in zip(selected, case_rows):
                    q = sample['qa'][c['question_index']]['question']
                    assert control.search(q, base_tick, 16, .4).tolist() == row['control_packet']
                    ids = retrieval.search(q, tick, 16, .4)
                    row[f'graph_{session}'] = measure(c['gold'], [inverse[int(i)] for i in ids])['evidence_recall']
                assert graph.stats() == before
                diagnostics.append({'conversation':sample['sample_id'], 'session':session, 'tick':tick, 'nodes':n, 'stats':before})
                print(sample['sample_id'], 'session', session, 'seconds', round(time.monotonic()-started, 1), flush=True)
        for row in case_rows:
            del row['control_packet']
        rows.extend(case_rows)
    curve = [{'session':s, 'graph':float(np.mean([r[f'graph_{s}'] for r in rows])), 'adjacency':float(np.mean([r['adjacency'] for r in rows])), 'lexical':float(np.mean([r['lexical'] for r in rows])), 'graph_minus_adjacency_ci95':interval(rows, f'graph_{s}', 'adjacency')} for s in protocol['checkpoints']]
    last = curve[-1]
    gain = last['graph'] - curve[1]['graph']
    advantage = last['graph'] - last['adjacency']
    passed = gain >= .01 and advantage >= .01 and last['graph_minus_adjacency_ci95'][0] > 0 and last['graph'] >= curve[-2]['graph']
    result = {'protocol':protocol, 'cases':len(rows), 'curve':curve, 'session50_minus1':gain, 'session50_minus_adjacency':advantage, 'promote_hebbian':bool(passed), 'elapsed_seconds':time.monotonic()-started, 'diagnostics':diagnostics}
    (out/'per-case.json').write_text(json.dumps(rows, indent=2))
    (ROOT/'benchmarks/protocols/longitudinal-per-case-v1.json').write_text(json.dumps(rows, indent=2))
    (ROOT/'benchmarks/protocols/longitudinal-results-v1.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('protocol','diagnostics')}, indent=2))

if __name__ == '__main__':
    main()
