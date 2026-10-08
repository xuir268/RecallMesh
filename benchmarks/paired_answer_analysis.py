"""Post-hoc paired diagnostics from versioned scores; never calls a model."""
from collections import defaultdict
import csv
import json
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TIE_TOLERANCE = 1e-12


def sign_test(wins, losses):
    """Two-sided exact binomial sign test, excluding ties."""
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)


def summarize(cases, candidate, reference='adjacency'):
    pairs = []
    groups = defaultdict(list)
    transitions = defaultdict(list)
    for case, arms in sorted(cases.items()):
        delta = arms[candidate]['answer_f1'] - arms[reference]['answer_f1']
        def state_of(row):
            return 'error' if row.get('status','success') != 'success' else ('abstain' if row['abstained'] else 'answer')
        state = state_of(arms[reference]) + '_to_' + state_of(arms[candidate])
        transitions[state].append(delta)
        groups[case.split(':')[0]].append(delta)
        pairs.append({'case':case, 'conversation':case.split(':')[0], 'delta_f1':delta, 'reference_f1':arms[reference]['answer_f1'], 'candidate_f1':arms[candidate]['answer_f1'], 'reference_abstained':arms[reference]['abstained'], 'candidate_abstained':arms[candidate]['abstained'], 'reference_status':arms[reference].get('status','success'), 'candidate_status':arms[candidate].get('status','success')})
    deltas = np.array([p['delta_f1'] for p in pairs])
    wins = int(sum(deltas > TIE_TOLERANCE))
    losses = int(sum(deltas < -TIE_TOLERANCE))
    cluster_means = {key:float(np.mean(v)) for key,v in sorted(groups.items())}
    cw = sum(v > TIE_TOLERANCE for v in cluster_means.values())
    cl = sum(v < -TIE_TOLERANCE for v in cluster_means.values())
    states = {}
    states_to_report=['abstain_to_answer','answer_to_abstain','answer_to_answer','abstain_to_abstain']
    if any('error' in k for k in transitions):
        states_to_report += ['error_to_answer','error_to_abstain','answer_to_error','abstain_to_error','error_to_error']
    for state in states_to_report:
        values = transitions[state]
        states[state] = {'n':len(values), 'wins':sum(v > TIE_TOLERANCE for v in values), 'ties':sum(abs(v) <= TIE_TOLERANCE for v in values), 'losses':sum(v < -TIE_TOLERANCE for v in values), 'mean_delta_f1':float(np.mean(values)) if values else None, 'contribution_to_overall_mean_delta':sum(values)/len(pairs)}
    return {'candidate':candidate,'reference':reference,'n':len(pairs),'wins':wins,'ties':len(pairs)-wins-losses,'losses':losses,'non_ties':wins+losses,'exact_sign_p_two_sided':sign_test(wins,losses),'delta_distribution':dict(zip(['minimum','q25','median','q75','maximum'],[float(v) for v in np.quantile(deltas,[0,.25,.5,.75,1])])), 'mean_delta_f1':float(np.mean(deltas)), 'abstention_transitions':states, 'conversation_mean_deltas':cluster_means, 'conversation_sign':{'n':len(groups),'wins':cw,'ties':len(groups)-cw-cl,'losses':cl,'exact_sign_p_two_sided':sign_test(cw,cl)}, 'per_question':pairs}


def analyze(result):
    comparisons = {}
    for backend, rows in result['per_call'].items():
        cases = defaultdict(dict)
        for row in rows:
            if row['arm'] in cases[row['case']]:
                raise ValueError('Duplicate case/arm')
            cases[row['case']][row['arm']] = row
        expected = set(result['protocol']['case_ids'])
        if set(cases) != expected or any(set(arms) != {'adjacency','hebbian_50','oracle'} for arms in cases.values()):
            raise ValueError('Incomplete paired cohort')
        comparisons[backend] = {arm:summarize(cases,arm) for arm in ['hebbian_50','oracle']}
    # Four exploratory question-level comparisons, adjusted together.
    tests = sorted((s['exact_sign_p_two_sided'],backend,arm) for backend,arms in comparisons.items() for arm,s in arms.items())
    previous = 0.0
    for rank,(p,backend,arm) in enumerate(tests):
        previous = max(previous,min(1.0,(len(tests)-rank)*p))
        comparisons[backend][arm]['holm_adjusted_question_sign_p'] = previous
    return {'analysis_date':'2026-10-08','status':'post-hoc diagnostic; original stopping rule unchanged','tie_tolerance_f1':TIE_TOLERANCE,'test':'two-sided exact sign test excludes ties; nominal question-level p assumes independent fair signs. Questions share seven conversations, so it is not cluster-robust. Conversation-level signs use equal-weight conversation mean deltas; seven clusters give little power. Four question-level p values also receive Holm adjustment.','comparisons':comparisons}


def main():
    path = ROOT/'benchmarks/protocols/medium-answer-results-v1.json'
    result = json.loads(path.read_text())
    result['paired_distribution'] = analyze(result)
    path.write_text(json.dumps(result,indent=2)+'\n')
    dest = ROOT/'docs/assets/medium-answer-paired-deltas.csv'
    with dest.open('w',newline='') as f:
        fields = ['backend','candidate','reference','case','conversation','reference_f1','candidate_f1','delta_f1','reference_abstained','candidate_abstained','reference_status','candidate_status']
        writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for backend,arms in result['paired_distribution']['comparisons'].items():
            for candidate,stats in arms.items():
                for row in stats['per_question']:
                    writer.writerow({'backend':backend,'candidate':candidate,'reference':'adjacency',**row})
    print(json.dumps({b:{a:{k:v for k,v in s.items() if k not in ['per_question','conversation_mean_deltas']} for a,s in arms.items()} for b,arms in result['paired_distribution']['comparisons'].items()},indent=2))

if __name__ == '__main__':
    main()
