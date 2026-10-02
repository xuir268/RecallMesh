import json
from pathlib import Path
import re
import sys
import pytest

pytest.importorskip('nltk')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from reasoning_evidence import make_cases, exact_answer, digest
from assoc_mem.agent import Evidence, make_prompt


def test_new_cases_have_independently_checkable_targets():
    cases=make_cases()
    assert len(cases)==12 and cases==make_cases()
    for case in cases:
        assert case['gold']!=case['counterfactual_gold']
        assert sum(a!=b for a,b in zip(case['facts'],case['alternate']))==1
        if case['family']=='arithmetic':
            a=int(re.search(r'requests (\d+) crates',case['facts'][0])[1])
            b=int(re.search(r'contains (\d+) packets',case['facts'][1])[1])
            c=int(re.search(r'contains (\d+) cells',case['facts'][2])[1])
            d=int(re.search(r'contains (\d+) cells',case['alternate'][2])[1])
            assert int(case['gold'])==a*b*c
            assert int(case['counterfactual_gold'])==a*b*d
        if case['family']=='temporal':
            def at_time(facts):
                events=[(int(re.search(r'At (\d+):',f)[1]),re.search(r'room (RM\d+)',f)[1]) for f in facts[1:]]
                return max(e for e in events if e[0]<=14)[1]
            assert at_time(case['facts'])==case['gold']
            assert at_time(case['alternate'])==case['counterfactual_gold']


def test_exact_scoring_does_not_reward_answer_mentions_or_long_guesses():
    assert exact_answer('Room RM1234.','RM1234')
    assert exact_answer('144 cells','144')
    assert not exact_answer('RM1234 or RM5555','RM1234')
    assert not exact_answer('It is not RM1234','RM1234')
    assert not exact_answer('1440','144')
    assert exact_answer('Not enough information.','Not enough information')


def test_locked_controls_only_pass_question_and_evidence_to_model():
    path=Path(__file__).resolve().parents[1]/'benchmarks/results/reasoning-evidence'
    jobs=json.loads((path/'jobs.json').read_text())
    manifest=json.loads((path/'manifest.json').read_text())
    assert len(jobs)==96 and digest(jobs)==manifest['jobs_sha256']
    lookup={(j['case'],j['arm']):j for j in jobs if j['suite']=='synthetic'}
    for job in jobs:
        prompt=make_prompt(job['question'],[Evidence(**e) for e in job['evidence']])
        data=json.loads(prompt.split('INPUT JSON:\n')[1])
        assert set(data)=={'question','evidence'}
        if job['suite']=='synthetic':assert len(job['evidence'])==4
    for case in make_cases():
        original=lookup[case['id'],'oracle'];missing=lookup[case['id'],'bridge_removed']
        changed=lookup[case['id'],'counterfactual']
        ids={e['id'] for e in missing['evidence']}
        assert missing['metadata']['removed_id'] not in ids
        assert len(set(original['required_evidence'])-ids)==1
        assert original['question']==changed['question']
        assert sum(a!=b for a,b in zip(original['evidence'],changed['evidence']))==1
