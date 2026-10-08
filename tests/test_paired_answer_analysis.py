from pathlib import Path
import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'benchmarks'))
    import paired_answer_analysis
    return paired_answer_analysis


def test_exact_sign_test_excludes_ties_and_is_symmetric(monkeypatch):
    m=module(monkeypatch)
    assert m.sign_test(0,0)==1
    assert m.sign_test(8,0)==pytest.approx(2/256)
    assert m.sign_test(0,8)==m.sign_test(8,0)
    assert m.sign_test(4,4)==1


def test_pairs_by_id_and_decomposes_transitions(monkeypatch):
    m=module(monkeypatch)
    def row(score,abstained):return {'answer_f1':score,'abstained':abstained}
    cases={'b:q1':{'adjacency':row(1,False),'oracle':row(0,True)},
           'a:q2':{'adjacency':row(0,True),'oracle':row(.5,False)},
           'a:q1':{'adjacency':row(0,True),'oracle':row(0,True)}}
    s=m.summarize(cases,'oracle')
    assert (s['wins'],s['ties'],s['losses'])==(1,1,1)
    assert s['mean_delta_f1']==pytest.approx(-.5/3)
    assert s['abstention_transitions']['abstain_to_answer']['n']==1
    assert s['abstention_transitions']['answer_to_abstain']['n']==1
    assert sum(t['contribution_to_overall_mean_delta'] for t in s['abstention_transitions'].values())==pytest.approx(s['mean_delta_f1'])
    assert s['conversation_mean_deltas']=={'a':.25,'b':-1}
    assert [r['case'] for r in s['per_question']]==['a:q1','a:q2','b:q1']


def test_refuses_unpaired_cohort(monkeypatch):
    m=module(monkeypatch)
    result={'protocol':{'case_ids':['a:q1']},'per_call':{'codex':[{'case':'a:q1','arm':'adjacency','answer_f1':0,'abstained':True}]}}
    with pytest.raises(ValueError,match='Incomplete'):
        m.analyze(result)


def test_call_errors_are_not_counted_as_answer_conversions(monkeypatch):
    m=module(monkeypatch)
    cases={'a:q1':{'adjacency':{'status':'error','answer_f1':0,'abstained':False},
                   'oracle':{'status':'success','answer_f1':1,'abstained':False}}}
    s=m.summarize(cases,'oracle')
    assert s['abstention_transitions']['error_to_answer']['n']==1
    assert s['abstention_transitions']['abstain_to_answer']['n']==0
    assert s['abstention_transitions']['answer_to_answer']['n']==0
    assert sum(t['contribution_to_overall_mean_delta'] for t in s['abstention_transitions'].values())==1
