"""Check benchmark accounting independently of real dataset outcomes."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('locomo_benchmark',Path(__file__).parents[1]/'benchmarks/locomo_retrieval.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

def test_evidence_metrics_are_not_answer_metrics():
    assert bench.measure(['D1:1','D1:2'],['D1:1','D1:1']) == {
        'evidence_recall':.5,'any_evidence':1,'all_evidence':0}
    assert bench.measure([],['D1:1']) is None
    assert bench.measure(['D99:99'],['D1:1'])['evidence_recall']==0

def test_evidence_normalization():
    assert bench.evidence_ids(['D1:1','D1:1','D2:3'])==['D1:1','D2:3']

def test_bm25_ranks_known_evidence():
    model=bench.BM25(['cat food dinner','mountain hiking boots','cat food'])
    assert model.rank('hiking',1).tolist()==[2]

def test_empty_evidence_excluded_from_denominator():
    rows=[{'metrics':bench.measure(['D1:1'],['D1:1']),'latency_ms':1,'retrieved':['D1:1']},
          {'metrics':None,'latency_ms':2,'retrieved':[]}]
    result=bench.aggregate(rows)
    assert result['questions']==2 and result['scored_questions']==1
    assert result['evidence_recall']==1
