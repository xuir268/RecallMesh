import pytest
from assoc_mem.agent import Completion, Evidence
from assoc_mem.pipeline import EvidencePipeline, fit_records, evidence_bytes


class Agent:
    def __init__(self):
        self.provider = self; self.calls = []; self.plans = 0; self.bad = False
    def context(self, query, mode, budget, hops):
        self.calls.append(query)
        return ([Evidence('a','Mira calls the token silver pebble.')] if len(self.calls)==1 else
                [Evidence('b','The silver pebble is in the blue drawer.')]), {}
    def structured(self, instructions, data, schema):
        if 'selected_ids' in schema['properties']:
            self.plans += 1
            return {'data': {'selected_ids': ['a'] if self.plans==1 else ['a','b'],
                             'queries': ['silver pebble location'] if self.plans==1 else [],
                             'missing': ''}, 'metadata': {}}
        return {'data': {'supported': not self.bad, 'reason':'checked'}, 'metadata':{}}
    def answer(self, question, evidence):
        assert [e.id for e in evidence] == ['a','b']
        return Completion('blue drawer',['a','b'],'linked facts','fake',0)


def test_second_query_retains_bridge_and_bounds_calls():
    agent=Agent(); result=EvidencePipeline(agent,candidates=4,final_records=2).ask('token location?')
    assert agent.calls==['token location?','silver pebble location']
    assert result['response']['answer']=='blue drawer'
    assert result['metrics']['model_calls']==4
    assert [e['id'] for e in result['evidence']]==['a','b']
    assert not result['support_rejected']


def test_failed_check_abstains_and_preserves_draft():
    agent=Agent(); agent.bad=True
    result=EvidencePipeline(agent,candidates=4,final_records=2).ask('where?')
    assert result['response']['answer']=='Not enough information'
    assert result['draft']['answer']=='blue drawer'


def test_compression_preserves_verbatim_unicode_and_budget():
    records=[Evidence('a','Été: 12:00 — not before 09:00.'),Evidence('b','long '*200)]
    limit=evidence_bytes(records[:1])
    result=fit_records(records,limit,2)
    assert result==records[:1] and evidence_bytes(result)<=limit
    assert fit_records(records,2,2)==[]


def test_invalid_plan_id_is_not_accepted():
    agent=Agent()
    agent.structured=lambda *args: {'data':{'selected_ids':['invented'],'queries':[],'missing':''},'metadata':{}}
    with pytest.raises(ValueError,match='invented'):
        EvidencePipeline(agent).ask('where?')


@pytest.mark.parametrize('kwargs',[{'rounds':4},{'candidates':2,'final_records':4}, {'context_bytes':1}])
def test_config_limits(kwargs):
    with pytest.raises(ValueError):EvidencePipeline(Agent(),**kwargs)


def test_plan_schema_limits_source_ids_and_empty_context():
    from assoc_mem.pipeline import constrained_plan_schema
    schema=constrained_plan_schema([Evidence('A','a'),Evidence('B','b')],1)
    selected=schema['properties']['selected_ids']
    assert selected['items']['enum']==['A','B'] and selected['maxItems']==1
    assert schema['properties']['queries']['maxItems']==2
    empty=constrained_plan_schema([],4)
    assert empty['properties']['selected_ids']['maxItems']==0


def test_invalid_plan_remains_in_audit():
    agent=Agent()
    agent.structured=lambda *args: {'data':{'selected_ids':['invented'],'queries':[],'missing':''},'metadata':{}}
    pipeline=EvidencePipeline(agent)
    with pytest.raises(ValueError):pipeline.ask('where?')
    assert pipeline.stages[-1]['data']['selected_ids']==['invented']
    assert pipeline.stages[-1]['validated'] is False
