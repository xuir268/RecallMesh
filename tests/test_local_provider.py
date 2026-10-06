import io
import json
import pytest
from assoc_mem.agent import Evidence
from assoc_mem.local_provider import LocalProvider


def test_local_transport_validates_tool_use_and_completion(monkeypatch):
    import assoc_mem.local_provider as module
    result={'model':'test-local','usage':{'prompt_tokens':42},'choices':[
        {'finish_reason':'stop','message':{'content':json.dumps({'answer':'blue',
         'evidence_ids':['A'],'rationale':'explicit'})}}]}
    requests=[]
    def request(req,timeout):
        requests.append(json.loads(req.data));return io.BytesIO(json.dumps(result).encode())
    monkeypatch.setattr(module,'urlopen',request)
    response=LocalProvider('test-local').answer('color?',[Evidence('A','blue')])
    assert response.answer=='blue' and response.usage['prompt_tokens']==42
    assert requests[0]['response_format']['type']=='json_schema'
    assert requests[0]['chat_template_kwargs']['enable_thinking'] is False
    result['choices'][0]['finish_reason']='length'
    with pytest.raises(RuntimeError):LocalProvider('test-local').answer('color?',[])


def test_local_provider_does_not_send_memory_to_remote_endpoint():
    with pytest.raises(ValueError):LocalProvider('x','https://example.com')
