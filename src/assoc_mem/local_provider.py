"""Local llama.cpp OpenAI-compatible transport; no cloud credentials needed."""
from dataclasses import asdict
import json
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from .agent import ANSWER_SCHEMA, INSTRUCTIONS, Completion, parse_answer


class LocalProvider:
    def __init__(self, model, url='http://127.0.0.1:8080', timeout=180):
        parsed = urlparse(url)
        if parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('Local provider requires a loopback HTTP URL')
        self.model = model; self.url = url.rstrip('/'); self.timeout = timeout

    def structured(self, instructions, data, schema):
        started = time.perf_counter()
        body = {'model': self.model, 'messages': [
            {'role': 'system', 'content': instructions},
            {'role': 'user', 'content': 'INPUT JSON:\n' + json.dumps(data, ensure_ascii=False)}],
            'temperature': 0, 'seed': 42, 'max_tokens': 1024,
            'chat_template_kwargs': {'enable_thinking': False},
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'result', 'schema': schema}}}
        request = Request(self.url+'/v1/chat/completions', data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        choice = result['choices'][0]
        if choice.get('finish_reason') != 'stop' or choice['message'].get('tool_calls'):
            raise RuntimeError('Local completion incomplete or attempted tool use')
        return {'data': json.loads(choice['message']['content']), 'metadata': {
            'model': result.get('model', self.model), 'seconds': time.perf_counter()-started,
            'usage': result.get('usage', {}), 'estimated_cost_usd': None}}

    def answer(self, question, evidence):
        result = self.structured(INSTRUCTIONS, {'question': question,
            'evidence': [asdict(e) for e in evidence]}, ANSWER_SCHEMA)
        payload = parse_answer(json.dumps(result['data']))
        allowed = {e.id for e in evidence}
        return Completion(**payload, **result['metadata'],
            invalid_evidence_ids=[i for i in payload['evidence_ids'] if i not in allowed])
