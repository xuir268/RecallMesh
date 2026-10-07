"""Validated, versioned JSON configuration; paths resolve relative to the file."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

DEFAULT_CONFIG = {
    'version': 1,
    'storage': {'path': '.context-memory/memory.sqlite3', 'max_database_bytes': 67108864, 'max_checkpoint_bytes': 67108864},
    'limits': {'max_nodes': 10000, 'max_text_bytes': 16777216,
               'max_record_bytes': 16384, 'max_request_bytes': 65536},
    'graph': {'initial_capacity': 8192, 'max_capacity': 262144,
              'allow_growth': True, 'edge_policy': 'star',
              'lambda': .05, 'eta': 1.0, 'floor': .001},
    'retention': {'overflow': 'reject', 'ttl_ticks': 0},
    'retrieval': {'default_limit': 8, 'max_limit': 64, 'max_context_bytes': 12000, 'hops': 2, 'graph_weight': .4},
}


# New CLI setups use the evaluated conservative profile. Keep v1 defaults
# above for backward-compatible loading of existing partial configurations.
PRODUCT_CONFIG = deepcopy(DEFAULT_CONFIG)
PRODUCT_CONFIG['graph'].update(edge_policy='adjacent', **{'lambda': 0.0})


class ConfigError(ValueError):
    pass


class MemoryConfig:
    def __init__(self, data=None, base_dir=None):
        self.data = deepcopy(DEFAULT_CONFIG)
        supplied = {} if data is None else data
        if not isinstance(supplied, dict):raise ConfigError('config must be a JSON object')
        for key,value in supplied.items():
            if key not in self.data:raise ConfigError('unknown config field: '+key)
            if isinstance(self.data[key], dict):
                if not isinstance(value,dict):raise ConfigError(key+' must be an object')
                if set(value)-set(self.data[key]):raise ConfigError('unknown fields in '+key)
                self.data[key].update(value)
            else:self.data[key]=value
        d=self.data
        if type(d['version']) is not int or d['version']!=1:raise ConfigError('unsupported config version')
        def integer(section,key,minimum,maximum):
            v=d[section][key]
            if type(v) is not int or not minimum<=v<=maximum:
                raise ConfigError(f'{section}.{key} must be an integer in {minimum}..{maximum}')
        integer('storage','max_database_bytes',65536,1<<40)
        integer('storage','max_checkpoint_bytes',65536,1<<40)
        for key,lo,hi in [('max_nodes',1,1000000),('max_text_bytes',1,1<<40),
                           ('max_record_bytes',1,1<<24),('max_request_bytes',1024,1<<25)]:
            integer('limits',key,lo,hi)
        for key in ['initial_capacity','max_capacity']:
            integer('graph',key,2,1<<26)
            if d['graph'][key] & (d['graph'][key]-1):raise ConfigError('graph capacities must be powers of two')
        if d['graph']['initial_capacity']>d['graph']['max_capacity']:raise ConfigError('initial capacity exceeds maximum')
        if type(d['graph']['allow_growth']) is not bool:raise ConfigError('allow_growth must be boolean')
        if d['graph']['edge_policy'] not in ('star','combined','adjacent'):raise ConfigError('invalid edge_policy')
        import math
        for section,key,lo,hi in [('graph','lambda',0,1),('graph','eta',0,1e6),('graph','floor',1e-12,1),('retrieval','graph_weight',0,100)]:
            v=d[section][key]
            if type(v) not in {float,int} or not math.isfinite(v) or not lo<=v<=hi:raise ConfigError(f'invalid {section}.{key}')
        integer('retention','ttl_ticks',0,(1<<32)-1)
        if d['retention']['overflow'] not in ('reject','oldest'):raise ConfigError('overflow must be reject or oldest')
        integer('retrieval','default_limit',1,64);integer('retrieval','max_limit',1,64)
        integer('retrieval','hops',0,8);integer('retrieval','max_context_bytes',2,1<<24)
        if d['retrieval']['default_limit']>d['retrieval']['max_limit']:raise ConfigError('default_limit exceeds max_limit')
        if d['limits']['max_record_bytes']>d['limits']['max_text_bytes']:raise ConfigError('record budget exceeds text budget')
        if not isinstance(d['storage']['path'],str) or not d['storage']['path'].strip() or d['storage']['path']==':memory:':
            raise ConfigError('storage.path must be a persistent filename')
        self.base_dir=Path(base_dir or '.').resolve()
        self.path=(self.base_dir/Path(d['storage']['path']).expanduser()).resolve()
        canonical=deepcopy(d);canonical['storage']['path']=str(self.path)
        self.fingerprint=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
        self.learning_fingerprint=json.dumps({k:d['graph'][k] for k in ['lambda','eta','floor']},sort_keys=True)

    @classmethod
    def load(cls,path='memory.json'):
        p=Path(path).resolve()
        try:return cls(json.loads(p.read_text()),p.parent)
        except (OSError,json.JSONDecodeError) as exc:raise ConfigError(str(exc)) from exc

    def __getitem__(self,key):return self.data[key]
