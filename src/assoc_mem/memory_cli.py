"""JSON memory commands and a line-oriented stdio service for any agent."""
import argparse
import json
import sys
import sqlite3
from pathlib import Path
from .config import PRODUCT_CONFIG,MemoryConfig,ConfigError
from .framework import ManagedMemory,QuotaExceeded

COMMANDS={'init','serve','remember','recall','get','forget','associate','advance','stats','compact','checkpoint'}


def dispatch(memory,method,params):
    if method not in COMMANDS-{'init','serve'}:raise ValueError('unknown method')
    if not isinstance(params,dict):raise ValueError('params must be an object')
    expected={'remember':{'text','writer'},'recall':{'query','limit','hops','mode'},
              'get':{'id'},'forget':{'id'},'associate':{'ids'},'advance':{'steps'},'stats':set(),'compact':set(),'checkpoint':set()}
    if set(params)-expected[method]:raise ValueError('unknown parameters')
    if method in {'get','forget'}:params={'nid':params['id']}
    return getattr(memory,method)(**params)


def response(memory,request):
    request_id=request.get('id') if isinstance(request,dict) else None
    try:
        if not isinstance(request,dict) or set(request)-{'id','method','params'}:raise ValueError('invalid request object')
        method=request.get('method')
        if not isinstance(method,str):raise ValueError('method must be a string')
        result=dispatch(memory,method,request.get('params',{}))
        return {'id':request_id,'ok':True,'result':result}
    except (QuotaExceeded,ConfigError,ValueError,TypeError,KeyError,sqlite3.Error) as exc:
        code='quota_exceeded' if isinstance(exc,QuotaExceeded) else 'config_error' if isinstance(exc,ConfigError) else 'storage_error' if isinstance(exc,sqlite3.Error) else 'invalid_request'
        return {'id':request_id,'ok':False,'error':{'code':code,'message':str(exc)}}


def serve(memory):
    limit=memory.config['limits']['max_request_bytes']
    stream=sys.stdin.buffer
    while True:
        raw=stream.readline(limit+1)
        if not raw:return
        if len(raw)>limit:
            while not raw.endswith(b'\n'):
                raw=stream.readline(limit+1)
                if not raw:break
            result={'id':None,'ok':False,'error':{'code':'request_too_large','message':'Request byte budget exceeded'}}
        else:
            try:result=response(memory,json.loads(raw))
            except (ValueError,UnicodeDecodeError) as exc:
                result={'id':None,'ok':False,'error':{'code':'invalid_json','message':str(exc)}}
        print(json.dumps(result,ensure_ascii=False),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for cmd in sorted(COMMANDS):
        p=sub.add_parser(cmd);p.add_argument('--config',default='memory.json')
        if cmd=='remember':p.add_argument('--text',required=True);p.add_argument('--writer',default='')
        if cmd=='recall':
            p.add_argument('--query',required=True);p.add_argument('--limit',type=int);p.add_argument('--hops',type=int)
            p.add_argument('--mode',choices=['lexical','associative','none'],default='associative')
        if cmd in {'get','forget'}:p.add_argument('--id',type=int,required=True)
        if cmd=='associate':p.add_argument('--ids',type=int,nargs='+',required=True)
        if cmd=='advance':p.add_argument('--steps',type=int,default=1)
    args=parser.parse_args(argv)
    try:
        if args.command=='init':
            path=Path(args.config);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x') as f:json.dump(PRODUCT_CONFIG,f,indent=2);f.write('\n')
            print(json.dumps({'ok':True,'result':{'config':str(path.resolve())}}));return
        config=MemoryConfig.load(args.config)
        with ManagedMemory(config) as memory:
            if args.command=='serve':serve(memory);return
            params={k:v for k,v in vars(args).items() if k not in {'command','config'} and v is not None}
            output=response(memory,{'method':args.command,'params':params})
            print(json.dumps(output,ensure_ascii=False))
            if not output['ok']:raise SystemExit(2)
    except (OSError,ConfigError,QuotaExceeded,sqlite3.Error) as exc:
        print(json.dumps({'ok':False,'error':{'code':'startup_error','message':str(exc)}}))
        raise SystemExit(2)
