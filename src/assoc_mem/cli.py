"""A small chat with either model, with memory above the model.

Run: python examples/agent_chat.py --backend codex --question 'Where does Mira keep the token?'
Or omit --question for an interactive terminal. /remember TEXT adds observations;
/session advances time. Answers are not automatically treated as factual memories.
"""
import argparse
import json
from assoc_mem import Memory
from assoc_mem.agent import CliProvider, MemoryAgent
from assoc_mem.pipeline import EvidencePipeline
from assoc_mem.local_provider import LocalProvider


def _chat_main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['codex','claude','local'],default='codex')
    parser.add_argument('--model')
    parser.add_argument('--config',help='persistent framework config; omitting uses the RAM demo')
    parser.add_argument('--question')
    parser.add_argument('--budget',type=int)
    parser.add_argument('--capacity',type=int,default=1<<18,help='mutable graph slot capacity')
    parser.add_argument('--mode',choices=['none','lexical','associative','connected'],default='associative')
    parser.add_argument('--edge-policy',choices=['combined','adjacent','star'],default='combined')
    parser.add_argument('--pipeline', choices=['single','bounded'], default='bounded')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--context-bytes', type=int, default=12000)
    parser.add_argument('--local-url', default='http://127.0.0.1:8080')
    parser.add_argument('--controller', choices=['codex','claude','local'])
    parser.add_argument('--controller-model')
    parser.add_argument('--facts', help='UTF-8 JSON list of observation strings; replaces demo facts')
    args=parser.parse_args(argv)
    if args.backend == 'local' and not args.model:
        parser.error('--backend local needs --model matching the server alias')
    if args.controller == 'local' and not (args.controller_model or (args.model if args.backend=='local' else None)):
        parser.error('--controller local needs --controller-model')
    def provider(backend, model):
        return LocalProvider(model, args.local_url) if backend=='local' else CliProvider(backend,model)

    if args.config:
        from .framework import ManagedMemory, FrameworkAgent
        resource=ManagedMemory(args.config)
        if args.budget is None:args.budget=resource.config['retrieval']['default_limit']
        args.budget=min(args.budget,resource.config['retrieval']['max_limit'])
        args.candidates=max(args.budget,min(args.candidates,resource.config['retrieval']['max_limit']))
        args.context_bytes=min(args.context_bytes,resource.config['retrieval']['max_context_bytes'])
    else:
        if args.budget is None:args.budget=2
        resource=Memory(capacity=args.capacity)
    with resource as memory:
        agent=(FrameworkAgent(memory,provider(args.backend,args.model)) if args.config else
               MemoryAgent(memory,provider(args.backend,args.model),edge_policy=args.edge_policy))
        controller=provider(args.controller,args.controller_model or (args.model if args.controller==args.backend else None)) if args.controller else None
        pipeline=EvidencePipeline(agent,controller=controller,rounds=args.rounds,candidates=args.candidates,
                                  final_records=args.budget,context_bytes=args.context_bytes,
                                  review_bytes=args.context_bytes if args.config else 32000)
        def ask(question):
            return pipeline.ask(question,mode=args.mode) if args.pipeline=='bounded' else agent.ask(question,mode=args.mode,budget=args.budget)
        demo = [
            'The garden gate was painted green.',
            'The museum opens at nine.',
            'Friday lunch will be tomato soup.',
            'The bicycle needs a new rear tire.',
            'The parcel arrives on Tuesday.',
            'The library has a new history section.',
        ] + ['Mira calls her security token the silver pebble.',
             'The silver pebble is stored in the blue drawer in the workshop.']
        if args.facts:
            from pathlib import Path
            demo=json.loads(Path(args.facts).read_text())
            if not isinstance(demo,list) or not all(isinstance(x,str) for x in demo):
                parser.error('--facts must contain a JSON list of strings')
        if args.config and not args.facts:demo=[]
        for i,fact in enumerate(demo):
            if i and i%6==0: memory.advance()
            agent.observe(fact)
        if args.question:
            print(json.dumps(ask(args.question),indent=2));return
        print('Memory chat: /remember TEXT, /session, /quit. '+
              ('Persistent storage: '+str(memory.config.path) if args.config else 'Memory is in RAM for this session.'))
        while True:
            try:query=input('You: ').strip()
            except (EOFError,KeyboardInterrupt):break
            if query=='/quit':break
            if query=='/session':memory.advance();continue
            if query.startswith('/remember '):agent.observe(query[10:]);continue
            if query:print(json.dumps(ask(query),indent=2))

def main():
    import sys
    from .memory_cli import COMMANDS, main as memory_main
    if len(sys.argv)==1 or sys.argv[1] in {'--help','-h'}:
        return memory_main(['--help'])
    if len(sys.argv)>1 and sys.argv[1] in COMMANDS:
        return memory_main(sys.argv[1:])
    return _chat_main(sys.argv[2:] if len(sys.argv)>1 and sys.argv[1]=='chat' else None)

if __name__=='__main__':main()
