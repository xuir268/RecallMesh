"""Experimental typed entity routing over a caller-owned adjacency snapshot.

Build once against matching dense node IDs. Speaker/alias provenance is supplied
by the caller. This index is not a learned coreference resolver or thread-safe
mutable store; rebuild it when records change.
"""
from collections import defaultdict
import re
import numpy as np
from .lexical import LexicalIndex, stable_topk


class EntityAwareRetriever:
    def __init__(self, adjacency, entities, aliases=None):
        self.adjacency = adjacency
        self.topic = LexicalIndex()
        self.roles = defaultdict(lambda: defaultdict(set))
        self.names = {}
        for entity in entities:
            self._register(entity, entity)
        for alias, entity in (aliases or {}).items():
            if entity not in self.names.values():
                raise ValueError('Alias target must be a declared entity')
            self._register(alias, entity)
        self.patterns = [(re.compile(r'(?<!\w)'+re.escape(name)+r'(?!\w)', re.I), entity)
                         for name, entity in sorted(self.names.items(), key=lambda p:(-len(p[0]),p[0]))]

    def _register(self, name, entity):
        if not isinstance(name,str) or not name.strip():
            raise ValueError('Entity names must be nonempty strings')
        key = name.casefold()
        if key in self.names and self.names[key] != entity:
            raise ValueError('Ambiguous alias')
        self.names[key] = entity

    def _matches(self, text):
        matches=[];occupied=set()
        for pattern, entity in self.patterns:
            for match in pattern.finditer(text):
                span=set(range(match.start(),match.end()))
                if not span & occupied:
                    matches.append((match.start(),match.end(),entity));occupied.update(span)
        return matches

    def add(self, node, body, speaker=None):
        if type(node) is not int or node != len(self.topic.lengths)+1:
            raise ValueError('Records must match dense node IDs in order')
        if speaker is not None and speaker not in self.names.values():
            raise ValueError('Unknown speaker entity')
        if not isinstance(body,str):
            raise ValueError('Record body must be a string')
        self.topic.add(body)
        if speaker is not None:
            self.roles[speaker]['authored_by'].add(node)
        for _,_,entity in self._matches(body):
            self.roles[entity]['mentions'].add(node)

    def search(self, query, tick, budget=16, graph_weight=.4, hops=2, seed_count=8, use_adjacency=True):
        return np.array([hit['node'] for hit in self.explain(query,tick,budget,graph_weight,hops,seed_count,use_adjacency)['results']], np.uint32)

    def explain(self, query, tick, budget=16, graph_weight=.4, hops=2, seed_count=8, use_adjacency=True):
        if len(self.topic.lengths) != len(self.adjacency.index.lengths):
            raise ValueError('Entity and adjacency records are out of sync; rebuild')
        matches = self._matches(query)
        entities = sorted({m[2] for m in matches})
        if len(entities) != 1:
            ids=self.adjacency.search(query,tick,budget,graph_weight,hops,seed_count)
            return {'mode':'adjacency_fallback','entities':entities,'results':[{'node':int(n)} for n in ids]}
        entity=entities[0]
        roles=self.roles.get(entity,{})
        candidates=sorted(set().union(*roles.values())) if roles else []
        if not candidates:
            ids=self.adjacency.search(query,tick,budget,graph_weight,hops,seed_count)
            return {'mode':'adjacency_fallback','entities':entities,'results':[{'node':int(n)} for n in ids]}
        topic_query=query
        for start,end,_ in sorted(matches,reverse=True):
            topic_query=topic_query[:start]+' '+topic_query[end:]
        scores=self.topic.scores(topic_query)
        indices=np.array(candidates,np.intp)-1
        lexical=scores[indices]
        graph=np.zeros_like(lexical)
        if use_adjacency:
            _,assoc,_,_,_=self.adjacency.components(query,tick,hops=hops,seed_count=seed_count)
            graph=assoc[indices]
        combined=lexical/max(float(lexical.max(initial=0)),1e-12)+graph_weight*graph/max(float(graph.max(initial=0)),1e-12)
        selected=indices[stable_topk(combined,budget)]
        return {'mode':'entity_topic_plus_adjacency' if use_adjacency else 'entity_topic_only','entities':entities,'topic_query':topic_query,'candidate_count':len(candidates),'results':[
            {'node':int(i)+1,'entity':entity,'link_types':sorted(role for role,nodes in roles.items() if int(i)+1 in nodes),'topic_score':float(scores[i])} for i in selected]}
