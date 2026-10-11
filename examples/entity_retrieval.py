"""Explicit entity provenance + typed routing; no model calls or default changes."""
import numpy as np
from assoc_mem import EdgeStore
from assoc_mem.lexical import LexicalIndex, AssociativeRetriever
from recallmesh import EntityAwareRetriever

index=LexicalIndex()
engine=EdgeStore(64)
adjacency=AssociativeRetriever(engine,index)
routing=EntityAwareRetriever(adjacency,['Mira','Nate'],aliases={'M':'Mira'})
records={}
for speaker,body in [('Mira','I keep turtles in the blue room.'),('Nate','I keep turtles in the red room.'),('Nate','Mira visits me on Fridays.')]:
    node=index.add(speaker+': '+body)
    routing.add(node,body,speaker)
    records[node]=body
    if node>1:engine.reinforce(np.array([node-1],np.uint32),np.array([node],np.uint32),1)
engine.freeze(1)
trace=routing.explain('Where does M keep turtles?',tick=1,budget=2)
for hit in trace['results']:
    print(hit['node'],hit['link_types'],records[hit['node']])
# For a model: turn these original records into Evidence objects and call the
# existing CliProvider.answer(). Do not store its answer as a new source fact.
