import numpy as np
import pytest
from assoc_mem import EdgeStore
from assoc_mem.lexical import LexicalIndex,AssociativeRetriever
from assoc_mem.entity import EntityAwareRetriever


def fixture():
    index=LexicalIndex();graph=EdgeStore(64);adj=AssociativeRetriever(graph,index)
    entity=EntityAwareRetriever(adj,['Mira','Nate'],{'M':'Mira'})
    for body,speaker in [('I keep turtles in the blue room.','Mira'),('I keep turtles in the red room.','Nate'),('Mira visits me on Fridays.','Nate')]:
        n=index.add(speaker+': '+body);entity.add(n,body,speaker)
    graph.reinforce(np.array([1,2],np.uint32),np.array([2,3],np.uint32),1);graph.freeze(1)
    return entity,adj,graph


def test_alias_routes_to_original_source_with_typed_provenance():
    e,_,g=fixture();before=g.stats()
    trace=e.explain('Where does M keep turtles?',1)
    assert trace['entities']==['Mira']
    assert trace['results'][0]['node']==1
    assert trace['results'][0]['link_types']==['authored_by']
    assert all(r['node'] in [1,3] for r in trace['results'])
    assert g.stats()==before


def test_ambiguous_pronoun_and_multiple_entities_fall_back_exactly():
    e,adj,_=fixture()
    for q in ['Where does he keep turtles?','What do Mira and Nate keep?']:
        assert e.search(q,1).tolist()==adj.search(q,1).tolist()
        assert e.explain(q,1)['mode']=='adjacency_fallback'


def test_rejects_alias_collision_and_stale_node_alignment():
    e,adj,_=fixture()
    with pytest.raises(ValueError,match='Ambiguous'):
        EntityAwareRetriever(adj,['Mira','Nate'],{'Mira':'Nate'})
    with pytest.raises(ValueError,match='dense'):
        e.add(5,'text','Mira')
    adj.index.add('new record')
    with pytest.raises(ValueError,match='sync'):
        e.search('Mira turtles',1)
