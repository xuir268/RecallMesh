import numpy as np
import pytest
from assoc_mem import Memory
from assoc_mem.lexical import LexicalIndex


def test_lexical_index_ranks_matching_document():
    index = LexicalIndex()
    index.add('apples orchard')
    index.add('train station ticket')
    assert np.argmax(index.scores('train ticket')) == 1
    assert index.scores('unknownword').sum() == 0


def test_memory_default_and_graph_explanation():
    with Memory(capacity=64, lambda_=0, graph_weight=2) as memory:
        seed = memory.remember('orchard apples')
        memory.remember('unrelated train')
        target = memory.remember('hidden destination')
        memory.engine.reinforce(np.array([seed], np.uint32),
                                np.array([target], np.uint32), 0)
        memory.flush()
        before = memory.stats()
        explanation = memory.explain_recall('apples', budget=2, k_seed=1, hops=1)
        hit = next(r for r in explanation['results'] if r['node'] == target)
        assert hit['path'] == [seed, target]
        assert hit['added_by_graph']
        assert hit['association_score'] > 0
        assert memory.stats()['updates'] == before['updates']
        memory.graph_weight = 0
        assert memory.recall('apples', budget=1)[0].id == seed


def test_invalid_retrieval_configuration():
    with pytest.raises(ValueError):
        Memory(retrieval='invalid')
    with pytest.raises(ValueError):
        Memory(graph_weight=float('nan'))
