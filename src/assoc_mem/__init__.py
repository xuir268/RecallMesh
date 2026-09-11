"""Associative memory layer with self-forming Hebbian structure.

Associations are not synthesised. They form from usage: memories retrieved
together have their connection strengthened, unused connections decay, and the
graph's shape is a record of what actually co-occurred rather than of what an
LLM guessed belonged together.

    mem = Memory()
    mem.remember("deploys on Fridays make him nervous")
    mem.remember("the Friday outage took four hours")
    hits = mem.recall("release schedule")
"""
from __future__ import annotations

import numpy as np

from ._engine import EdgeStore
from .coactivate import CoActivation
from .embed import Embedder, HashEmbedder
from .lexical import LexicalIndex, AssociativeRetriever
from .retrieve import axis as _axis
from .retrieve import recall as _recall
from .retrieve import sectors as _sectors
from .retrieve import seed as _seed
from .retrieve import unexplored as _unexplored
from .store import Node, NodeStore

__all__ = ["Memory", "Node", "NodeStore", "Embedder", "HashEmbedder", "EdgeStore"]


class Memory:
    """The surface agents talk to.

    A tick is one session by default, giving roughly a four-month horizon at
    daily use. Ticking per retrieval instead makes the graph aggressively
    recency-biased; that is a product decision rather than an engine one, so
    ticks advance explicitly rather than from a clock.
    """

    def __init__(
        self,
        path: str = ":memory:",
        embedder: Embedder | None = None,
        capacity: int = 1 << 20,
        lambda_: float = 0.05,
        eta: float = 1.0,
        floor: float = 1e-3,
        background: bool = False,
        retrieval: str = "lexical",
        graph_weight: float = 0.4,
    ):
        if retrieval not in {"lexical", "embedding"}:
            raise ValueError("retrieval must be lexical or embedding")
        if not np.isfinite(graph_weight) or graph_weight < 0:
            raise ValueError("graph_weight must be finite and nonnegative")
        self.retrieval = retrieval
        self.graph_weight = graph_weight
        self.embedder = embedder or HashEmbedder()
        self.store = NodeStore(path, dim=self.embedder.dim)
        self.engine = EdgeStore(capacity, lambda_, eta, floor)
        self.lexical_index = LexicalIndex()
        self.associative = AssociativeRetriever(self.engine, self.lexical_index)
        self.queue = CoActivation(self.engine)
        self.tick = 0
        self._dirty = False
        if background:
            self.queue.start()

    # -- writes ---------------------------------------------------------------

    def remember(self, content: str, writer: str = "", layer: int = 0) -> int:
        emb = self.embedder.encode([content])[0]
        nid = self.store.add(content, emb, self.tick, writer=writer, layer=layer)
        self.lexical_index.add(content)
        self._dirty = True
        return nid

    def supersede(self, old_id: int, content: str, **kw) -> int:
        """Record that a fact changed. The old one stays readable as history."""
        emb = self.embedder.encode([content])[0]
        nid = self.store.supersede(old_id, content, emb, self.tick, **kw)
        self.lexical_index.add(content)
        self._dirty = True
        return nid

    # -- reads ----------------------------------------------------------------

    def recall(self, query: str, *, budget: int = 16, hops: int = 2, k_seed: int = 8):
        """Retrieve, then queue the co-activation this retrieval implies.

        The queueing closes the Hebbian loop: every recall is also a small
        amount of learning about which memories belong together.
        """
        self._sync()
        if budget <= 0:
            return []
        if self.retrieval == "lexical":
            ids = self.associative.search(query, self.tick, budget=budget,
                graph_weight=self.graph_weight, hops=hops, seed_count=k_seed)
            nodes = [n for n in self.store.many(ids)
                     if n.valid_to is None and n.layer == 0]
        else:
            q = self.embedder.encode([query])[0]
            nodes, _ = _recall(
                self.engine, self.store, q,
                tick=self.tick, k_seed=k_seed, hops=hops, budget=budget,
            )
        if len(nodes) > 1:
            self.queue.record([n.id for n in nodes], self.tick)
        return nodes

    def explain_recall(self, query: str, *, budget: int = 16, hops: int = 2, k_seed: int = 8):
        """Read-only lexical/association diagnostics; does not reinforce the graph.

        Paths are traversed witnesses, not causal explanations of every score.
        This method exposes lexical mode, regardless of the recall mode setting.
        """
        self._sync()
        result = self.associative.explain(query, self.tick, budget, self.graph_weight,
                                          hops=hops, seed_count=k_seed)
        visible = []
        for item in result['results']:
            node = self.store.get(item['node'])
            if node and node.valid_to is None and node.layer == 0:
                item['content'] = node.content
                visible.append(item)
        result['results'] = visible
        return result

    def unexplored(self, around: str, k: int = 32) -> np.ndarray:
        """Pairs that are semantically close but were never co-activated."""
        self._sync()
        q = self.embedder.encode([around])[0]
        ids, _ = _seed(self.store.embeddings, q, 8)
        return _unexplored(self.engine, self.store.embeddings, ids, tick=self.tick, k=k)

    def axis(self, nodes):
        """Ordering along the local principal direction, plus its edge mass.

        Low mass means the line is a geometric coincidence rather than a path
        the conversation actually travels.
        """
        self._sync()
        direction, ordered = _axis(self.store.embeddings, nodes)
        mass = self.engine.edge_mass_along(np.asarray(ordered, np.uint32), self.tick)
        return direction, ordered, float(mass)

    def sectors(self, node: int, n_sectors: int = 12):
        """Angular occupancy around a node; empty sectors are unexplored directions."""
        self._sync()
        nbrs, _ = self.engine.neighbors(node, 1e-3, self.tick, 256)
        return _sectors(self.store.embeddings, node, nbrs, n_sectors)

    # -- lifecycle ------------------------------------------------------------

    def advance(self) -> int:
        """End the current session. Decay is measured in ticks."""
        self.flush()
        self.tick += 1
        return self.tick

    def flush(self) -> None:
        self._sync()
        self.queue.flush(self.tick)
        self._dirty = False

    def stats(self) -> dict:
        s = self.engine.stats()
        s["nodes"] = len(self.store)
        s["tick"] = self.tick
        return s

    def close(self) -> None:
        self.queue.stop()
        self.flush()
        self.store.db.close()

    def _sync(self) -> None:
        """Republish if nodes were added, so the engine sees current embeddings."""
        if self._dirty:
            self.engine.attach_embeddings(self.store.embeddings)
            self.engine.freeze(self.tick)
            self._dirty = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
