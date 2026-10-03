"""Retrieval: ANN seeds, spreading activation, ranking.

Brute-force cosine is fine at PoC scale -- a numpy matmul over the whole
embedding matrix beats an index until well past a million nodes, and swapping
in faiss later changes only ``seed``.
"""
from __future__ import annotations

import numpy as np


def seed(emb: np.ndarray, query: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Top-k nodes by cosine. Row 0 is reserved and never returned."""
    if emb.shape[0] <= 1 or k <= 0:
        return np.empty(0, np.uint32), np.empty(0, np.float32)
    sims = emb[1:] @ query
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return (idx + 1).astype(np.uint32), np.maximum(sims[idx], 0.0).astype(np.float32)


def recall(engine, store, query_emb, *, tick=0, k_seed=8, hops=2, cutoff=1e-3, budget=16):
    """Seed by similarity, spread by association, rank, truncate.

    The two-hop spread is the whole point: it reaches memories that sit far
    from the query in embedding space but were co-activated with something
    near it. Those are the ones similarity-only retrieval cannot find.
    """
    ids, sims = seed(store.embeddings, query_emb, k_seed)
    if len(ids) == 0:
        return [], np.empty(0, np.float32)

    nodes, scores = engine.activate(ids, sims, hops=hops, cutoff=cutoff, tick=tick, cap=budget * 4)
    merged = {int(n): float(w) for n, w in zip(nodes, scores)}
    for n, w in zip(ids, sims):
        merged[int(n)] = max(merged.get(int(n), 0.0), float(w))
    ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
    nodes = np.asarray([n for n, _ in ranked], dtype=np.uint32)
    scores = np.asarray([w for _, w in ranked], dtype=np.float32)

    recs = store.many(nodes)
    live = [(n, s) for n, s in zip(recs, scores) if n.valid_to is None and n.layer == 0]
    live = live[:budget]
    return [n for n, _ in live], np.array([s for _, s in live], dtype=np.float32)


def unexplored(engine, emb, nodes, *, tick=0, k=32, tau=0.35, eps=1e-3):
    """Structural holes: semantically close, never co-activated.

    Nothing is synthesised. This points at a place where a perspective is
    missing and hands that to the model; the reasoning happens there, in front
    of the user, not silently in storage.
    """
    nodes = np.asarray(nodes, dtype=np.uint32)
    if len(nodes) == 0 or emb.shape[0] <= 1 or k <= 0:
        return np.empty((0, 2), np.uint32)

    sims = emb[nodes] @ emb[1:].T
    kk = min(k, sims.shape[1])
    cand = np.argpartition(-sims, kk - 1, axis=1)[:, :kk] + 1

    a = np.repeat(nodes, kk).astype(np.uint32)
    b = cand.ravel().astype(np.uint32)
    keep = a != b
    a, b = a[keep], b[keep]

    close = np.take_along_axis(sims, cand - 1, axis=1).ravel()[keep] > tau
    w = engine.weights_for_pairs(a, b, tick)
    hole = close & (w < eps)
    return np.unique(np.sort(np.stack([a[hole], b[hole]], axis=1), axis=1), axis=0)


def axis(emb: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local principal direction and the nodes ordered along it.

    Returned ordering is a candidate only. Validate it with
    ``engine.edge_mass_along``: embedding spaces are full of accidental
    collinearity, so geometry proposes and usage confirms.
    """
    nodes = np.asarray(nodes, dtype=np.uint32)
    if len(nodes) < 2:
        return np.zeros(emb.shape[1], np.float32), nodes
    pts = emb[nodes]
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0].astype(np.float32)
    return direction, nodes[np.argsort(centred @ direction)]


def sectors(emb: np.ndarray, centre: int, neighbours: np.ndarray, n_sectors: int = 12):
    """Angular occupancy in the local tangent plane around a node.

    Ambient angles in high dimensions are meaningless -- concentration of
    measure puts nearly every pair near 90 degrees -- so angles are only
    computed after projecting into the neighbourhood's own top-2 PCA frame.
    Empty sectors are directions the conversation has not gone.
    """
    neighbours = np.asarray(neighbours, dtype=np.uint32)
    if len(neighbours) < 3:
        return np.zeros(n_sectors, np.int64), np.empty(0, np.float32)
    rel = emb[neighbours] - emb[centre]
    _, _, vt = np.linalg.svd(rel - rel.mean(axis=0), full_matrices=False)
    coords = rel @ vt[:2].T
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    bins = np.clip(np.digitize(theta, np.linspace(-np.pi, np.pi, n_sectors + 1)) - 1, 0, n_sectors - 1)
    return np.bincount(bins, minlength=n_sectors), theta.astype(np.float32)
