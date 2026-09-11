"""Node table: text, embeddings, and bitemporal validity.

Bitemporal from the start because it cannot be retrofitted. A fact that
changes is never overwritten: the old node gets ``valid_to`` set and the new
one is inserted with a ``supersedes`` link. That relation is directed, which
is one more reason it lives here rather than in the symmetric edge store.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT    NOT NULL,
    writer      TEXT,
    layer       INTEGER NOT NULL DEFAULT 0,  -- 0 observed, 1 conjecture
    valid_from  INTEGER NOT NULL,
    valid_to    INTEGER,
    supersedes  INTEGER REFERENCES nodes(id)
);
CREATE INDEX IF NOT EXISTS idx_nodes_live ON nodes(valid_to) WHERE valid_to IS NULL;
"""


@dataclass
class Node:
    id: int
    content: str
    layer: int
    valid_from: int
    valid_to: int | None


class NodeStore:
    """Node id 0 is reserved: the engine uses a zero key for an empty slot."""

    def __init__(self, path: str = ":memory:", dim: int = 128):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        # Start ids at 1, never 0.
        self.db.execute(
            "INSERT OR IGNORE INTO sqlite_sequence(name, seq) VALUES ('nodes', 0)"
        )
        self.db.commit()
        self.dim = dim
        # Embeddings are a dense matrix indexed by node id so the engine can
        # borrow it directly. Row 0 stays zero and unused.
        self._emb = np.zeros((1, dim), dtype=np.float32)
        self._used_rows = 1

    # -- writes ---------------------------------------------------------------

    def add(self, content: str, emb: np.ndarray, tick: int, writer: str = "", layer: int = 0) -> int:
        cur = self.db.execute(
            "INSERT INTO nodes(content, writer, layer, valid_from) VALUES (?,?,?,?)",
            (content, writer, layer, tick),
        )
        self.db.commit()
        nid = int(cur.lastrowid)
        self._ensure(nid)
        self._emb[nid] = emb
        self._used_rows = max(self._used_rows, nid + 1)
        return nid

    def supersede(self, old_id: int, content: str, emb: np.ndarray, tick: int, **kw) -> int:
        """Evolution, not replacement: the old node stays readable as history."""
        new_id = self.add(content, emb, tick, **kw)
        self.db.execute("UPDATE nodes SET valid_to=? WHERE id=?", (tick, old_id))
        self.db.execute("UPDATE nodes SET supersedes=? WHERE id=?", (old_id, new_id))
        self.db.commit()
        return new_id

    def promote(self, nid: int) -> None:
        """Conjecture confirmed by a later observation; it now counts."""
        self.db.execute("UPDATE nodes SET layer=0 WHERE id=?", (nid,))
        self.db.commit()

    # -- reads ----------------------------------------------------------------

    def get(self, nid: int) -> Node | None:
        row = self.db.execute(
            "SELECT id, content, layer, valid_from, valid_to FROM nodes WHERE id=?", (nid,)
        ).fetchone()
        return Node(*row) if row else None

    def many(self, ids) -> list[Node]:
        if len(ids) == 0:
            return []
        q = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT id, content, layer, valid_from, valid_to FROM nodes WHERE id IN ({q})",
            [int(i) for i in ids],
        ).fetchall()
        by_id = {r[0]: Node(*r) for r in rows}
        return [by_id[int(i)] for i in ids if int(i) in by_id]

    @property
    def embeddings(self) -> np.ndarray:
        return self._emb[:self._used_rows]

    def __len__(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])

    def _ensure(self, nid: int) -> None:
        if nid < self._emb.shape[0]:
            return
        grown = max(nid + 1, self._emb.shape[0] * 2)
        buf = np.zeros((grown, self.dim), dtype=np.float32)
        buf[: self._emb.shape[0]] = self._emb
        self._emb = buf
