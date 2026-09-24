"""Co-activation queue and drain thread.

A retrieval returning K memories implies K(K-1)/2 pairwise reinforcements --
45 edge updates at K=10. Doing that synchronously would put write
amplification on the read path, so retrieval only enqueues the activated set
and a background thread folds batches into the engine.

Freeze is triggered explicitly from here rather than by a timer or a slot
threshold: explicit triggering makes benchmarks deterministic, which matters
more right now than smoothness.
"""
from __future__ import annotations

import threading
from queue import Empty, Queue

import numpy as np


class CoActivation:
    def __init__(self, engine, batch: int = 256, freeze_every: int = 8):
        self.engine = engine
        self.batch = batch
        self.freeze_every = freeze_every
        self._q: Queue = Queue()
        self._drain_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._since_freeze = 0

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
            self._thread = None

    def record(self, nodes, tick: int) -> None:
        """Queue one co-activation set. Returns immediately."""
        if len(nodes) > 1:
            self._q.put((np.asarray(nodes, dtype=np.uint32), tick))

    def drain(self, block: bool = False) -> int:
        """Apply events in order, retaining each event's own tick."""
        with self._drain_lock:
            total = 0
            pulled = 0
            latest_tick = 0
            while pulled < self.batch:
                try:
                    nodes, tick = (self._q.get(timeout=0.05)
                                   if block and not pulled else self._q.get_nowait())
                except Empty:
                    break
                nodes = np.unique(nodes)
                i, j = np.triu_indices(len(nodes), k=1)
                self.engine.reinforce(nodes[i], nodes[j], tick)
                total += len(i)
                pulled += 1
                latest_tick = max(latest_tick, tick)
            if pulled:
                self._since_freeze += 1
                if self._since_freeze >= self.freeze_every:
                    self.engine.freeze(latest_tick)
                    self._since_freeze = 0
            return total

    def flush(self, tick: int) -> None:
        """Drain pending events and publish after any active drain completes."""
        with self._drain_lock:
            while not self._q.empty():
                self.drain()
            self.engine.freeze(tick)
            self._since_freeze = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.drain(block=True) == 0:
                self._stop.wait(0.01)
