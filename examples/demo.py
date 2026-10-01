"""Run with: .venv/bin/python examples/demo.py"""
import numpy as np
from assoc_mem import Memory

with Memory(capacity=1024) as memory:
    for text in ["cat food dinner", "cat sleep bed", "dog walk park", "cat play toy"]:
        memory.remember(text)
    print("Recall:", [n.content for n in memory.recall("cat", k_seed=4)])
    memory.flush()
    print("After co-activation:", memory.stats())
    ids, scores = memory.engine.activate(np.array([1], np.uint32),
        np.array([1], np.float32), hops=2, tick=memory.tick)
    print("Activation:", list(zip(ids.tolist(), scores.tolist())))
    print("Neighbors:", memory.engine.neighbors(1, tick=memory.tick))
    direction, ordered, mass = memory.axis([1, 2, 3, 4])
    print("Axis order:", ordered.tolist(), "edge mass:", mass)
    print("Directional walk:", memory.engine.walk_directional(1, direction, tick=memory.tick).tolist())
    print("Unexplored pairs:", memory.unexplored("cat", k=2).tolist())
    print("Sector counts:", memory.sectors(1)[0].tolist())
    memory.advance()
    print("Next tick:", memory.tick)
