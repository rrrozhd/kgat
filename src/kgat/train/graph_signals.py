"""Exact graph signals for training (train-time only — uses gold labels).

Because the environment is a fully-observed per-question subgraph, quantities that
generic RL must estimate (distance-to-goal, reachability, minimum required depth)
are computable exactly by BFS. This module is the shared foundation for:

* **Regret-style cost** (DESIGN-GRAPH-RL.md §A): the per-question oracle minimum
  depth is the zero point of the cost penalty — waste is penalized, need is not.
* **Potential-based shaping** (§B): distance-to-gold as an exact potential gives
  every hop immediate credit; per Ng et al. (1999) shaping preserves the optimal
  policy, and the group-relative baseline absorbs the constant Φ(s0) term.
* **Dead-end detection** (§D): a frontier whose min-distance hits the cap can
  never reach a gold.

Pure Python, no model deps, fully unit-tested. Uses gold answers, so these
signals are legitimate ONLY at training time — same status as the reward itself.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

from kgat.data.schemas import Entity, Triple


def gold_distance_map(
    triples: Iterable[Triple],
    golds: Sequence[Entity],
    *,
    max_dist: int,
) -> dict[Entity, int]:
    """Exact hop distance from every node to its NEAREST gold, following forward
    edges (reverse BFS from the gold set). Nodes farther than ``max_dist`` (or
    unreachable) are omitted — treat missing as ``max_dist``.

    O(edges); run once per question at rollout time.
    """
    reverse_adj: dict[Entity, list[Entity]] = {}
    for t in triples:
        reverse_adj.setdefault(t.tail, []).append(t.head)

    dist: dict[Entity, int] = {}
    queue: deque[Entity] = deque()
    for gold in golds:
        if gold not in dist:
            dist[gold] = 0
            queue.append(gold)
    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= max_dist:
            continue
        for prev in reverse_adj.get(node, ()):
            if prev not in dist:
                dist[prev] = d + 1
                queue.append(prev)
    return dist


def frontier_min_dist(
    dist_map: dict[Entity, int],
    nodes: Iterable[Entity],
    *,
    cap: int,
) -> int:
    """Minimum distance-to-gold over a frontier; ``cap`` for empty/unreachable."""
    best = cap
    for node in nodes:
        d = dist_map.get(node, cap)
        if d < best:
            best = d
            if best == 0:
                break
    return best


def shaping_deltas(
    step_dists: Sequence[int],
    final_dist: int,
    *,
    scale: float,
) -> list[float]:
    """Per-step shaping advantages: ``scale * (d_t - d_T)`` for each decision t.

    Telescoped potential-to-go with Φ(state) = -distance: a step taken while the
    trajectory still went on to make net progress (d_t > d_T) gets positive
    credit; steps at/after the point where progress stalled or died get zero or
    negative credit. The group-relative terminal advantage is added separately by
    the caller — these deltas only *redistribute* credit within a trajectory.
    """
    return [scale * (d_t - final_dist) for d_t in step_dists]


__all__ = ["gold_distance_map", "frontier_min_dist", "shaping_deltas"]
