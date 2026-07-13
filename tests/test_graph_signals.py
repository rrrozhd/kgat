"""Tests for exact graph training signals (kgat.train.graph_signals) and the
regret reward mode + shaping wiring that build on them."""

from __future__ import annotations

import math

import pytest

from kgat.data.schemas import Action, ActionType, Trajectory, TrajectoryStep, Triple
from kgat.eval.cost import CostRecord
from kgat.train.graph_signals import frontier_min_dist, gold_distance_map, shaping_deltas
from kgat.train.grpo import trajectory_to_rollout
from kgat.train.reward import compute_reward

CHAIN = (
    Triple("a", "r1", "b"),
    Triple("b", "r2", "c"),
    Triple("a", "shortcut", "c"),
    Triple("x", "r3", "y"),  # disconnected from gold
)


def test_gold_distance_map_exact():
    dmap = gold_distance_map(CHAIN, ["c"], max_dist=5)
    assert dmap["c"] == 0
    assert dmap["a"] == 1  # via the shortcut, not the 2-hop chain
    assert dmap["b"] == 1
    assert "x" not in dmap and "y" not in dmap  # unreachable -> omitted


def test_gold_distance_map_multi_gold_takes_nearest():
    dmap = gold_distance_map(CHAIN, ["c", "b"], max_dist=5)
    assert dmap["a"] == 1 and dmap["b"] == 0 and dmap["c"] == 0


def test_frontier_min_dist_and_cap():
    dmap = gold_distance_map(CHAIN, ["c"], max_dist=5)
    assert frontier_min_dist(dmap, ["a", "x"], cap=9) == 1
    assert frontier_min_dist(dmap, ["x", "y"], cap=9) == 9  # unreachable -> cap
    assert frontier_min_dist(dmap, [], cap=9) == 9


def test_shaping_deltas_redistribute_credit():
    # dists per decision: 2 -> 1 -> 0 (reached gold); final dist 0.
    deltas = shaping_deltas([2, 1], 0, scale=0.1)
    assert deltas == pytest.approx([0.2, 0.1])  # earlier steps carry progress-to-go
    # Dead trajectory: step 1 at dist 1, then killed (cap 5); the pre-kill step
    # gets negative credit, post-kill steps get zero.
    deltas = shaping_deltas([1, 5], 5, scale=0.1)
    assert deltas == pytest.approx([-0.4, 0.0])


def test_regret_reward_ignores_needed_depth():
    gold = ("g",)
    deep_needed = compute_reward(
        ("g",),
        gold,
        CostRecord(hops=3),
        lam=0.4,
        cost_mode="regret",
        oracle_cost=3,
        cost_axis="hops",
        cost_cap=4.0,
    )
    assert deep_needed == 1.0  # 3 hops on a 3-hop question: zero penalty
    overshoot = compute_reward(
        ("g",),
        gold,
        CostRecord(hops=3),
        lam=0.4,
        cost_mode="regret",
        oracle_cost=1,
        cost_axis="hops",
        cost_cap=4.0,
    )
    assert math.isclose(overshoot, 1.0 - 0.4 * (2 / 4))  # 2 excess hops
    assert deep_needed > overshoot


def test_regret_requires_oracle_and_valid_mode():
    with pytest.raises(ValueError):
        compute_reward(("g",), ("g",), CostRecord(hops=1), cost_mode="regret")
    with pytest.raises(ValueError):
        compute_reward(("g",), ("g",), CostRecord(hops=1), cost_mode="bogus")


def _traj(frontiers: list[tuple[str, ...]], final: tuple[str, ...]) -> Trajectory:
    steps = [
        TrajectoryStep(
            state_repr=f"s{i}",
            candidates=("r1", "r2"),
            action=Action(ActionType.EXPAND, "r1", score=0.5),
            frontier_nodes=f,
        )
        for i, f in enumerate(frontiers)
    ]
    return Trajectory(
        qid="q",
        steps=steps,
        predicted_answers=("z",),
        hit=False,
        cost=CostRecord(hops=len(frontiers)),
        final_frontier=final,
    )


def test_rollout_carries_per_step_shaping():
    dmap = gold_distance_map(CHAIN, ["c"], max_dist=5)
    traj = _traj([("a",), ("b",)], final=("c",))  # dists 1, 1 -> final 0
    rollout = trajectory_to_rollout(traj, 1.0, dist_map=dmap, shaping_scale=0.1, dist_cap=5)
    shapes = [s[4] for s in rollout.steps]
    assert shapes == pytest.approx([0.1, 0.1])
    # Shaping off -> zeros, same structure.
    plain = trajectory_to_rollout(traj, 1.0)
    assert [s[4] for s in plain.steps] == [0.0, 0.0]
