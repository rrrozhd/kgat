"""Cost-penalized RL reward for the traversal controller.

The reward is the project's steering signal: it must make the controller answer
correctly *and* stop early when deeper search would not help — without letting the
"stop early = cheap = high reward" shortcut win. It is a pure function so it can be
unit-tested exhaustively; a bug here silently corrupts every downstream experiment.

    reward = correctness_term - lam * normalized_cost

* ``correctness_term`` in [0, 1] — ``f1`` (default) or ``hit``.
* ``normalized_cost`` in [0, 1] — a chosen cost axis divided by a configured cap and
  clamped.
* ``lam`` >= 0 — the cost penalty weight; the swept knob that traces the frontier.

Reward-hacking guard (the λ range that keeps argmax aligned with answering):

    lam < (minimum correctness gain of answering) / (maximum normalized-cost increase)

The worst case is a *premature STOP*: it is cheap but scores low correctness. For a
correct-but-expensive answer to still beat it, the correctness gain must dominate the
extra cost penalty. With costs normalized to [0, 1] the max cost increase is 1, so:

* binary ``hit`` (gain wrong→right = 1)              => any ``lam < 1`` is safe.
* ``f1`` with partial credit (e.g. 0.5 → 1.0, gain 0.5) => needs ``lam < 0.5``.

``DEFAULT_LAMBDA = 0.1`` sits well inside both regimes. λ=0 ignores cost entirely
(pure accuracy objective).
"""

from __future__ import annotations

from collections.abc import Sequence

from kgat.data.schemas import Entity
from kgat.eval.cost import CostRecord
from kgat.eval.metrics import f1 as _f1
from kgat.eval.metrics import hit as _hit

DEFAULT_LAMBDA: float = 0.1
DEFAULT_COST_AXIS: str = "llm_calls"


def _cost_scalar(cost: CostRecord | float | int, axis: str) -> float:
    """Reduce a cost input to a single scalar along ``axis``.

    Accepts a ``CostRecord`` (extracts the named axis) or a raw number.
    """
    if isinstance(cost, CostRecord):
        return cost.scalar(axis)
    if isinstance(cost, (int, float)):
        return float(cost)
    raise TypeError(f"cost must be a CostRecord or a number, got {type(cost).__name__}")


def normalized_cost(cost: CostRecord | float | int, *, cost_cap: float, axis: str) -> float:
    """Normalize a cost to [0, 1] by dividing by ``cost_cap`` and clamping."""
    if cost_cap <= 0:
        raise ValueError(f"cost_cap must be positive, got {cost_cap}")
    scalar = _cost_scalar(cost, axis)
    return max(0.0, min(scalar / cost_cap, 1.0))


def correctness_term(
    predictions: Sequence[Entity],
    gold: Sequence[Entity],
    correctness: str,
) -> float:
    """The correctness component in [0, 1]."""
    if correctness == "f1":
        return _f1(predictions, gold)
    if correctness == "hit":
        return 1.0 if _hit(predictions, gold) else 0.0
    raise ValueError(f"correctness must be 'f1' or 'hit', got {correctness!r}")


def compute_reward(
    predictions: Sequence[Entity],
    gold: Sequence[Entity],
    cost: CostRecord | float | int,
    *,
    lam: float = DEFAULT_LAMBDA,
    correctness: str = "f1",
    cost_cap: float = 20.0,
    cost_axis: str = DEFAULT_COST_AXIS,
    cost_mode: str = "absolute",
    oracle_cost: float | None = None,
) -> float:
    """Cost-penalized reward for a completed traversal.

    Two cost modes (DESIGN-GRAPH-RL.md §A):

    * ``"absolute"`` — penalize the raw cost scalar. Calibration warning: size
      ``cost_cap`` to the dataset's achievable cost RANGE, or λ gets no leverage
      (the first FinKG sweep's null result).
    * ``"regret"`` — penalize only the EXCESS over the per-question oracle
      minimum (``max(0, cost - oracle_cost)``). A 3-hop question is not punished
      for needing 3 hops; overshoot is punished everywhere equally. ``cost_cap``
      then caps the *excess*. Requires ``oracle_cost`` (BFS min depth at train
      time, same axis as ``cost_axis``).

    Args:
        predictions: predicted answer entities.
        gold: gold answer entities.
        cost: the traversal's ``CostRecord`` (or a precomputed scalar cost).
        lam: cost-penalty weight (>= 0). See module docstring for the safe range.
        correctness: "f1" (partial credit) or "hit" (binary).
        cost_cap: cost (or excess) value that maps to a normalized penalty of 1.0.
        cost_axis: which ``CostRecord`` axis to penalize (ignored for scalar cost).
        cost_mode: "absolute" | "regret".
        oracle_cost: per-question oracle minimum on ``cost_axis`` (regret mode).

    Returns:
        ``correctness_term - lam * normalized_penalty``.
    """
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if cost_mode not in ("absolute", "regret"):
        raise ValueError(f"cost_mode must be 'absolute' or 'regret', got {cost_mode!r}")
    corr = correctness_term(predictions, gold, correctness)
    if cost_mode == "regret":
        if oracle_cost is None:
            raise ValueError("cost_mode='regret' requires oracle_cost")
        excess = max(0.0, _cost_scalar(cost, cost_axis) - float(oracle_cost))
        penalty = lam * normalized_cost(excess, cost_cap=cost_cap, axis=cost_axis)
    else:
        penalty = lam * normalized_cost(cost, cost_cap=cost_cap, axis=cost_axis)
    return corr - penalty


__all__ = [
    "DEFAULT_LAMBDA",
    "DEFAULT_COST_AXIS",
    "compute_reward",
    "correctness_term",
    "normalized_cost",
]
