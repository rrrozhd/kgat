"""Phase-2 routing scaffold: per-chunk {skip | extract | escalate} episodes.

DESIGN-BACKFILL.md phase 2: an episode is one FILING; for every chunk the policy
either skips (emits ``NONE``), extracts triples itself, or escalates the chunk to
the big teacher pipeline. All three are expressed in ONE constrained decode — the
triple grammar with the ``ESCALATE`` sentinel (``build_triple_grammar(...,
sentinels=(ESCALATE_LABEL,))``) — so the router and the extractor are the same
0.6B model and GRPO optimizes the trie-masked logprobs it already knows how to
score.

The reward follows the write-path objective verbatim: **critic-audited precision
+ distant recall − λ·tokens**. alphina's per-edge critic (relationship_critic.py)
IS the reward model — it is injected here as a plain callable so this module
stays pure Python: unit tests use fakes, the training run wraps the real
LLM-as-judge API. Distant recall credits an escalated chunk with all of its
teacher edges (the big pipeline is the label source), which is what lets the
policy learn WHEN escalation is worth the tokens. This is the arm that can
exceed the teacher: SFT imitates its labels; the critic rewards precision the
teacher itself lacked.

Like ``kgat.train.reward``, everything here is a pure function of the episode —
a bug in the reward silently corrupts every downstream experiment, so it must be
exhaustively unit-testable without torch or network.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kgat.controller.constrained_decoding import TripleDecodeResult
from kgat.data.backfill_export import ExtractionPair

ESCALATE_LABEL = "ESCALATE"

ROUTE_SKIP = "skip"
ROUTE_EXTRACT = "extract"
ROUTE_ESCALATE = "escalate"


@dataclass(frozen=True)
class ChunkDecision:
    """One routing decision, derived from a constrained decode of one chunk."""

    route: str  # ROUTE_SKIP | ROUTE_EXTRACT | ROUTE_ESCALATE
    triples: tuple[tuple[str, str], ...]  # emitted edges (extract route only)
    gen_tokens: int  # tokens the small model generated for this chunk
    logprob: float  # temp-1 masked logprob of the decode (GRPO gradient pass)
    n_choices: int  # real decision steps in the decode


def decision_from_result(result: TripleDecodeResult) -> ChunkDecision:
    """Map a grammar decode onto the routing action space.

    ``ESCALATE`` sentinel -> escalate; empty triples (``NONE``) -> skip;
    otherwise the emitted triples ARE the extract action.
    """
    if result.sentinel == ESCALATE_LABEL:
        route = ROUTE_ESCALATE
    elif result.sentinel is not None:
        raise ValueError(f"unknown routing sentinel {result.sentinel!r}")
    else:
        route = ROUTE_EXTRACT if result.triples else ROUTE_SKIP
    return ChunkDecision(
        route=route,
        triples=result.triples,
        gen_tokens=len(result.ids),
        logprob=result.logprob,
        n_choices=result.n_choices,
    )


def rollout_filing(
    pairs: Sequence[ExtractionPair],
    decide_fn: Callable[[ExtractionPair], TripleDecodeResult],
) -> list[ChunkDecision]:
    """Route every chunk of one filing with the policy's constrained decode.

    ``decide_fn`` wraps the model (prompt build + grammar decode, sampled at the
    rollout temperature); this function stays model-free.
    """
    return [decision_from_result(decide_fn(pair)) for pair in pairs]


@dataclass(frozen=True)
class CriticVerdict:
    """kgat-side mirror of alphina's ``EdgeVerdict`` (the reward model's ruling)."""

    verdict: str  # "accept" | "reject" | "relabel"
    corrected_type: str | None = None
    faithfulness: float = 0.0


# (filer, relation, target, chunk_text) -> verdict. The training run wraps
# alphina's critic prompt behind this; tests inject fakes.
CriticFn = Callable[[str, str, str, str], CriticVerdict]


@dataclass(frozen=True)
class RoutingReward:
    """The episode reward and its audited components (for logging/ablation)."""

    reward: float
    precision: float  # critic-audited (or distant, when critic is None)
    recall: float  # distant: recovered teacher edges / all teacher edges
    cost_tokens: float  # small-model tokens + priced escalations
    normalized_cost: float
    n_emitted: int
    n_accepted: int
    n_escalated: int


def routing_reward(
    pairs: Sequence[ExtractionPair],
    decisions: Sequence[ChunkDecision],
    *,
    critic: CriticFn | None = None,
    lam: float = 0.1,
    escalation_cost_tokens: float = 2000.0,
    cost_cap: float = 20000.0,
    precision_weight: float = 0.5,
    recall_weight: float = 0.5,
) -> RoutingReward:
    """Score one filing episode: critic-audited precision + distant recall − λ·tokens.

    * **precision** — over the edges the policy EMITTED (extract routes): the
      fraction the critic accepts (relabel/reject are the emitter's mistakes).
      With ``critic=None`` (offline runs) it falls back to distant precision
      against the teacher labels. No emitted edges -> 1.0 (nothing claimed).
    * **recall** — distant, over ALL teacher edges of the filing: an extract
      chunk recovers ``emitted ∩ gold``; an escalated chunk recovers all its
      gold edges (the teacher pipeline is the label source); a skip recovers
      nothing. No gold edges -> 1.0.
    * **cost** — small-model generated tokens across every chunk plus
      ``escalation_cost_tokens`` per escalation (the big-model call, priced in
      the same unit), normalized by ``cost_cap`` and clamped to [0, 1].

    ``escalate-everything`` scores perfect recall at maximum cost;
    ``skip-everything`` is free but scores recall 0 on any filing with edges —
    λ within (0, min(precision_weight, recall_weight)) keeps both degenerate
    policies dominated, same guard logic as ``kgat.train.reward``.
    """
    if len(pairs) != len(decisions):
        raise ValueError(f"{len(pairs)} pairs vs {len(decisions)} decisions")
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if cost_cap <= 0:
        raise ValueError(f"cost_cap must be positive, got {cost_cap}")

    n_emitted = n_accepted = n_escalated = 0
    gold_total = gold_recovered = 0
    cost = 0.0

    for pair, decision in zip(pairs, decisions, strict=True):
        gold = set(pair.triples)
        gold_total += len(gold)
        cost += decision.gen_tokens

        if decision.route == ROUTE_ESCALATE:
            n_escalated += 1
            cost += escalation_cost_tokens
            gold_recovered += len(gold)
            continue
        if decision.route == ROUTE_SKIP:
            continue
        if decision.route != ROUTE_EXTRACT:
            raise ValueError(f"unknown route {decision.route!r}")

        gold_recovered += len(set(decision.triples) & gold)
        for relation, target in decision.triples:
            n_emitted += 1
            if critic is None:
                n_accepted += int((relation, target) in gold)
            else:
                ruling = critic(pair.filer, relation, target, pair.text)
                n_accepted += int(ruling.verdict == "accept")

    precision = n_accepted / n_emitted if n_emitted else 1.0
    recall = gold_recovered / gold_total if gold_total else 1.0
    normalized = max(0.0, min(cost / cost_cap, 1.0))
    reward = precision_weight * precision + recall_weight * recall - lam * normalized
    return RoutingReward(
        reward=reward,
        precision=precision,
        recall=recall,
        cost_tokens=cost,
        normalized_cost=normalized,
        n_emitted=n_emitted,
        n_accepted=n_accepted,
        n_escalated=n_escalated,
    )


__all__ = [
    "ESCALATE_LABEL",
    "ROUTE_SKIP",
    "ROUTE_EXTRACT",
    "ROUTE_ESCALATE",
    "ChunkDecision",
    "decision_from_result",
    "rollout_filing",
    "CriticVerdict",
    "CriticFn",
    "RoutingReward",
    "routing_reward",
]
