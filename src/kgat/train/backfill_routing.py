"""Phase-2 routing scaffold: per-chunk {skip | extract | escalate} episodes.

DESIGN-BACKFILL.md phase 2: an episode is one FILING; for every chunk the policy
either skips (emits ``NONE``), extracts triples itself, or escalates the chunk to
the big teacher pipeline. All three are expressed in ONE constrained decode — the
triple grammar with the ``ESCALATE`` sentinel (``build_triple_grammar(...,
sentinels=(ESCALATE_LABEL,))``) — so the router and the extractor are the same
0.6B model and GRPO optimizes the trie-masked logprobs it already knows how to
score.

The reward follows the write-path objective verbatim: **judged precision +
distant recall − λ·tokens**. Edge quality comes from ``kgat.train.edge_judge``
(objective existence gates + the critic DISTILLED into a small cross-encoder
from alphina's ~490k logged verdicts) injected as a plain callable, so this
module stays pure Python: unit tests use fakes, and no LLM runs inside the
training loop — the live critic (relationship_critic.py) is reserved for
eval-time audits via ``judge_from_critic``. Distant recall credits an escalated
chunk with all of its teacher edges (the big pipeline is the label source),
which is what lets the policy learn WHEN escalation is worth the tokens. This is
the arm that can exceed the teacher: SFT imitates its labels; the judge rewards
precision the teacher itself lacked.

Like ``kgat.train.reward``, everything here is a pure function of the episode —
a bug in the reward silently corrupts every downstream experiment, so it must be
exhaustively unit-testable without torch or network.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kgat.controller.constrained_decoding import TripleDecodeResult
from kgat.data.backfill_export import ExtractionPair
from kgat.train.edge_judge import EdgeJudgeFn

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
    """kgat-side mirror of alphina's ``EdgeVerdict`` (the logged judgment schema)."""

    verdict: str  # "accept" | "reject" | "relabel"
    corrected_type: str | None = None
    faithfulness: float = 0.0


# (filer, relation, target, chunk_text) -> verdict. Wraps an LLM-as-judge; kept
# for eval-time audits — the TRAINING loop uses EdgeJudgeFn (kgat.train.edge_judge).
CriticFn = Callable[[str, str, str, str], CriticVerdict]


def judge_from_critic(critic: CriticFn) -> EdgeJudgeFn:
    """Adapt an LLM critic to the edge-judge contract (accept -> faithfulness, else 0)."""

    def judge(pair: ExtractionPair, relation: str, target: str) -> float:
        ruling = critic(pair.filer, relation, target, pair.text)
        if ruling.verdict == "accept":
            return max(0.0, min(1.0, ruling.faithfulness))
        return 0.0

    return judge


@dataclass(frozen=True)
class RoutingReward:
    """The episode reward and its audited components (for logging/ablation)."""

    reward: float
    precision: float  # mean judge score over emitted edges (distant when judge=None)
    recall: float  # distant: recovered teacher edges / all teacher edges
    cost_tokens: float  # small-model tokens + priced escalations
    normalized_cost: float
    n_emitted: int
    n_accepted: int  # emitted edges with judge score >= 0.5 (diagnostic)
    n_escalated: int


def routing_reward(
    pairs: Sequence[ExtractionPair],
    decisions: Sequence[ChunkDecision],
    *,
    judge: EdgeJudgeFn | None = None,
    lam: float = 0.1,
    escalation_cost_tokens: float = 2000.0,
    cost_cap: float = 20000.0,
    precision_weight: float = 0.5,
    recall_weight: float = 0.5,
) -> RoutingReward:
    """Score one filing episode: judged precision + distant recall − λ·tokens.

    * **precision** — the MEAN judge score over the edges the policy EMITTED
      (extract routes). The judge (``kgat.train.edge_judge``) layers objective
      existence gates (grounding, target-is-company, filer-is-party) under a
      type-faithfulness score in [0, 1] — the distilled critic when available,
      the distant-label anchor otherwise. ``judge=None`` falls back to pure
      distant precision (emitted ∩ teacher). No emitted edges -> 1.0 (nothing
      claimed, nothing wrong — recall carries the pressure to claim).
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
    policies dominated, same guard logic as ``kgat.train.reward``. A judge that
    can certify novel edges (the distilled critic) is what lets total reward
    exceed the pure-imitation ceiling: precision then pays for true edges the
    teacher never had, while recall keeps the teacher's edges from being
    abandoned.
    """
    if len(pairs) != len(decisions):
        raise ValueError(f"{len(pairs)} pairs vs {len(decisions)} decisions")
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if cost_cap <= 0:
        raise ValueError(f"cost_cap must be positive, got {cost_cap}")

    n_emitted = n_accepted = n_escalated = 0
    gold_total = gold_recovered = 0
    score_sum = 0.0
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
            if judge is None:
                score = 1.0 if (relation, target) in gold else 0.0
            else:
                score = max(0.0, min(1.0, float(judge(pair, relation, target))))
            score_sum += score
            n_accepted += int(score >= 0.5)

    precision = score_sum / n_emitted if n_emitted else 1.0
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
    "judge_from_critic",
    "RoutingReward",
    "routing_reward",
]
