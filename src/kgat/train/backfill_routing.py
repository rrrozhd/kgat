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
    cost_cap_per_chunk: float = 2000.0,
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
      the same unit), normalized by ``cost_cap_per_chunk × n_chunks`` and
      clamped to [0, 1]. Per-CHUNK normalization is the FinKG sweep-v1 lesson:
      a fixed episode-level cap gives λ different leverage on small vs large
      filings. With the default cap equal to the escalation price, normalized
      cost ≈ the fraction of chunks escalated — λ reads directly against the
      frontier's escalation axis.

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
    if cost_cap_per_chunk <= 0:
        raise ValueError(f"cost_cap_per_chunk must be positive, got {cost_cap_per_chunk}")

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
        uncertain = set(pair.uncertain)
        for relation, target in decision.triples:
            if (relation, target) in uncertain:
                continue  # sub-floor teacher edge: neither rewarded nor punished
            n_emitted += 1
            if judge is None:
                score = 1.0 if (relation, target) in gold else 0.0
            else:
                score = max(0.0, min(1.0, float(judge(pair, relation, target))))
            score_sum += score
            n_accepted += int(score >= 0.5)

    precision = score_sum / n_emitted if n_emitted else 1.0
    recall = gold_recovered / gold_total if gold_total else 1.0
    cap = cost_cap_per_chunk * max(1, len(pairs))
    normalized = max(0.0, min(cost / cap, 1.0))
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


def per_chunk_rewards(
    pairs: Sequence[ExtractionPair],
    decisions: Sequence[ChunkDecision],
    *,
    judge: EdgeJudgeFn | None = None,
    lam: float = 0.1,
    escalation_cost_tokens: float = 2000.0,
    cost_cap_per_chunk: float = 2000.0,
    precision_weight: float = 0.5,
    recall_weight: float = 0.5,
) -> tuple[list[float], float]:
    """Per-chunk reward decomposition — exact credit assignment for routing GRPO.

    Unlike traversal (one entangled terminal reward), the routing reward
    decomposes: each chunk gets its own
    ``w_p·precision_i + w_r·recall_i − λ·cost_i`` where precision_i is the mean
    judge score over the chunk's emitted edges (1.0 when nothing is claimed —
    recall carries the pressure to claim), recall_i is the chunk's teacher edges
    recovered (escalate recovers all, skip none; 1.0 when the chunk has none),
    and cost_i is the chunk's tokens (+ escalation price) over
    ``cost_cap_per_chunk``. Returns ``(per_chunk, mean)``.

    Note the aggregation difference vs :func:`routing_reward`: this is the
    MACRO (chunk-equal) objective — the right shape for per-decision credit —
    while ``routing_reward`` is micro (edge-weighted) and stays the evaluation
    aggregate. GRPO groups roll out the SAME chunk list, so group-relative
    advantages computed per chunk position are matched-pair comparisons: the
    write-path analog of exact potential shaping (DESIGN-GRAPH-RL §B), with no
    approximation at all.
    """
    if len(pairs) != len(decisions):
        raise ValueError(f"{len(pairs)} pairs vs {len(decisions)} decisions")
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if cost_cap_per_chunk <= 0:
        raise ValueError(f"cost_cap_per_chunk must be positive, got {cost_cap_per_chunk}")

    rewards: list[float] = []
    for pair, decision in zip(pairs, decisions, strict=True):
        gold = set(pair.triples)
        cost = decision.gen_tokens

        if decision.route == ROUTE_ESCALATE:
            precision_i, recall_i = 1.0, 1.0
            cost += escalation_cost_tokens
        elif decision.route == ROUTE_SKIP:
            precision_i, recall_i = 1.0, (0.0 if gold else 1.0)
        elif decision.route == ROUTE_EXTRACT:
            uncertain = set(pair.uncertain)
            scores = []
            for relation, target in decision.triples:
                if (relation, target) in uncertain:
                    continue  # sub-floor teacher edge: masked from precision
                if judge is None:
                    scores.append(1.0 if (relation, target) in gold else 0.0)
                else:
                    scores.append(max(0.0, min(1.0, float(judge(pair, relation, target)))))
            precision_i = sum(scores) / len(scores) if scores else 1.0
            recall_i = (
                len(set(decision.triples) & gold) / len(gold) if gold else 1.0
            )
        else:
            raise ValueError(f"unknown route {decision.route!r}")

        cost_i = max(0.0, min(cost / cost_cap_per_chunk, 1.0))
        rewards.append(
            precision_weight * precision_i + recall_weight * recall_i - lam * cost_i
        )
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    return rewards, mean


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
    "per_chunk_rewards",
]
