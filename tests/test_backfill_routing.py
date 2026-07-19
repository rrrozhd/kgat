"""Tests for the phase-2 routing scaffold (pure, no torch, fake critic)."""

from __future__ import annotations

import pytest

from kgat.controller.constrained_decoding import TripleDecodeResult
from kgat.data.backfill_export import ExtractionPair
from kgat.train.backfill_routing import (
    ESCALATE_LABEL,
    QUALITY_WEIGHTED_PR,
    ROUTE_ESCALATE,
    ROUTE_EXTRACT,
    ROUTE_SKIP,
    ChunkDecision,
    CriticVerdict,
    decision_from_result,
    judge_from_critic,
    rollout_filing,
    routing_reward,
)


def pair(triples, text="chunk text") -> ExtractionPair:
    return ExtractionPair(
        text=text, filer="Filer Co", triples=tuple(triples), filing="acc-1", company="co-1"
    )


def decision(route, triples=(), gen_tokens=10) -> ChunkDecision:
    return ChunkDecision(
        route=route, triples=tuple(triples), gen_tokens=gen_tokens, logprob=-1.0, n_choices=3
    )


def result(triples=(), sentinel=None) -> TripleDecodeResult:
    return TripleDecodeResult(
        ids=(1, 2, 3), triples=tuple(triples), logprob=-1.0, n_choices=2, sentinel=sentinel
    )


def test_decision_from_result_maps_all_routes():
    assert decision_from_result(result()).route == ROUTE_SKIP
    assert decision_from_result(result(sentinel=ESCALATE_LABEL)).route == ROUTE_ESCALATE
    extract = decision_from_result(result(triples=[("supplier", "A")]))
    assert extract.route == ROUTE_EXTRACT
    assert extract.triples == (("supplier", "A"),)
    assert extract.gen_tokens == 3
    with pytest.raises(ValueError):
        decision_from_result(result(sentinel="DEFER"))


def test_rollout_filing_is_model_free():
    pairs = [pair([("supplier", "A")]), pair([])]
    outs = iter([result(triples=[("supplier", "A")]), result()])
    decisions = rollout_filing(pairs, lambda p: next(outs))
    assert [d.route for d in decisions] == [ROUTE_EXTRACT, ROUTE_SKIP]


def test_distant_reward_without_critic():
    pairs = [
        pair([("supplier", "A"), ("partner", "B")]),  # extract: gets one of two
        pair([("customer", "C")]),  # escalated: teacher recovers it
        pair([]),  # correctly skipped
    ]
    decisions = [
        decision(ROUTE_EXTRACT, [("supplier", "A"), ("partner", "WRONG")]),
        decision(ROUTE_ESCALATE),
        decision(ROUTE_SKIP),
    ]
    r = routing_reward(
        pairs, decisions, lam=0.0, escalation_cost_tokens=1000, cost_cap_per_chunk=4000
    )
    assert r.precision == 1 / 2  # one of two emitted edges matches the teacher
    assert r.recall == 2 / 3  # A extracted + C escalated; B missed
    assert r.n_escalated == 1
    assert r.cost_tokens == 10 * 3 + 1000
    # Default quality is F1 (the weighted sum is hackable — see
    # test_routing_reward_hack.py); the components above are unchanged.
    assert r.reward == pytest.approx(
        2 * r.precision * r.recall / (r.precision + r.recall)
    )

    old = routing_reward(
        pairs, decisions, lam=0.0, escalation_cost_tokens=1000, cost_cap_per_chunk=4000,
        quality_mode=QUALITY_WEIGHTED_PR,
    )
    assert old.reward == 0.5 * r.precision + 0.5 * r.recall  # prior runs reproducible


def test_judged_precision_overrides_distant():
    # The judge can score up an edge the teacher never had (exceed-the-teacher
    # path) and zero one the teacher did have.
    pairs = [pair([("supplier", "A")])]
    decisions = [decision(ROUTE_EXTRACT, [("supplier", "A"), ("competitor", "D")])]

    def judge(p, relation, target) -> float:
        assert p.filer == "Filer Co"
        return 0.9 if target == "D" else 0.1

    r = routing_reward(pairs, decisions, judge=judge, lam=0.0)
    assert r.precision == pytest.approx((0.9 + 0.1) / 2)  # mean score, judge rules
    # Recall stays DISTANT: emitted A matches gold (recovered); extra D has no effect.
    assert r.recall == 1.0
    assert r.n_accepted == 1  # only D clears the 0.5 diagnostic bar


def test_judge_from_critic_adapter():
    def critic(filer, relation, target, text) -> CriticVerdict:
        if target == "A":
            return CriticVerdict(verdict="accept", faithfulness=0.8)
        return CriticVerdict(verdict="reject", faithfulness=0.7)  # reject scores 0

    judge = judge_from_critic(critic)
    p = pair([])
    assert judge(p, "supplier", "A") == 0.8
    assert judge(p, "supplier", "B") == 0.0


def test_degenerate_policies_are_dominated():
    pairs = [pair([("supplier", "A")]), pair([])]
    gold_extract = [decision(ROUTE_EXTRACT, [("supplier", "A")]), decision(ROUTE_SKIP)]
    skip_all = [decision(ROUTE_SKIP), decision(ROUTE_SKIP)]
    escalate_all = [decision(ROUTE_ESCALATE), decision(ROUTE_ESCALATE)]
    kw = dict(lam=0.4, escalation_cost_tokens=5000, cost_cap_per_chunk=5000)
    r_good = routing_reward(pairs, gold_extract, **kw)
    r_skip = routing_reward(pairs, skip_all, **kw)
    r_esc = routing_reward(pairs, escalate_all, **kw)
    assert r_good.reward > r_skip.reward  # skipping loses recall
    assert r_good.reward > r_esc.reward  # escalating everything pays λ
    assert r_esc.recall == 1.0 and r_esc.normalized_cost == 1.0
    assert r_skip.recall == 0.0 and r_skip.precision == 1.0  # claims nothing


def test_per_chunk_rewards_decompose_credit():
    from kgat.train.backfill_routing import per_chunk_rewards

    pairs = [
        pair([("supplier", "A")]),  # extracted correctly
        pair([("partner", "B")]),  # skipped despite gold -> punished HERE, not globally
        pair([]),  # correct skip
        pair([("customer", "C")]),  # escalated
    ]
    decisions = [
        decision(ROUTE_EXTRACT, [("supplier", "A")], gen_tokens=20),
        decision(ROUTE_SKIP, gen_tokens=5),
        decision(ROUTE_SKIP, gen_tokens=5),
        decision(ROUTE_ESCALATE, gen_tokens=5),
    ]
    per_chunk, mean = per_chunk_rewards(
        pairs, decisions, lam=0.4, escalation_cost_tokens=2000, cost_cap_per_chunk=2000
    )
    assert len(per_chunk) == 4
    # The bad skip is the WORST chunk; the correct skip the best; escalation
    # pays lam for its full recall.
    assert per_chunk[1] == min(per_chunk)
    assert per_chunk[2] == max(per_chunk)
    assert per_chunk[0] > per_chunk[3]  # correct extraction beats paying for escalation
    assert mean == pytest.approx(sum(per_chunk) / 4)
    # Cost normalization is per chunk: the escalated chunk alone hits the cap
    # (5 gen + 2000 escalation > 2000 cap, clamped) -> P=1, R=1, minus lam.
    assert per_chunk[3] == pytest.approx(1.0 - 0.4)

    with pytest.raises(ValueError):
        per_chunk_rewards(pairs, decisions[:2], lam=0.1)


def test_empty_filing_and_validation():
    r = routing_reward([pair([])], [decision(ROUTE_SKIP)], lam=0.1)
    assert r.precision == 1.0 and r.recall == 1.0
    with pytest.raises(ValueError):
        routing_reward([pair([])], [], lam=0.1)
    with pytest.raises(ValueError):
        routing_reward([pair([])], [decision(ROUTE_SKIP)], lam=-1)
    with pytest.raises(ValueError):
        routing_reward([pair([])], [decision("defer")], lam=0.1)
