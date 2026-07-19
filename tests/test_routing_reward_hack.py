"""Regression tests for the vacuous-precision reward hack.

Measured 2026-07-19 (``results/grpo-routing-2026-07-19``): 252 GRPO updates drove
extraction from 20.8% of chunks to 3.2% while reward, precision and recall all ROSE
to ~0.98. Skipping and escalating both scored precision 1.0 for claiming nothing, so
silence on a gold-bearing chunk banked a free ``w_p``. These tests pin the fix:
under ``f1`` quality, silence on a gold chunk scores 0.

The final test is the one that matters — it asserts the degenerate policy no longer
out-earns a competent extractor.
"""

from __future__ import annotations

import pytest

from kgat.data.backfill_export import ExtractionPair
from kgat.train.backfill_routing import (
    QUALITY_F1,
    QUALITY_WEIGHTED_PR,
    ROUTE_ESCALATE,
    ROUTE_EXTRACT,
    ROUTE_SKIP,
    ChunkDecision,
    chunk_quality,
    per_chunk_rewards,
)

GOLD = [("supplier", "Acme")]


def pair(triples, text="chunk text") -> ExtractionPair:
    return ExtractionPair(
        text=text, filer="Filer Co", triples=tuple(triples), filing="acc-1", company="co-1"
    )


def decision(route, triples=(), gen_tokens=10) -> ChunkDecision:
    return ChunkDecision(
        route=route, triples=tuple(triples), gen_tokens=gen_tokens, logprob=-1.0, n_choices=3
    )


def reward_for(pairs, decisions, *, mode, lam=0.0):
    rewards, _ = per_chunk_rewards(pairs, decisions, lam=lam, quality_mode=mode)
    return rewards


# --- the defect itself -------------------------------------------------------


def test_skipping_a_gold_chunk_scored_free_precision_under_the_old_reward():
    """Documents the hack: silence on a gold chunk used to bank w_p = 0.5."""
    r = reward_for([pair(GOLD)], [decision(ROUTE_SKIP)], mode=QUALITY_WEIGHTED_PR)
    assert r[0] == pytest.approx(0.5)


def test_skipping_a_gold_chunk_now_scores_zero():
    """The fix: nothing claimed on a chunk that HAS gold earns nothing."""
    r = reward_for([pair(GOLD)], [decision(ROUTE_SKIP)], mode=QUALITY_F1)
    assert r[0] == pytest.approx(0.0)


def test_skipping_an_EMPTY_chunk_still_scores_full_marks():
    """56.4% of real chunks have no gold edges — skipping them is CORRECT."""
    r = reward_for([pair([])], [decision(ROUTE_SKIP)], mode=QUALITY_F1)
    assert r[0] == pytest.approx(1.0)


def test_false_positives_on_an_empty_chunk_now_score_zero():
    """Old reward paid w_r = 0.5 for inventing an edge where there is no gold."""
    p, d = [pair([])], [decision(ROUTE_EXTRACT, triples=[("supplier", "Ghost")])]
    assert reward_for(p, d, mode=QUALITY_WEIGHTED_PR)[0] == pytest.approx(0.5)
    assert reward_for(p, d, mode=QUALITY_F1)[0] == pytest.approx(0.0)


def test_escalating_still_scores_full_quality_minus_cost():
    """The teacher solves it; only the cost term pushes back.

    Cost is CLAMPED to 1.0 per chunk: 10 generated + 2000 escalation over a 2000
    cap saturates, so the penalty is exactly lam.
    """
    r = reward_for([pair(GOLD)], [decision(ROUTE_ESCALATE)], mode=QUALITY_F1, lam=0.3)
    assert r[0] == pytest.approx(1.0 - 0.3, abs=1e-6)


def test_perfect_extraction_scores_full_quality():
    r = reward_for([pair(GOLD)], [decision(ROUTE_EXTRACT, triples=GOLD)], mode=QUALITY_F1)
    assert r[0] == pytest.approx(1.0)


# --- the property that actually closes the hack -------------------------------


# A REALISTIC extractor. The hack only bites when extraction is imperfect — a
# perfect extractor beats silence under either reward. The real policy runs at
# P/R around 0.65, so model a chunk it partly fumbles: 2 gold edges, one recovered,
# two spurious -> P = 1/3, R = 1/2.
GOLD2 = [("supplier", "Acme"), ("customer", "Beta")]
SLOPPY = [("supplier", "Acme"), ("partner", "Wrong"), ("investor", "AlsoWrong")]


def test_silence_outranked_honest_extraction_under_the_old_reward():
    """Route ranking on ONE gold chunk — the defect in its purest form.

    Skip banks w_p = 0.5 for claiming nothing; a real (imperfect) extraction
    attempt scores 0.5·(1/3) + 0.5·(1/2) = 0.417. Silence literally paid better
    than trying, independent of lambda.
    """
    p = [pair(GOLD2)]
    skip = reward_for(p, [decision(ROUTE_SKIP)], mode=QUALITY_WEIGHTED_PR)[0]
    work = reward_for(p, [decision(ROUTE_EXTRACT, triples=SLOPPY)], mode=QUALITY_WEIGHTED_PR)[0]
    assert skip > work, "the old reward paid more for silence than for imperfect work"


def test_honest_extraction_now_outranks_silence():
    """The fix, same chunk: skip -> 0, imperfect extraction -> F1 = 0.4."""
    p = [pair(GOLD2)]
    skip = reward_for(p, [decision(ROUTE_SKIP)], mode=QUALITY_F1)[0]
    work = reward_for(p, [decision(ROUTE_EXTRACT, triples=SLOPPY)], mode=QUALITY_F1)[0]
    assert work > skip
    assert work == pytest.approx(0.4, abs=1e-3)
    assert skip <= 0.0


def test_degenerate_never_extract_policy_no_longer_wins_the_episode():
    """THE regression test, at the episode level.

    2 empty chunks + 3 gold chunks. The degenerate policy is what GRPO actually
    converged to on 2026-07-19: skip what it can, escalate the rest, never emit an
    edge. The honest policy skips the empties and attempts extraction, imperfectly.

    lam = 0.7 prices escalation above the honest extractor's F1, so escalation is
    not the free win it was at lam = 0.05 — this isolates the SILENCE defect rather
    than re-testing lambda.
    """
    pairs = [pair([]), pair([]), pair(GOLD2), pair(GOLD2), pair(GOLD2)]
    degenerate = [
        decision(ROUTE_SKIP), decision(ROUTE_SKIP),
        decision(ROUTE_SKIP), decision(ROUTE_ESCALATE), decision(ROUTE_SKIP),
    ]
    honest = [
        decision(ROUTE_SKIP), decision(ROUTE_SKIP),
        decision(ROUTE_EXTRACT, triples=SLOPPY),
        decision(ROUTE_EXTRACT, triples=SLOPPY),
        decision(ROUTE_EXTRACT, triples=SLOPPY),
    ]
    lam = 0.7
    old_deg = sum(reward_for(pairs, degenerate, mode=QUALITY_WEIGHTED_PR, lam=lam))
    old_hon = sum(reward_for(pairs, honest, mode=QUALITY_WEIGHTED_PR, lam=lam))
    new_deg = sum(reward_for(pairs, degenerate, mode=QUALITY_F1, lam=lam))
    new_hon = sum(reward_for(pairs, honest, mode=QUALITY_F1, lam=lam))

    assert old_deg > old_hon, "the old reward should favour silence — that was the bug"
    assert new_hon > new_deg, "f1 must make honest extraction the better-paying policy"


def test_chunk_quality_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown quality mode"):
        chunk_quality(1.0, 1.0, "bogus", 0.5, 0.5)
