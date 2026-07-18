"""Escalate-probability sweep: the routing analog of the cascade's tau curve.

Sweep-1 reported ONE greedy point per policy and lost the continuous signal the
grammar already exposes (P(ESCALATE) in the first decision segment). These cover
the pure parts — path scoring and the threshold sweep — with no torch involved.
"""

from __future__ import annotations

import math

from kgat.controller.constrained_decoding import build_triple_grammar
from kgat.eval.routing_frontier import escalate_logprob, sweep_escalate_threshold
from kgat.train.backfill_routing import ESCALATE_LABEL


class _CharTokenizer:
    """One id per character; ids offset so 0 stays free for eos."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [ord(c) for c in text]


def _grammar():
    return build_triple_grammar(
        ["supplier", "customer"],
        ["Acme", "Globex"],
        _CharTokenizer(),
        eos_id=0,
        max_triples=4,
        sentinels=(ESCALATE_LABEL,),
    )


def test_escalate_logprob_is_a_normalized_path_probability():
    """Uniform logits => P(ESCALATE) = product of 1/|allowed| along its path."""
    g = _grammar()
    seen: list[int] = []

    def logits_fn(generated, allowed):  # noqa: ARG001
        seen.append(len(allowed))
        return [0.0] * len(allowed)  # uniform

    lp = escalate_logprob(logits_fn, g)
    expected = sum(-math.log(n) for n in seen)
    assert lp == expected
    assert 0.0 < math.exp(lp) <= 1.0


def test_escalate_logprob_responds_to_logits():
    """Favouring the sentinel's DISTINGUISHING token must raise its probability.

    Not the first token: every first-segment entry is ``target_text``-prefixed with
    a space, so step 0 is a forced step. That is exactly why this scores the whole
    path instead of reading a single step-0 scalar.
    """
    g = _grammar()
    esc_ids = g.enc_sentinel[ESCALATE_LABEL]
    branch = next(t for t in esc_ids if t != esc_ids[0])

    def flat(generated, allowed):  # noqa: ARG001
        return [0.0] * len(allowed)

    def favour(generated, allowed):  # noqa: ARG001
        return [5.0 if t == branch else 0.0 for t in allowed]

    assert escalate_logprob(favour, g) > escalate_logprob(flat, g)


def test_escalate_path_first_token_is_forced():
    """Guards the assumption above: step 0 carries no routing information."""
    g = _grammar()
    esc_first = g.enc_sentinel[ESCALATE_LABEL][0]
    assert g.first.trie.allowed() == [esc_first], "step 0 must be a single forced token"


def _rows():
    # p_escalate ascending; the 2 high-p chunks are the ones the model got wrong.
    return [
        {"p_escalate": 0.1, "pred": [["supplier", "Acme"]], "gold": [["supplier", "Acme"]]},
        {"p_escalate": 0.2, "pred": [], "gold": []},
        {"p_escalate": 0.8, "pred": [["supplier", "Globex"]], "gold": [["customer", "Globex"]]},
        {"p_escalate": 0.9, "pred": [], "gold": [["supplier", "Acme"]]},
    ]


def test_sweep_is_monotone_in_escalation():
    curve = sweep_escalate_threshold(_rows())
    rates = [c["escalation_rate"] for c in curve]
    assert rates == sorted(rates, reverse=True), "lower theta must escalate more"
    assert max(rates) == 1.0  # theta=0 escalates everything
    assert min(rates) == 0.0  # theta above every p escalates nothing


def test_sweep_endpoints_are_correct():
    curve = sweep_escalate_threshold(_rows())
    escalate_all = next(c for c in curve if c["escalation_rate"] == 1.0)
    # The teacher solves every chunk: perfect recall, no false positives.
    assert escalate_all["recall"] == 1.0
    assert escalate_all["precision"] == 1.0

    escalate_none = next(c for c in curve if c["escalation_rate"] == 0.0)
    # Own decodes only: 3 gold edges, 1 recovered; 2 emitted, 1 of them wrong.
    assert escalate_none["recall"] == 1 / 3
    assert escalate_none["precision"] == 0.5


def test_sweep_rewards_a_signal_that_ranks_errors_first():
    """Escalating the 2 wrong chunks first must beat escalating the 2 right ones."""
    good = sweep_escalate_threshold(_rows())
    flipped = [
        {**r, "p_escalate": 1.0 - r["p_escalate"]} for r in _rows()
    ]
    bad = sweep_escalate_threshold(flipped)

    def f1_at(curve, rate):
        return max(c["f1"] for c in curve if c["escalation_rate"] <= rate)

    assert f1_at(good, 0.5) > f1_at(bad, 0.5)


def test_sweep_handles_empty_rows():
    assert sweep_escalate_threshold([]) != []  # still returns thetas, no crash
