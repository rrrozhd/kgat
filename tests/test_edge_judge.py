"""Tests for the objective edge-judge rule stack (pure, no torch)."""

from __future__ import annotations

from kgat.data.backfill_export import ExtractionPair
from kgat.train.edge_judge import (
    RuleGates,
    filer_is_party,
    grounded,
    make_rule_judge,
    normalize_people,
)

CHUNK = (
    "We rely on Taiwan Semiconductor Manufacturing Company for substantially all of "
    "our wafer fabrication. We compete directly with Intel Corporation and NVIDIA "
    "Corporation. John A. Smith has served as our Chief Executive Officer since 2019."
)


def pair(text=CHUNK, triples=()) -> ExtractionPair:
    return ExtractionPair(
        text=text, filer="Advanced Micro Devices, Inc.", triples=tuple(triples),
        filing="acc-1", company="co-1",
    )


def test_grounding_handles_suffix_drift_and_case():
    assert grounded("Intel Corporation", CHUNK)
    assert grounded("INTEL", CHUNK)  # suffix-stripped later mention
    assert grounded("NVIDIA Corp", CHUNK)  # different suffix than the text's
    assert not grounded("Advanced Micro Devices", CHUNK)  # filer, not in own chunk
    assert not grounded("Broadcom Inc", CHUNK)
    # Alias table grounds resolver-normalized names that never appear verbatim.
    assert grounded("TSMC", CHUNK, aliases={"TSMC": ["Taiwan Semiconductor Manufacturing"]})
    # Suffix stripping never empties a name into a match-everything core.
    assert not grounded("Inc", CHUNK)


def test_filer_party_cues():
    assert filer_is_party(CHUNK)  # "We rely on ..."
    assert filer_is_party("The Company maintains a joint venture with Bolt.")
    assert not filer_is_party(
        "Intel Corporation acquired Altera. Broadcom announced a partnership with Apple."
    )


def test_person_gate_uses_known_people():
    gates = RuleGates(known_people=normalize_people(["John A. Smith"]))
    assert not gates.evaluate(pair(), "John A. Smith").passed  # board-bio edge killed
    assert gates.evaluate(pair(), "Intel Corporation").passed


def test_rule_judge_distant_anchor_and_model_lift():
    p = pair(triples=[("customer", "Taiwan Semiconductor Manufacturing Company")])
    judge = make_rule_judge()
    # Grounded + party + in teacher labels -> 1.0.
    assert judge(p, "customer", "Taiwan Semiconductor Manufacturing Company") == 1.0
    # Grounded but NOT in teacher labels -> 0.0 (rules alone cannot certify novel edges).
    assert judge(p, "competitor", "Intel Corporation") == 0.0
    # Ungrounded target -> gate zero regardless of labels.
    assert judge(p, "customer", "Broadcom Inc") == 0.0

    # The distilled model lifts the ceiling: novel grounded edges earn their score.
    lifted = make_rule_judge(type_score=lambda pr, r, t: 0.87)
    assert lifted(p, "competitor", "Intel Corporation") == 0.87
    assert lifted(p, "customer", "Broadcom Inc") == 0.0  # gates still veto
    clamped = make_rule_judge(type_score=lambda pr, r, t: 3.0)
    assert clamped(p, "competitor", "Intel Corporation") == 1.0


def test_bystander_chunk_fails_party_gate():
    bystander = pair(text="Intel Corporation acquired Altera in an all-cash deal.")
    judge = make_rule_judge(type_score=lambda pr, r, t: 1.0)
    assert judge(bystander, "acquirer", "Intel Corporation") == 0.0
