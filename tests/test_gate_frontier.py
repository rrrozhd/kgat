"""Gate curve mechanics — pure, no torch."""

from __future__ import annotations

from kgat.eval.gate_frontier import curve_for_ordering, evaluate_gate


def rows():
    # 2 empty chunks (extractor correct), 2 gold chunks it fumbles.
    return [
        {"gold": set(), "pred": set(), "confidence": 0.9},
        {"gold": set(), "pred": set(), "confidence": 0.8},
        {"gold": {("s", "A")}, "pred": set(), "confidence": 0.7},
        {"gold": {("s", "B")}, "pred": {("s", "WRONG")}, "confidence": 0.6},
    ]


def test_escalating_everything_recovers_all_gold():
    c = curve_for_ordering(rows(), [0, 1, 2, 3], [1.0])[0]
    assert c["recall"] == 1.0 and c["precision"] == 1.0


def test_escalating_nothing_reflects_the_raw_extractor():
    c = curve_for_ordering(rows(), [], [0.0])[0]
    assert c["recall"] == 0.0  # both gold edges missed
    assert c["precision"] == 0.0  # the one emitted edge is wrong


def test_a_gate_that_ranks_failures_first_beats_one_that_does_not():
    r = rows()
    good = [0.1, 0.1, 0.9, 0.9]  # scores the EMPTY chunks lowest -> escalates them (bad)
    bad = [0.9, 0.9, 0.1, 0.1]   # scores the FUMBLED chunks lowest -> escalates them (good)
    at_half = lambda s: evaluate_gate(r, s, fractions=[0.5])["gate"][0]["f1"]  # noqa: E731
    assert at_half(bad) > at_half(good)


def test_evaluate_gate_returns_matched_reference_orderings():
    out = evaluate_gate(rows(), [0.5] * 4, fractions=[0.25])
    assert set(out) == {"gate", "confidence", "oracle", "random"}
    # The oracle must be at least as good as any other ordering at equal cost.
    assert out["oracle"][0]["f1"] >= max(out[k][0]["f1"] for k in ("gate", "confidence", "random"))


def test_agreement_orderings_appear_when_logged():
    r = rows()
    # Old outcomes (no agreement column) -> no agreement curves.
    assert "agreement" not in evaluate_gate(r, [0.5] * 4, fractions=[0.25])
    # The fumbled chunks decoded confidently but wanted out-of-grammar mass:
    # agreement ranks them first even though confidence ranks them LAST.
    for row, a in zip(r, [0.95, 0.9, 0.2, 0.1], strict=True):
        row["agreement"] = a
    for row in r:
        row["confidence"] = 1.0 - row["confidence"]  # invert: failures now most confident
    out = evaluate_gate(r, [0.5] * 4, fractions=[0.5])
    assert {"agreement", "conf_x_agree"} <= set(out)
    assert out["agreement"][0]["f1"] > out["confidence"][0]["f1"]
    # Rows missing the column mid-file degrade to full agreement, not a crash.
    r[0]["agreement"] = None
    assert "agreement" in evaluate_gate(r, [0.5] * 4, fractions=[0.5])


def test_score_length_mismatch_is_an_error():
    import pytest

    with pytest.raises(ValueError, match="gate scores"):
        evaluate_gate(rows(), [0.1, 0.2])
