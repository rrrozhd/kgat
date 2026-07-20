"""Tests for the cascade threshold sweep (pure, no torch)."""

from __future__ import annotations

import json

from kgat.eval.extractor_cascade import (
    SIGNALS,
    ExtractionOutcome,
    cascade_rows,
    headline,
    micro_prf,
    quantile_taus,
    routing_auroc,
    write_summaries,
)
from kgat.eval.frontier import frontier_dataframe, load_summaries


def test_micro_prf_hand_computed():
    items = [
        ({("supplier", "A")}, {("supplier", "A")}),  # perfect positive
        ({("customer", "B")}, {("supplier", "B")}),  # 1 FP + 1 FN
        (set(), set()),  # correct NONE: invisible to P/R, visible to exact
        (set(), {("partner", "C")}),  # 1 FN (missed edge)
    ]
    m = micro_prf(items)
    assert m["precision"] == 1 / 2  # TP=1, FP=1
    assert m["recall"] == 1 / 3  # TP=1, FN=2
    assert m["exact"] == 2 / 4
    assert 0 < m["f1"] < 1


def test_micro_prf_all_none_is_perfect():
    m = micro_prf([(set(), set()), (set(), set())])
    assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact": 1.0}


OUTCOMES = [
    # confident and right
    ExtractionOutcome(gold=(("supplier", "A"),), pred=(("supplier", "A"),), confidence=0.95),
    # unconfident and wrong -> the case escalation should rescue
    ExtractionOutcome(gold=(("partner", "B"),), pred=(), confidence=0.30),
    # confident NONE, correct
    ExtractionOutcome(gold=(), pred=(), confidence=0.90),
    # mid-confidence, wrong relation
    ExtractionOutcome(gold=(("customer", "C"),), pred=(("investor", "C"),), confidence=0.60),
]


def test_cascade_endpoints_and_monotonicity():
    taus = [0.0, 0.4, 0.7, 1.05]
    rows = cascade_rows(OUTCOMES, taus)

    assert rows[0]["escalation_rate"] == 0.0  # tau=0: pure small model
    assert rows[0]["recall"] == 1 / 3
    assert rows[-1]["escalation_rate"] == 1.0  # tau>max conf: pure teacher
    assert rows[-1]["recall"] == 1.0
    assert rows[-1]["precision"] == 1.0
    assert rows[-1]["exact"] == 1.0

    # Escalation (cost) and recall both rise monotonically with tau.
    esc = [r["escalation_rate"] for r in rows]
    rec = [r["recall"] for r in rows]
    assert esc == sorted(esc)
    assert rec == sorted(rec)

    # tau=0.4 escalates exactly the 0.30-confidence miss and rescues it.
    row = rows[1]
    assert row["escalation_rate"] == 1 / 4
    assert row["recall"] == 2 / 3


def test_headline_picks_cheapest_gate_pass():
    rows = cascade_rows(OUTCOMES, [0.0, 0.4, 0.7, 1.05])
    best = headline(rows, min_recall=0.8)
    assert best is not None
    assert best["tau"] == 0.7  # first tau reaching recall >= 0.8 at min escalation
    assert headline(rows, min_recall=0.5)["tau"] == 0.4
    assert headline([r for r in rows if r["recall"] < 0.8], min_recall=0.8) is None


def test_cascade_rows_2d_and_pareto():
    from kgat.eval.extractor_cascade import cascade_rows_2d, pareto_front

    rows = cascade_rows_2d(OUTCOMES, taus_none=[0.0, 1.05], taus_extract=[0.0, 0.7])
    assert len(rows) == 4
    # (0,0): nothing escalates — matches the uniform tau=0 row.
    r00 = next(r for r in rows if r["tau_none"] == 0.0 and r["tau_extract"] == 0.0)
    assert r00["escalation_rate"] == 0.0 and r00["recall"] == 1 / 3
    # Escalating only low-confidence EXTRACTIONS rescues the wrong-relation
    # chunk without paying for NONE decodes.
    r_ext = next(r for r in rows if r["tau_none"] == 0.0 and r["tau_extract"] == 0.7)
    assert r_ext["escalation_rate"] == 1 / 4  # only the conf-0.60 extraction
    assert r_ext["recall"] == 2 / 3
    # tau_none=1.05 escalates every NONE decode (incl. the wrong empty pred).
    r_none = next(r for r in rows if r["tau_none"] == 1.05 and r["tau_extract"] == 0.0)
    assert r_none["escalation_rate"] == 2 / 4  # the empty-pred miss + correct NONE
    assert r_none["recall"] == 2 / 3

    front = pareto_front(rows)
    costs = [r["escalation_rate"] for r in front]
    quals = [r["recall"] for r in front]
    assert costs == sorted(costs)
    assert quals == sorted(quals)  # envelope is monotone
    assert r00 in front  # cheapest point always survives


# The wrong-relation chunk decodes confidently INSIDE the grammar's renormalized
# distribution but wanted out-of-grammar mass — exactly the case the masked
# confidence cannot see and grammar agreement can.
AGREE_OUTCOMES = [
    ExtractionOutcome(
        gold=(("supplier", "A"),), pred=(("supplier", "A"),), confidence=0.95,
        agreement=0.9, min_agreement=0.8,
    ),
    ExtractionOutcome(  # confident-but-clipped miss: agreement is the only tell
        gold=(("partner", "B"),), pred=(), confidence=0.90,
        agreement=0.3, min_agreement=0.1,
    ),
    ExtractionOutcome(
        gold=(), pred=(), confidence=0.90, agreement=0.95, min_agreement=0.9,
    ),
]


def test_signals_registry_and_fallback():
    o = AGREE_OUTCOMES[1]
    assert SIGNALS["confidence"](o) == 0.90
    assert SIGNALS["agreement"](o) == 0.3
    assert SIGNALS["min_agreement"](o) == 0.1
    assert SIGNALS["conf_x_agree"](o) == 0.90 * 0.3
    # Outcomes from decodes that never measured clipped mass degrade to the
    # confidence baseline instead of crashing or silently escalating everything.
    legacy = ExtractionOutcome(gold=(), pred=(), confidence=0.7)
    assert SIGNALS["agreement"](legacy) == 1.0
    assert SIGNALS["conf_x_agree"](legacy) == 0.7


def test_cascade_rows_with_agreement_signal():
    # Confidence ties the miss with a correct NONE at 0.90: rescuing it costs two
    # escalations. Agreement isolates the miss and pays for one.
    rows = cascade_rows(AGREE_OUTCOMES, [0.5], signal=SIGNALS["agreement"])
    assert rows[0]["escalation_rate"] == 1 / 3  # only the clipped miss
    assert rows[0]["recall"] == 1.0  # ...and escalating it rescues full recall
    conf = cascade_rows(AGREE_OUTCOMES, [0.91])
    assert conf[0]["escalation_rate"] == 2 / 3  # confidence pays double
    assert conf[0]["recall"] == 1.0


def test_routing_auroc_ranks_signals():
    # Agreement is a perfect error detector on this set; confidence ties the
    # erroneous chunk with a correct one (0.90 vs 0.90 -> half credit).
    assert routing_auroc(AGREE_OUTCOMES, SIGNALS["agreement"]) == 1.0
    assert routing_auroc(AGREE_OUTCOMES, SIGNALS["min_agreement"]) == 1.0
    assert routing_auroc(AGREE_OUTCOMES, SIGNALS["confidence"]) == 0.75
    assert routing_auroc(AGREE_OUTCOMES[:1]) is None  # degenerate: no errors


def test_routing_auroc_respects_uncertain_masking():
    # A prediction matching only sub-floor teacher edges scores as correct-empty,
    # so it must not count as an error the router should catch.
    masked = ExtractionOutcome(
        gold=(), pred=(("supplier", "X"),), confidence=0.5,
        uncertain=(("supplier", "X"),),
    )
    assert routing_auroc([masked, AGREE_OUTCOMES[0]]) is None  # both correct


def test_quantile_taus_trace_signal_range():
    taus = quantile_taus(AGREE_OUTCOMES, SIGNALS["agreement"])
    assert taus[0] == 0.0
    assert taus[-1] > 0.95  # past the max observed value: escalate everything
    assert taus == sorted(taus)
    # Sweeping them reaches both endpoints of the frontier.
    rows = cascade_rows(AGREE_OUTCOMES, taus, signal=SIGNALS["agreement"])
    assert rows[0]["escalation_rate"] == 0.0
    assert rows[-1]["escalation_rate"] == 1.0
    # Large value sets subsample to a bounded grid.
    many = [
        ExtractionOutcome(gold=(), pred=(), confidence=i / 1000, agreement=i / 1000)
        for i in range(1000)
    ]
    assert len(quantile_taus(many, SIGNALS["agreement"], n=41)) <= 41


def test_summaries_feed_the_standard_frontier(tmp_path):
    rows = cascade_rows(OUTCOMES, [0.0, 0.4, 1.05])
    paths = write_summaries(rows, tmp_path, n_questions=len(OUTCOMES))
    assert len(paths) == 3
    parsed = json.loads(paths[0].read_text())
    assert parsed["metrics"]["recall"] == rows[0]["recall"]
    assert parsed["mean_cost"]["escalation_rate"] == rows[0]["escalation_rate"]

    df = frontier_dataframe(
        load_summaries(tmp_path), accuracy_metric="recall", cost_axis="escalation_rate"
    )
    assert len(df) == 3
    assert list(df["mean_cost"]) == sorted(df["mean_cost"])  # frontier reads left-to-right
    assert df["accuracy"].iloc[-1] == 1.0
