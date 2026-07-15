"""Tests for the routing-GRPO pure helpers (no torch)."""

from __future__ import annotations

import pytest

from kgat.data.backfill_export import ExtractionPair
from kgat.train.grpo_routing import build_judge, group_by_filing
from kgat.train.judge import threshold_sweep, verdict_agreement


def pair(filing: str, i: int) -> ExtractionPair:
    return ExtractionPair(
        text=f"chunk {i} of {filing}", filer="F", triples=(), filing=filing, company=filing
    )


def test_group_by_filing_caps_and_determinism():
    pairs = [pair(f"acc-{f}", i) for f in range(5) for i in range(6)]
    eps = group_by_filing(pairs, max_chunks=4, max_filings=3, seed=7)
    assert len(eps) == 3
    assert all(len(ep) == 4 for ep in eps)
    assert all(len({p.filing for p in ep}) == 1 for ep in eps)  # episodes are one filing
    assert eps == group_by_filing(pairs, max_chunks=4, max_filings=3, seed=7)  # deterministic
    assert eps != group_by_filing(pairs, max_chunks=4, max_filings=3, seed=8)


def test_build_judge_specs():
    assert build_judge("distant") is None
    rules = build_judge("rules")
    grounded_pair = ExtractionPair(
        text="We compete with Acme.", filer="F", triples=(("competitor", "Acme"),),
        filing="a", company="a",
    )
    assert rules(grounded_pair, "competitor", "Acme") == 1.0  # gates + distant anchor
    assert rules(grounded_pair, "competitor", "Ghost Co") == 0.0  # ungrounded


def test_threshold_sweep_trades_the_classes():
    # Higher bar -> more rejects agreed, fewer accepts: monotone in both classes.
    preds = [0.9, 0.7, 0.55, 0.45, 0.35, 0.2]
    verdicts = ["accept", "accept", "reject", "accept", "reject", "reject"]
    rows = threshold_sweep(preds, verdicts, thresholds=[0.3, 0.5, 0.6, 0.8])
    rej = [r["dev_reject_agreement"] for r in rows]
    acc = [r["dev_accept_agreement"] for r in rows]
    assert rej == sorted(rej)
    assert acc == sorted(acc, reverse=True)
    assert all(0 <= r["balanced"] <= 1 for r in rows)
    # verdict_agreement at a custom bar matches the sweep row.
    assert rows[1] == {
        "threshold": 0.5,
        **verdict_agreement(preds, verdicts, threshold=0.5),
        "balanced": rows[1]["balanced"],
    }


def test_group_by_filing_rejects_nothing_silently():
    with pytest.raises(TypeError):
        group_by_filing(None)  # type: ignore[arg-type]
