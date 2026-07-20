"""Escalation-gate training data: the target must match the routing reward's F1.

If these two definitions drift, the gate learns to predict something the system
does not score — so the equivalence is pinned by test, not by comment.
"""

from __future__ import annotations

import json

import pytest

from kgat.data.gate_export import chunk_f1, gate_examples, write_gate_splits
from kgat.train.backfill_routing import QUALITY_F1, chunk_quality


def test_chunk_f1_conventions():
    assert chunk_f1(set(), set()) == 1.0  # correct skip — 56.4% of real chunks
    assert chunk_f1({("s", "A")}, set()) == 0.0  # missed everything
    assert chunk_f1(set(), {("s", "A")}) == 0.0  # pure false positives
    assert chunk_f1({("s", "A")}, {("s", "A")}) == 1.0
    # one of two found, one spurious -> P=1/2, R=1/2
    assert chunk_f1({("s", "A"), ("c", "B")}, {("s", "A"), ("p", "X")}) == pytest.approx(0.5)


def test_target_matches_the_routing_reward_quality():
    """The gate predicts exactly what the fixed reward scores."""
    gold, pred = {("s", "A"), ("c", "B")}, {("s", "A"), ("p", "X")}
    tp = len(gold & pred)
    precision, recall = tp / len(pred), tp / len(gold)
    assert chunk_f1(gold, pred) == pytest.approx(
        chunk_quality(precision, recall, QUALITY_F1, 0.5, 0.5)
    )


def test_gate_examples_carry_targets_and_extra_signals():
    outcomes = [
        {"gold": [["s", "A"]], "pred": [["s", "A"]], "confidence": 0.9, "p_escalate": 0.05},
        {"gold": [["s", "B"]], "pred": [], "confidence": 0.3, "p_escalate": 0.80},
    ]
    ex = gate_examples(outcomes, ["chunk one", "chunk two"], ["Filer A", "Filer B"])
    assert [e["target"] for e in ex] == [1.0, 0.0]
    assert ex[1]["confidence"] == 0.3 and ex[1]["p_escalate"] == 0.80
    assert ex[0]["n_gold"] == 1 and ex[1]["n_pred"] == 0


def test_gate_examples_rejects_misalignment():
    """A silent zip-truncation here would train the gate on the wrong chunks."""
    with pytest.raises(ValueError, match="length mismatch"):
        gate_examples([{"gold": [], "pred": []}], ["a", "b"], ["F", "F"])


def test_gate_examples_applies_loose_normalization():
    outcomes = [{"gold": [["s", "Acme Inc."]], "pred": [["s", "Acme"]]}]
    strict = gate_examples(outcomes, ["t"], ["F"])
    loose = gate_examples(outcomes, ["t"], ["F"], normalize=lambda s: s.replace(" Inc.", ""))
    assert strict[0]["target"] == 0.0
    assert loose[0]["target"] == 1.0


def test_splits_hold_filers_apart(tmp_path):
    examples = [
        {"text": f"c{i}", "filer": f"F{i % 10}", "target": 0.5} for i in range(200)
    ]
    counts = write_gate_splits(examples, tmp_path, dev_frac=0.2)
    assert counts["train"] + counts["dev"] == 200
    train = [json.loads(x) for x in (tmp_path / "train.jsonl").read_text().splitlines()]
    dev = [json.loads(x) for x in (tmp_path / "dev.jsonl").read_text().splitlines()]
    assert {e["filer"] for e in train}.isdisjoint({e["filer"] for e in dev})


def test_splits_are_deterministic(tmp_path):
    examples = [{"text": f"c{i}", "filer": f"F{i % 7}", "target": 0.1} for i in range(50)]
    a = write_gate_splits(examples, tmp_path / "a")
    b = write_gate_splits(examples, tmp_path / "b")
    assert a == b
