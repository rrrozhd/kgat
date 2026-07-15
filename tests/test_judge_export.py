"""Tests for the judge distillation data (pure, no torch)."""

from __future__ import annotations

import json

import pytest

from kgat.data.judge_export import (
    JudgeExample,
    export_judge_dataset,
    load_judge_export_jsonl,
    read_judge_jsonl,
    render_judge_input,
    split_judge_examples,
)
from kgat.train.judge import encode_judge_example, verdict_agreement


def make_rows():
    return [
        {  # accept with chunk text
            "evidence_text": "We purchase key components from Bolt Inc.",
            "chunk_text": "Long chunk. We purchase key components from Bolt Inc. More text.",
            "filer": "Filer Co",
            "relationship_type": "customer",
            "target": "Bolt Inc",
            "faithfulness": 0.92,
            "verdict": "accept",
            "accession_number": "acc-1",
        },
        {  # critic reject (payload-sourced)
            "evidence_text": "John Smith previously served at Bolt Inc.",
            "chunk_text": "Bio section. John Smith previously served at Bolt Inc.",
            "filer": "Filer Co",
            "relationship_type": "partner",
            "target": "Bolt Inc",
            "faithfulness": 0.1,
            "verdict": "reject",
            "accession_number": "acc-2",
        },
        {  # malformed: LLM payload lost the target
            "evidence_text": "t",
            "chunk_text": "t",
            "filer": "Filer Co",
            "relationship_type": "supplier",
            "target": None,
            "faithfulness": 0.5,
            "verdict": "reject",
            "accession_number": "acc-2",
        },
        {  # malformed: off-taxonomy legacy type
            "evidence_text": "t",
            "chunk_text": "t",
            "filer": "Filer Co",
            "relationship_type": "holds",
            "target": "X",
            "faithfulness": 0.5,
            "verdict": "accept",
            "accession_number": "acc-2",
        },
        {  # missing faithfulness -> defaults by verdict; no chunk -> quote fallback
            "evidence_text": "We compete with Acme Corp.",
            "chunk_text": None,
            "filer": "Filer Co",
            "relationship_type": "competitor",
            "target": "Acme Corp",
            "faithfulness": None,
            "verdict": "accept",
            "accession_number": "acc-3",
        },
    ]


def test_load_export_skips_malformed_and_prefers_chunk(tmp_path):
    path = tmp_path / "judge.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in make_rows()))
    examples, skipped = load_judge_export_jsonl(path)
    assert len(examples) == 3 and skipped == 2
    accept = examples[0]
    assert accept.text.startswith("Long chunk.")  # chunk preferred over quote
    assert accept.faithfulness == 0.92
    fallback = examples[2]
    assert fallback.text == "We compete with Acme Corp."  # quote fallback
    assert fallback.faithfulness == 1.0  # default for accept without a score


def test_split_and_roundtrip(tmp_path):
    path = tmp_path / "judge.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in make_rows()))
    examples, _ = load_judge_export_jsonl(path)
    splits = split_judge_examples(examples, fractions=(0.4, 0.3, 0.3), seed=0)
    homes = {e.filing: s for s, items in splits.items() for e in items}
    assert len(homes) == 3  # each filing in exactly one split
    counts = export_judge_dataset(examples, tmp_path / "out", fractions=(0.4, 0.3, 0.3), seed=0)
    assert sum(counts.values()) == len(examples)
    nonempty = next(s for s, c in counts.items() if c)
    reread = read_judge_jsonl(tmp_path / "out" / f"{nonempty}.jsonl")
    assert all(isinstance(e, JudgeExample) for e in reread)


def test_render_judge_input_is_stable():
    text = render_judge_input("Filer Co", "customer", "Bolt Inc", "evidence here")
    assert text == (
        "filer: Filer Co\nclaim: the filer is the customer of Bolt Inc\nevidence: evidence here"
    )


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, truncation: bool = False, max_length: int | None = None):
        ids = [ord(c) for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_encode_judge_example_truncates_and_labels():
    example = JudgeExample(
        text="long evidence " * 50,
        filer="F",
        relation="supplier",
        target="T",
        verdict="accept",
        faithfulness=0.75,
        filing="acc-1",
    )
    enc = encode_judge_example(example, FakeTokenizer(), max_seq_len=64)
    assert len(enc["input_ids"]) == 64
    assert enc["labels"] == 0.75


def test_verdict_agreement_is_per_class():
    preds = [0.9, 0.2, 0.4, 0.6]  # right, wrong, right, wrong
    verdicts = ["accept", "accept", "reject", "reject"]
    m = verdict_agreement(preds, verdicts)
    assert m["dev_verdict_accuracy"] == 0.5
    assert m["dev_accept_agreement"] == 0.5
    assert m["dev_reject_agreement"] == 0.5
    # Asymmetry is visible: perfect on accepts, broken on rejects.
    m = verdict_agreement([0.9, 0.8, 0.7], ["accept", "accept", "reject"])
    assert m["dev_accept_agreement"] == 1.0
    assert m["dev_reject_agreement"] == 0.0
    assert verdict_agreement([0.9], ["accept"])["dev_reject_agreement"] == 1.0  # vacuous
    with pytest.raises(ValueError):
        verdict_agreement([], [])


def test_empty_export_raises(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        load_judge_export_jsonl(empty)
