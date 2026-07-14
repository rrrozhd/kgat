"""Tests for the backfill distant-supervision pairs (loader + synthetic generator)."""

from __future__ import annotations

import json

import pytest

from kgat.data.backfill_export import (
    FINKG_RELATION_TO_TYPE,
    RELATIONSHIP_TYPES,
    ExtractionPair,
    export_dataset,
    generate_synthetic_pairs,
    load_export_jsonl,
    read_pairs_jsonl,
    split_by_company,
    split_by_filing,
    target_vocabulary,
    write_pairs_jsonl,
)
from kgat.data.finkg import FinKG


def test_taxonomy_matches_alphina_seven_types():
    assert RELATIONSHIP_TYPES == (
        "supplier",
        "customer",
        "competitor",
        "partner",
        "acquirer",
        "subsidiary",
        "investor",
    )
    assert set(FINKG_RELATION_TO_TYPE.values()) == set(RELATIONSHIP_TYPES)


def test_synthetic_filer_centric_mapping():
    # (X, supplier_to, C) means X supplies C -> filer X states type "supplier",
    # and the auto-added reverse (C, customer_of, X) -> filer C states "customer".
    kg = FinKG()
    kg.companies = ["Xcorp", "Ccorp"]
    kg.add_pair("Xcorp", "supplier_to", "Ccorp")
    pairs = generate_synthetic_pairs(kg, n_filings=2, seed=0, negatives_per_filing=1)

    facts = {(p.filer, t) for p in pairs for t in p.triples}
    assert ("Xcorp", ("supplier", "Ccorp")) in facts
    assert ("Ccorp", ("customer", "Xcorp")) in facts
    # The evidence sentence mentions the target by name.
    for p in pairs:
        for _, target in p.triples:
            assert target in p.text


def test_synthetic_pairs_deterministic_and_typed():
    a = generate_synthetic_pairs(n_filings=8, seed=7)
    b = generate_synthetic_pairs(n_filings=8, seed=7)
    assert a == b
    assert any(not p.triples for p in a)  # negatives present
    assert any(len(p.triples) > 1 for p in a)  # multi-triple chunks present
    for p in a:
        assert p.filing == f"synthetic:{p.filer}"
        for relation, _ in p.triples:
            assert relation in RELATIONSHIP_TYPES
        if p.triples:
            assert p.confidence is not None and 0.7 <= p.confidence <= 0.99


def test_split_by_filing_is_filing_disjoint():
    pairs = generate_synthetic_pairs(n_filings=20, seed=3)
    splits = split_by_filing(pairs, fractions=(0.6, 0.2, 0.2), seed=1)
    assert sum(len(v) for v in splits.values()) == len(pairs)
    filings = {s: {p.filing for p in v} for s, v in splits.items()}
    assert not (filings["train"] & filings["dev"])
    assert not (filings["train"] & filings["test"])
    assert not (filings["dev"] & filings["test"])


def test_pairs_jsonl_roundtrip(tmp_path):
    pairs = generate_synthetic_pairs(n_filings=4, seed=5)
    path = tmp_path / "pairs.jsonl"
    write_pairs_jsonl(pairs, path)
    assert read_pairs_jsonl(path) == pairs


def test_read_pairs_empty_raises(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        read_pairs_jsonl(empty)


def test_split_by_company_groups_filings_of_one_company(tmp_path):
    # Two filings of the same company (successive 10-Ks) must land in ONE split.
    pairs = [
        ExtractionPair(text=f"t{i}", filer="A Corp", triples=(), filing=f"acc-{i}", company="co-1")
        for i in range(4)
    ] + [
        ExtractionPair(text=f"u{i}", filer=f"B{i}", triples=(), filing=f"bcc-{i}", company=f"co-{i+2}")
        for i in range(8)
    ]
    splits = split_by_company(pairs, fractions=(0.5, 0.25, 0.25), seed=0)
    homes = {s for s, items in splits.items() if any(p.company == "co-1" for p in items)}
    assert len(homes) == 1  # never straddles splits
    assert sum(len(v) for v in splits.values()) == len(pairs)


def test_load_export_v2_groups_by_chunk_and_uses_chunk_text(tmp_path):
    rows = [
        # two edges in ONE chunk -> one pair whose text is the full chunk
        {
            "evidence_text": "We sell to A.",
            "filer": "Filer Co",
            "relationship_type": "supplier",
            "target": "A",
            "confidence": 0.9,
            "accession_number": "acc-1",
            "chunk_id": "ch-1",
            "chunk_text": "Long chunk. We sell to A. We compete with B. Boilerplate.",
            "item_key": "Item 1",
            "company_id": "co-9",
        },
        {
            "evidence_text": "We compete with B.",
            "filer": "Filer Co",
            "relationship_type": "competitor",
            "target": "B",
            "confidence": 0.8,
            "accession_number": "acc-1",
            "chunk_id": "ch-1",
            "chunk_text": "Long chunk. We sell to A. We compete with B. Boilerplate.",
            "item_key": "Item 1",
            "company_id": "co-9",
        },
        # negative chunk: NULL evidence, text from chunk_text
        {
            "evidence_text": None,
            "filer": "Filer Co",
            "relationship_type": None,
            "target": None,
            "confidence": None,
            "accession_number": "acc-1",
            "chunk_id": "ch-2",
            "chunk_text": "Pure boilerplate chunk.",
            "item_key": "Item 7",
            "company_id": "co-9",
        },
    ]
    path = tmp_path / "export_v2.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    pairs = load_export_jsonl(path)
    assert len(pairs) == 2
    multi = next(p for p in pairs if p.triples)
    assert multi.text.startswith("Long chunk.")
    assert set(multi.triples) == {("supplier", "A"), ("competitor", "B")}
    assert multi.company == "co-9" and multi.item_key == "Item 1"
    negative = next(p for p in pairs if not p.triples)
    assert negative.text == "Pure boilerplate chunk." and negative.item_key == "Item 7"
    # roundtrip keeps the new fields
    write_pairs_jsonl(pairs, tmp_path / "pairs.jsonl")
    assert read_pairs_jsonl(tmp_path / "pairs.jsonl") == pairs


def test_load_export_groups_rows_by_evidence(tmp_path):
    rows = [
        # two edges sharing one evidence chunk -> one multi-triple pair
        {
            "evidence_text": "We sell to A and compete with B.",
            "filer": "Filer Co",
            "relationship_type": "supplier",
            "target": "A",
            "confidence": 0.9,
            "accession_number": "0001-24-000001",
        },
        {
            "evidence_text": "We sell to A and compete with B.",
            "filer": "Filer Co",
            "relationship_type": "competitor",
            "target": "B",
            "confidence": 0.8,
            "accession_number": "0001-24-000001",
        },
        # negative chunk (NULL relationship_type)
        {
            "evidence_text": "Our fiscal year ends in June.",
            "filer": "Filer Co",
            "relationship_type": None,
            "target": None,
            "confidence": None,
            "accession_number": "0001-24-000001",
        },
    ]
    path = tmp_path / "export.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    pairs = load_export_jsonl(path)
    assert len(pairs) == 2
    multi = next(p for p in pairs if p.triples)
    assert set(multi.triples) == {("supplier", "A"), ("competitor", "B")}
    assert multi.confidence == 0.8  # min over the group's rows
    negative = next(p for p in pairs if not p.triples)
    assert negative.confidence is None


def test_load_export_rejects_unknown_type(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "evidence_text": "t",
                "filer": "F",
                "relationship_type": "owns",
                "target": "T",
                "confidence": 0.9,
                "accession_number": "a1",
            }
        )
    )
    with pytest.raises(ValueError, match="taxonomy"):
        load_export_jsonl(path)


def test_export_dataset_writes_splits_and_vocab(tmp_path):
    pairs = generate_synthetic_pairs(n_filings=10, seed=9)
    counts = export_dataset(pairs, tmp_path, seed=9)
    assert set(counts) == {"train", "dev", "test"}
    assert sum(counts.values()) == len(pairs)
    vocab = json.loads((tmp_path / "vocab.json").read_text())
    assert vocab["relations"] == list(RELATIONSHIP_TYPES)
    assert vocab["targets"] == target_vocabulary(pairs)
    assert vocab["targets"] == sorted(set(vocab["targets"]))
    reread = read_pairs_jsonl(tmp_path / "train.jsonl")
    assert all(isinstance(p, ExtractionPair) for p in reread)
