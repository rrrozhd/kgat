"""Direction-flipped hard negatives for judge v3 (weak-supervision contrastive).

Verifies the minting guards: only from confident accepts, skip genuinely
bidirectional pairs, skip existing (chunk, claim) rows, and dedup.
"""

from __future__ import annotations

from kgat.data.judge_export import (
    DIRECTION_FLIP,
    JudgeExample,
    mint_direction_negatives,
)


def ex(relation, target, verdict="accept", faith=0.9, text="chunk-A", filer="Acme", filing="F1"):
    return JudgeExample(
        text=text, filer=filer, relation=relation, target=target,
        verdict=verdict, faithfulness=faith, filing=filing,
    )


def test_flips_supplier_to_customer_as_reject():
    negs = mint_direction_negatives([ex("supplier", "BoltCo")])
    assert len(negs) == 1
    n = negs[0]
    assert n.relation == "customer" and n.target == "BoltCo"
    assert n.verdict == "reject" and n.faithfulness == 0.0
    assert n.text == "chunk-A" and n.filer == "Acme" and n.filing == "F1"


def test_all_directional_pairs_flip():
    for r, flipped in DIRECTION_FLIP.items():
        negs = mint_direction_negatives([ex(r, "X")])
        assert negs and negs[0].relation == flipped


def test_non_directional_relations_are_skipped():
    for r in ("partner", "competitor", "investor"):
        assert mint_direction_negatives([ex(r, "X")]) == []


def test_only_from_accepts():
    assert mint_direction_negatives([ex("supplier", "BoltCo", verdict="reject", faith=0.0)]) == []


def test_low_confidence_source_skipped():
    # An accept below the confidence floor is not a trustworthy positive to flip.
    assert mint_direction_negatives([ex("supplier", "BoltCo", faith=0.3)]) == []


def test_bidirectional_pair_not_minted():
    # Acme is BOTH supplier and customer of BoltCo (both accepted) -> minting the
    # customer flip would contradict a real accepted edge; must be skipped.
    exs = [ex("supplier", "BoltCo"), ex("customer", "BoltCo", text="chunk-B")]
    negs = mint_direction_negatives(exs)
    assert negs == []


def test_existing_chunk_claim_not_duplicated():
    # A real reject of the exact flipped claim on the same chunk already exists.
    exs = [
        ex("supplier", "BoltCo"),
        ex("customer", "BoltCo", verdict="reject", faith=0.0),  # same chunk-A
    ]
    negs = mint_direction_negatives(exs)
    assert negs == []


def test_dedup_within_batch():
    # Two identical accepted supplier edges on the same chunk -> one negative.
    negs = mint_direction_negatives([ex("supplier", "BoltCo"), ex("supplier", "BoltCo")])
    assert len(negs) == 1


def test_same_flip_different_chunk_both_minted():
    # Same claim but different evidence chunks are distinct negatives.
    negs = mint_direction_negatives([
        ex("supplier", "BoltCo", text="chunk-A"),
        ex("supplier", "BoltCo", text="chunk-B"),
    ])
    assert len(negs) == 2
    assert {n.text for n in negs} == {"chunk-A", "chunk-B"}


def test_case_insensitive_bidirectional_guard():
    # "boltco" vs "BoltCo" must be treated as the same target for the guard.
    exs = [ex("supplier", "BoltCo"), ex("customer", "boltco", text="chunk-B")]
    assert mint_direction_negatives(exs) == []
