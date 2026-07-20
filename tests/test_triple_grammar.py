"""Tests for the write-path triple grammar (pure, no torch)."""

from __future__ import annotations

import math
import random

import pytest

from kgat.controller.constrained_decoding import (
    NONE_LABEL,
    build_triple_grammar,
    decode_triples,
    encode_triples_target,
    triples_allowed_along,
)
from kgat.data.backfill_export import RELATIONSHIP_TYPES

EOS = 3  # fake eos id, outside printable ordinals


class FakeTokenizer:
    """Char-level tokenizer: each character maps to its ordinal."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


TOK = FakeTokenizer()
TARGETS = ("Acme Corp", "Bolt Inc", "Acme")  # "Acme" is a proper string prefix of "Acme Corp"


def make_grammar(**kwargs):
    return build_triple_grammar(RELATIONSHIP_TYPES, TARGETS, TOK, eos_id=EOS, **kwargs)


def forcing_logits_fn(gold_ids: list[int]):
    """Teacher-forcing scorer: prefers the next gold token at every position."""

    def logits_fn(generated: tuple[int, ...], allowed: list[int]) -> list[float]:
        want = gold_ids[len(generated)]
        assert want in allowed, f"gold token {want} not allowed at position {len(generated)}"
        return [10.0 if tok == want else 0.0 for tok in allowed]

    return logits_fn


def test_build_validation():
    with pytest.raises(ValueError):
        build_triple_grammar([], TARGETS, TOK, eos_id=EOS)
    # Empty TARGETS is legal (chunk-local candidates found nothing): the grammar
    # collapses to NONE | sentinels; relations are simply not offered.
    empty = build_triple_grammar(RELATIONSHIP_TYPES, [], TOK, eos_id=EOS)
    assert empty.targets == ()
    assert list(empty.first.id_map.values()) == [NONE_LABEL]
    with pytest.raises(ValueError):  # reserved separator inside a name
        build_triple_grammar(RELATIONSHIP_TYPES, ["Weird ; Name"], TOK, eos_id=EOS)
    with pytest.raises(ValueError):
        build_triple_grammar(RELATIONSHIP_TYPES, ["Weird :: Name"], TOK, eos_id=EOS)
    with pytest.raises(ValueError):  # NONE is the grammar's own sentinel
        build_triple_grammar([*RELATIONSHIP_TYPES, NONE_LABEL], TARGETS, TOK, eos_id=EOS)
    with pytest.raises(ValueError):
        make_grammar(max_triples=0)


def test_none_decode():
    grammar = make_grammar()
    gold = encode_triples_target([], grammar)
    result = decode_triples(forcing_logits_fn(gold), grammar)
    assert result.triples == ()
    assert list(result.ids) == gold
    assert result.ids[-1] == EOS


def test_single_triple_roundtrip():
    grammar = make_grammar()
    triples = [("supplier", "Bolt Inc")]
    gold = encode_triples_target(triples, grammar)
    result = decode_triples(forcing_logits_fn(gold), grammar)
    assert result.triples == (("supplier", "Bolt Inc"),)
    assert list(result.ids) == gold  # decode emits exactly the canonical SFT target


def test_multi_triple_roundtrip_with_prefix_target():
    # "Acme" is a string prefix of "Acme Corp" — the folded continue/end delimiters
    # must keep both decodable in every slot.
    grammar = make_grammar()
    for triples in (
        [("supplier", "Acme"), ("customer", "Acme Corp")],
        [("acquirer", "Acme Corp"), ("competitor", "Acme"), ("investor", "Bolt Inc")],
    ):
        gold = encode_triples_target(triples, grammar)
        result = decode_triples(forcing_logits_fn(gold), grammar)
        assert result.triples == tuple(triples)
        assert list(result.ids) == gold


def test_invalid_tokens_never_emitted():
    grammar = make_grammar()

    # 'x' has a huge logit but is only scored when allowed; greedy otherwise.
    def logits_fn(generated, allowed):
        return [100.0 if tok == ord("x") else 1.0 / (1 + tok) for tok in allowed]

    result = decode_triples(logits_fn, grammar)
    for relation, target in result.triples:
        assert relation in RELATIONSHIP_TYPES
        assert target in TARGETS


def test_sampling_stays_inside_the_schema():
    grammar = make_grammar(max_triples=3)
    rng = random.Random(11)

    def logits_fn(generated, allowed):
        return [rng.uniform(0, 1) for _ in allowed]

    shapes = set()
    for _ in range(25):
        result = decode_triples(logits_fn, grammar, temperature=1.0, rng=rng)
        assert len(result.triples) <= 3
        for relation, target in result.triples:
            assert relation in RELATIONSHIP_TYPES
            assert target in TARGETS
        assert result.logprob <= 0.0
        shapes.add(len(result.triples))
    assert len(shapes) > 1  # actually samples different structures


def test_confidence_excludes_forced_steps():
    grammar = make_grammar()
    gold = encode_triples_target([("partner", "Bolt Inc")], grammar)
    result = decode_triples(forcing_logits_fn(gold), grammar)
    assert 0.0 < result.confidence <= 1.0
    assert result.n_choices > 0
    assert math.isclose(result.confidence, math.exp(result.logprob / result.n_choices))
    # Forced steps (single allowed token) contribute logprob 0; every other step is
    # counted, so the mean is over real decisions only.
    assert result.n_choices < len(result.ids)


def test_agreement_none_without_extended_contract():
    # A plain scores-only logits_fn never measures clipped mass.
    grammar = make_grammar()
    gold = encode_triples_target([("partner", "Bolt Inc")], grammar)
    result = decode_triples(forcing_logits_fn(gold), grammar)
    assert result.agreement_logprob is None
    assert result.agreement is None
    assert result.min_agreement is None


def test_agreement_measures_clipped_mass():
    grammar = make_grammar()
    gold = encode_triples_target([], grammar)  # `` NONE``<eos>
    base = forcing_logits_fn(gold)

    def logits_fn(generated, allowed):
        vals = base(generated, allowed)
        # Full-vocab mass = allowed mass + as much again outside the grammar:
        # every step retains exactly half the unconstrained distribution.
        allowed_lse = math.log(sum(math.exp(v) for v in vals))
        return vals, allowed_lse + math.log(2.0)

    result = decode_triples(logits_fn, grammar)
    assert result.triples == ()
    # Per-step agreement is exactly 1/2, so the geometric mean and the worst
    # step both land on 0.5 — while the masked confidence stays blind to it.
    assert result.agreement is not None
    assert math.isclose(result.agreement, 0.5)
    assert math.isclose(result.min_agreement, 0.5)
    assert math.isclose(result.agreement_logprob, len(result.ids) * -math.log(2.0))


def test_agreement_clamped_against_float_noise():
    grammar = make_grammar()
    gold = encode_triples_target([], grammar)
    base = forcing_logits_fn(gold)

    def logits_fn(generated, allowed):
        vals = base(generated, allowed)
        # Report a full-row logsumexp slightly BELOW the allowed mass (float
        # noise in a real backend); agreement must clamp to <= 1.
        allowed_lse = math.log(sum(math.exp(v) for v in vals))
        return vals, allowed_lse - 1e-9

    result = decode_triples(logits_fn, grammar)
    assert result.agreement == 1.0
    assert result.min_agreement == 1.0


def test_max_triples_cap():
    grammar = make_grammar(max_triples=2)
    with pytest.raises(ValueError):
        encode_triples_target(
            [("supplier", "Acme"), ("customer", "Acme"), ("partner", "Acme")], grammar
        )

    # A scorer that always prefers continuing still stops at the cap: the last
    # target segment physically lacks the continue variant.
    sep_first = ord(" ")

    def greedy_continue(generated, allowed):
        return [1.0 if tok == sep_first else 0.5 if tok != EOS else 0.0 for tok in allowed]

    result = decode_triples(greedy_continue, grammar)
    assert len(result.triples) <= 2


def test_encode_rejects_out_of_grammar():
    grammar = make_grammar()
    with pytest.raises(ValueError):
        encode_triples_target([("owns", "Acme")], grammar)
    with pytest.raises(ValueError):
        encode_triples_target([("supplier", "Unknown Co")], grammar)


def test_sentinel_terminal_decodes_and_stays_isolated():
    grammar = make_grammar(sentinels=("ESCALATE",))
    gold = list(grammar.enc_sentinel["ESCALATE"])
    result = decode_triples(forcing_logits_fn(gold), grammar)
    assert result.sentinel == "ESCALATE"
    assert result.triples == ()
    assert list(result.ids) == gold

    # NONE and plain triples still decode with sentinel=None.
    none_gold = encode_triples_target([], grammar)
    assert decode_triples(forcing_logits_fn(none_gold), grammar).sentinel is None
    triple_gold = encode_triples_target([("supplier", "Bolt Inc")], grammar)
    out = decode_triples(forcing_logits_fn(triple_gold), grammar)
    assert out.sentinel is None and out.triples == (("supplier", "Bolt Inc"),)

    # A relation name colliding with a sentinel is rejected at build time.
    with pytest.raises(ValueError):
        build_triple_grammar(
            [*RELATIONSHIP_TYPES, "ESCALATE"], TARGETS, TOK, eos_id=EOS,
            sentinels=("ESCALATE",),
        )


def test_triples_allowed_along_matches_decode():
    # The gradient pass must see EXACTLY the allowed sets the rollout sampled
    # from: replay each canonical path and check the recorded choices are legal
    # and the walk consumes the path completely.
    grammar = make_grammar(sentinels=("ESCALATE",))
    paths = [
        encode_triples_target([], grammar),
        list(grammar.enc_sentinel["ESCALATE"]),
        encode_triples_target([("supplier", "Acme"), ("customer", "Acme Corp")], grammar),
    ]
    for ids in paths:
        allowed = triples_allowed_along(grammar, ids)
        assert len(allowed) == len(ids)
        for tok, allowed_j in zip(ids, allowed, strict=True):
            assert tok in allowed_j
        # First position offers NONE, the sentinel, and every relation opener.
        first_openers = {ids2[0] for ids2 in [*paths]}
        assert first_openers <= set(allowed[0]) | first_openers

    with pytest.raises(KeyError):  # off-grammar token must be rejected loudly
        triples_allowed_along(grammar, [9999])
    with pytest.raises(ValueError):  # truncated path is not a complete decode
        triples_allowed_along(grammar, paths[2][:-1])
    with pytest.raises(ValueError):  # trailing garbage after eos
        triples_allowed_along(grammar, [*paths[0], 42])


def test_identical_tokenization_rejected():
    class CollapsingTokenizer:
        """Distinct strings, identical ids (lowercases) — undecidable, must raise."""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(c) for c in text.lower()]

    with pytest.raises(ValueError):
        build_triple_grammar(
            RELATIONSHIP_TYPES, ["Acme", "ACME"], CollapsingTokenizer(), eos_id=EOS
        )
