"""Tests for the token-trie constrained decoding (pure, no torch)."""

from __future__ import annotations

import math
import random

import pytest

from kgat.controller.constrained_decoding import (
    STOP_TOKEN,
    allowed_along,
    build_relation_trie,
    constrained_decode,
    target_text,
)

END = 1  # fake end-of-candidate token id


class FakeTokenizer:
    """Char-level tokenizer: each character maps to its ordinal."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


TOK = FakeTokenizer()
SP, A, B, C, Z = ord(" "), ord("a"), ord("b"), ord("c"), ord("z")
VOCAB = 500


def _logits(prefer: dict[int, float]) -> list[float]:
    row = [0.0] * VOCAB
    for tok, val in prefer.items():
        row[tok] = val
    return row


def test_trie_structure_and_id_map():
    trie, id_map = build_relation_trie(["ab", "ac", "zz"], TOK, end_id=END)
    # Every candidate (+ auto-added STOP) leads with the target_text space.
    assert trie.allowed() == [SP]
    after_space = trie.step(SP)
    assert set(after_space.allowed()) == {A, Z, ord("[")}  # 'a', 'z', '[STOP]'
    ab_ids = tuple(TOK.encode(target_text("ab"))) + (END,)
    assert id_map[ab_ids] == "ab"
    stop_ids = tuple(TOK.encode(target_text(STOP_TOKEN))) + (END,)
    assert id_map[stop_ids] == STOP_TOKEN


def test_prefix_candidates_are_disambiguated_by_end_id():
    # "a" is a strict prefix of "ab" — the end_id leaf keeps both decodable.
    trie, id_map = build_relation_trie(["a", "ab"], TOK, end_id=END, include_stop=False)
    node = trie.step(SP).step(A)
    assert set(node.allowed()) == {END, B}
    assert id_map[(SP, A, END)] == "a"
    assert id_map[(SP, A, B, END)] == "ab"


def test_greedy_decode_masks_out_invalid_tokens():
    trie, id_map = build_relation_trie(["ab", "ac", "zz"], TOK, end_id=END, include_stop=False)

    # 'x' has a huge logit everywhere but is never a valid continuation; among the
    # valid ones, prefer 'a' then 'c' => decode "ac".
    def logits_fn(generated, allowed):
        row = _logits({ord("x"): 100.0, A: 2.0, Z: 1.0, C: 3.0, B: 1.0, END: 0.5})
        return [row[t] for t in allowed]

    result = constrained_decode(logits_fn, trie, id_map)
    assert result.candidate == "ac"
    assert result.logprob < 0.0  # a real log-probability, not a raw logit


def test_greedy_logprob_matches_manual_softmax():
    trie, id_map = build_relation_trie(["a", "z"], TOK, end_id=END, include_stop=False)

    def logits_fn(generated, allowed):
        row = _logits({A: 1.0, Z: 0.0})
        return [row[t] for t in allowed]

    result = constrained_decode(logits_fn, trie, id_map)
    assert result.candidate == "a"
    # Positions: space (forced, lp=0), then a-vs-z, then END (forced).
    expected = math.log(math.exp(1.0) / (math.exp(1.0) + math.exp(0.0)))
    assert math.isclose(result.logprob, expected, rel_tol=1e-9)


def test_sampling_stays_inside_the_candidate_set():
    trie, id_map = build_relation_trie(["ab", "ac", "zz"], TOK, end_id=END)
    valid = {"ab", "ac", "zz", STOP_TOKEN}

    def logits_fn(generated, allowed):
        row = _logits({ord("x"): 50.0})  # huge invalid lure, never requested
        return [row[t] for t in allowed]

    rng = random.Random(7)
    seen = set()
    for _ in range(50):
        result = constrained_decode(logits_fn, trie, id_map, temperature=1.0, rng=rng)
        assert result.candidate in valid
        assert result.logprob <= 0.0
        seen.add(result.candidate)
    assert len(seen) > 1  # actually samples, not secretly greedy


def test_allowed_along_walks_the_trie():
    trie, _ = build_relation_trie(["ab", "ac"], TOK, end_id=END, include_stop=False)
    ids = [SP, A, B, END]
    allowed = allowed_along(trie, ids)
    assert allowed[0] == [SP]
    assert allowed[1] == [A]
    assert set(allowed[2]) == {B, C}
    assert allowed[3] == [END]
    with pytest.raises(KeyError):  # off-trie sequence must be rejected loudly
        allowed_along(trie, [SP, Z])


def test_empty_candidates_rejected():
    with pytest.raises(ValueError):
        build_relation_trie([], TOK, end_id=END, include_stop=False)
