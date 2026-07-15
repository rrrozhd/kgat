"""Valid-relation constrained decoding over a token trie.

The controller's generation is restricted to exactly the candidate relation strings
(plus the ``STOP_TOKEN`` sentinel): a prefix trie over the candidates' token ids
masks the model's logits at every step, so a sub-1B model can never emit an invalid
or hallucinated relation. This constraint is load-bearing for small-model viability.

Design: this module is pure Python (no torch). The model is abstracted as a
``logits_fn(generated_ids) -> full-vocab logits`` callable, which keeps the trie and
the decode loop unit-testable with a fake tokenizer/model. Every candidate sequence
is terminated with an ``end_id`` token when the trie is built, which makes the
sequence set prefix-free (a candidate that is a prefix of another still gets its own
leaf), so decoding always ends at a unique candidate.

Log-probabilities are computed under the mask-renormalized temperature-1 model
distribution (softmax over the *allowed* tokens only). Sampling may use a different
temperature; recorded logprobs stay at temperature 1 so the GRPO gradient pass
(``kgat.train.grpo``) scores the same distribution it optimizes.

Besides the single-candidate relation trie (the read path), this module hosts the
write-path **triple grammar** (``build_triple_grammar`` / ``decode_triples``): the
backfill extractor's ``NONE | <relation> :: <target> ( ; ...)*`` output language,
built from the same ``TokenTrie`` pieces and equally torch-free.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

STOP_TOKEN = "[STOP]"


def target_text(candidate: str) -> str:
    """The exact text generated for a candidate (leading space for BPE boundary).

    Used by trie construction, SFT target building, and the GRPO gradient pass —
    all three MUST agree on this transformation.
    """
    return " " + candidate


@dataclass
class TokenTrie:
    """Prefix trie over token-id sequences."""

    children: dict[int, TokenTrie] = field(default_factory=dict)

    def insert(self, ids: Sequence[int]) -> None:
        node = self
        for tok in ids:
            node = node.children.setdefault(tok, TokenTrie())

    def allowed(self) -> list[int]:
        """Token ids permitted as the next step from this node."""
        return list(self.children)

    def step(self, tok: int) -> TokenTrie:
        return self.children[tok]

    @property
    def is_leaf(self) -> bool:
        return not self.children


def build_relation_trie(
    candidates: Sequence[str],
    tokenizer,
    *,
    end_id: int,
    include_stop: bool = True,
) -> tuple[TokenTrie, dict[tuple[int, ...], str]]:
    """Build the constraint trie for a candidate set.

    ``tokenizer`` needs ``encode(text, add_special_tokens=False) -> list[int]``.
    Each candidate is encoded as ``target_text(candidate) + [end_id]``. Returns the
    trie and a map from complete id-paths back to the candidate string.

    Raises ``ValueError`` on an empty candidate set or on two candidates that encode
    to the same token sequence (undecidable).
    """
    cands = list(candidates)
    if include_stop and STOP_TOKEN not in cands:
        cands.append(STOP_TOKEN)
    if not cands:
        raise ValueError("cannot build a constraint trie over zero candidates")

    trie = TokenTrie()
    id_map: dict[tuple[int, ...], str] = {}
    for cand in cands:
        ids = tuple(tokenizer.encode(target_text(cand), add_special_tokens=False)) + (end_id,)
        if ids in id_map and id_map[ids] != cand:
            raise ValueError(f"candidates {id_map[ids]!r} and {cand!r} tokenize identically")
        id_map[ids] = cand
        trie.insert(ids)
    return trie, id_map


def allowed_along(trie: TokenTrie, ids: Sequence[int]) -> list[list[int]]:
    """The allowed-token set at every position while consuming ``ids``.

    ``result[j]`` is the allowed set *before* consuming ``ids[j]``. Raises
    ``KeyError`` if ``ids`` leaves the trie — the caller passed a sequence that the
    constraint could never have produced. Used by the GRPO gradient pass to
    renormalize logits exactly as the rollout did.
    """
    out: list[list[int]] = []
    node = trie
    for tok in ids:
        out.append(node.allowed())
        node = node.step(tok)  # KeyError => sequence not in trie
    return out


@dataclass(frozen=True)
class DecodeResult:
    """Outcome of one constrained decode."""

    ids: tuple[int, ...]  # full path incl. the end_id
    candidate: str  # the decoded candidate string (or STOP_TOKEN)
    logprob: float  # sum of temp-1 mask-renormalized token logprobs


def _logsoftmax(vals: Sequence[float]) -> list[float]:
    """Log-softmax of a small score vector (temp 1)."""
    m = max(vals)
    log_z = m + math.log(sum(math.exp(v - m) for v in vals))
    return [v - log_z for v in vals]


def _choose(
    vals: Sequence[float], temperature: float, rng: random.Random | None
) -> tuple[int, float]:
    """Pick one index from allowed-token scores; return (index, temp-1 logprob).

    Selection is greedy at ``temperature<=0`` and samples from the temperature-scaled
    distribution otherwise; the returned logprob is always the temperature-1
    mask-renormalized value (see module docstring). A single-entry vector is a
    forced step: index 0, logprob 0.
    """
    lp = _logsoftmax(vals)
    if len(vals) == 1:
        return 0, lp[0]
    if temperature <= 0.0:
        idx = max(range(len(vals)), key=lambda i: lp[i])
        return idx, lp[idx]
    scaled = _logsoftmax([v / temperature for v in vals])
    r = (rng or random).random()
    acc, idx = 0.0, len(vals) - 1
    for i in range(len(vals)):
        acc += math.exp(scaled[i])
        if r <= acc:
            idx = i
            break
    return idx, lp[idx]


def constrained_decode(
    logits_fn: Callable[[tuple[int, ...], list[int]], Sequence[float]],
    trie: TokenTrie,
    id_map: dict[tuple[int, ...], str],
    *,
    temperature: float = 0.0,
    rng: random.Random | None = None,
) -> DecodeResult:
    """Decode exactly one candidate under the trie constraint.

    ``logits_fn(generated_ids, allowed_ids)`` returns the scores for exactly the
    ``allowed_ids`` continuations (aligned, ``len == len(allowed_ids)``). Passing
    the allowed set INTO the scorer is the load-bearing structured-decoding
    contract: a GPU backend gathers just those entries on-device instead of
    materializing the full vocabulary per token (with a ~151k vocab that transfer
    plus the Python list conversion dominated per-token latency at batch 1).

    ``temperature=0`` decodes greedily; ``temperature>0`` samples from the
    temperature-scaled masked distribution (``rng`` for determinism). The returned
    ``logprob`` is always the temperature-1 masked logprob of the chosen path.
    """
    node = trie
    generated: tuple[int, ...] = ()
    total_logprob = 0.0

    while not node.is_leaf:
        allowed = node.allowed()
        vals = list(logits_fn(generated, allowed))
        idx, lp = _choose(vals, temperature, rng)
        total_logprob += lp
        choice = allowed[idx]
        generated = (*generated, choice)
        node = node.step(choice)

    return DecodeResult(ids=generated, candidate=id_map[generated], logprob=total_logprob)


# ---------------------------------------------------------------------------
# Triple grammar — the write-path constraint (DESIGN-BACKFILL.md).
#
# The extractor emits either `` NONE`` or one-or-more `` <relation> :: <target>``
# items joined by `` ;``, then EOS:
#
#     `` NONE``<eos>
#     `` supplier :: Acme Corp ; customer :: Bolt Inc``<eos>
#
# Relations are constrained to the taxonomy (alphina's seven filer-centric types),
# targets to a closed entity vocabulary (the entity-resolver universe stands in for
# free-form names in the pilot) — emitting outside the schema is physically
# impossible. The grammar is a chain of segment tries; the "continue vs stop"
# decision is folded into the target segment (a target ends with either the `` ;``
# separator or EOS), which also keeps every segment prefix-free without artificial
# terminator tokens. The canonical token ids for a given triple list are defined BY
# the grammar (segment-wise encoding, NOT one flat ``tokenizer.encode`` of the
# joined string — BPE may merge across segment boundaries); SFT targets must come
# from ``encode_triples_target`` so training and constrained inference agree.
# ---------------------------------------------------------------------------

NONE_LABEL = "NONE"
REL_TARGET_SEP = " ::"  # between a relation and its target
TRIPLE_SEP = " ;"  # between consecutive triples


@dataclass(frozen=True)
class _Segment:
    """One grammar segment: a prefix trie plus id-path -> value map."""

    trie: TokenTrie
    id_map: dict[tuple[int, ...], object]


def _build_segment(entries: dict[tuple[int, ...], object]) -> _Segment:
    """Build a segment trie, enforcing token-level prefix-freeness of its entries.

    A proper-prefix entry would make decoding ambiguous (no end_id disambiguation
    here — segment boundaries are real text). Token-prefix implies string-prefix
    (ids concatenate to exact strings), so distinct non-prefix names are safe.
    """
    if not entries:
        raise ValueError("cannot build a grammar segment over zero entries")
    trie = TokenTrie()
    for ids in entries:
        if not ids:
            raise ValueError("grammar segment entry encoded to zero tokens")
        trie.insert(ids)
    for ids in entries:
        node = trie
        for tok in ids:
            node = node.step(tok)
        if not node.is_leaf:
            raise ValueError(
                f"grammar segment entry {ids} is a proper token-prefix of another entry"
            )
    return _Segment(trie=trie, id_map=dict(entries))


@dataclass(frozen=True)
class TripleGrammar:
    """Compiled triple grammar over a relation taxonomy and a target vocabulary."""

    relations: tuple[str, ...]
    targets: tuple[str, ...]
    eos_id: int
    max_triples: int
    first: _Segment  # NONE | sentinel | relation (opens the output)
    rel: _Segment  # relation only (after a `` ;`` continuation)
    target: _Segment  # target + (continue | end)
    target_last: _Segment  # target + end only (at the max_triples cap)
    enc_none: tuple[int, ...]  # `` NONE``<eos>
    enc_rel: dict[str, tuple[int, ...]]  # `` <relation> ::``
    enc_target: dict[str, tuple[int, ...]]  # `` <target>`` (bare)
    enc_sep: tuple[int, ...]  # `` ;``
    sentinels: tuple[str, ...] = ()  # extra whole-output terminals (e.g. ESCALATE)
    enc_sentinel: dict[str, tuple[int, ...]] = field(default_factory=dict)


def build_triple_grammar(
    relations: Sequence[str],
    targets: Sequence[str],
    tokenizer,
    *,
    eos_id: int,
    max_triples: int = 8,
    sentinels: Sequence[str] = (),
) -> TripleGrammar:
    """Compile the extraction grammar for a relation taxonomy + target vocabulary.

    ``tokenizer`` needs ``encode(text, add_special_tokens=False) -> list[int]``.
    ``sentinels`` are extra NONE-like terminals available ONLY as the whole output
    (e.g. ``ESCALATE`` for the phase-2 routing policy: the model either routes or
    extracts, in one constrained decode). Raises ``ValueError`` on empty inputs,
    names that collide with the grammar's separators/sentinels, or entries that
    are token-prefixes of each other.
    """
    rels = list(dict.fromkeys(relations))
    tgts = list(dict.fromkeys(targets))
    sents = list(dict.fromkeys(sentinels))
    if not rels or not tgts:
        raise ValueError("triple grammar needs at least one relation and one target")
    if max_triples < 1:
        raise ValueError("max_triples must be >= 1")
    reserved = (NONE_LABEL, REL_TARGET_SEP.strip(), TRIPLE_SEP.strip(), *sents)
    for name in (*rels, *tgts):
        if not name or not name.strip():
            raise ValueError("empty grammar candidate name")
        if name in (NONE_LABEL, *sents) or any(sep in name for sep in reserved[1:3]):
            raise ValueError(f"candidate {name!r} collides with a reserved grammar token")

    def enc(text: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(text, add_special_tokens=False))

    enc_none = enc(target_text(NONE_LABEL)) + (eos_id,)
    enc_sentinel = {s: enc(target_text(s)) + (eos_id,) for s in sents}
    enc_rel = {r: enc(target_text(r)) + enc(REL_TARGET_SEP) for r in rels}
    enc_target = {t: enc(target_text(t)) for t in tgts}
    enc_sep = enc(TRIPLE_SEP)

    rel_entries: dict[tuple[int, ...], object] = {}
    for r, ids in enc_rel.items():
        if ids in rel_entries:
            raise ValueError(f"relations {rel_entries[ids]!r} and {r!r} tokenize identically")
        rel_entries[ids] = r

    end_entries: dict[tuple[int, ...], object] = {}
    cont_entries: dict[tuple[int, ...], object] = {}
    for t, ids in enc_target.items():
        if ids + (eos_id,) in end_entries:
            other = end_entries[ids + (eos_id,)][0]  # type: ignore[index]
            raise ValueError(f"targets {other!r} and {t!r} tokenize identically")
        end_entries[ids + (eos_id,)] = (t, False)
        cont_entries[ids + enc_sep] = (t, True)

    sentinel_entries: dict[tuple[int, ...], object] = {
        ids: s for s, ids in enc_sentinel.items()
    }
    return TripleGrammar(
        relations=tuple(rels),
        targets=tuple(tgts),
        eos_id=eos_id,
        max_triples=max_triples,
        first=_build_segment({enc_none: NONE_LABEL, **sentinel_entries, **rel_entries}),
        rel=_build_segment(rel_entries),
        target=_build_segment({**end_entries, **cont_entries}),
        target_last=_build_segment(end_entries),
        enc_none=enc_none,
        enc_rel=enc_rel,
        enc_target=enc_target,
        enc_sep=enc_sep,
        sentinels=tuple(sents),
        enc_sentinel=enc_sentinel,
    )


@dataclass(frozen=True)
class TripleDecodeResult:
    """Outcome of one constrained triple extraction."""

    ids: tuple[int, ...]  # full emitted path incl. the final eos
    triples: tuple[tuple[str, str], ...]  # (relation, target); empty == NONE/sentinel
    logprob: float  # sum of temp-1 mask-renormalized step logprobs
    n_choices: int  # steps with >1 allowed token (forced steps excluded)
    sentinel: str | None = None  # which extra terminal was emitted (None for NONE/triples)

    @property
    def confidence(self) -> float:
        """``exp(mean logprob)`` over real decisions — the cascade's escalation signal.

        Forced steps carry logprob 0 and are excluded from the mean so the score
        stays comparable across outputs of different token lengths.
        """
        if self.n_choices == 0:
            return 1.0
        return math.exp(self.logprob / self.n_choices)


def decode_triples(
    logits_fn: Callable[[tuple[int, ...], list[int]], Sequence[float]],
    grammar: TripleGrammar,
    *,
    temperature: float = 0.0,
    rng: random.Random | None = None,
) -> TripleDecodeResult:
    """Decode one NONE-or-triples output under the grammar constraint.

    ``logits_fn`` has the same contract as :func:`constrained_decode` — it receives
    ALL ids generated so far (across segments) plus the allowed continuations, and
    returns scores for exactly those continuations.
    """
    generated: list[int] = []
    total_logprob = 0.0
    n_choices = 0

    def run_segment(segment: _Segment) -> object:
        nonlocal total_logprob, n_choices
        node = segment.trie
        path: list[int] = []
        while not node.is_leaf:
            allowed = node.allowed()
            vals = list(logits_fn(tuple(generated), allowed))
            idx, lp = _choose(vals, temperature, rng)
            if len(allowed) > 1:
                n_choices += 1
            total_logprob += lp
            tok = allowed[idx]
            generated.append(tok)
            path.append(tok)
            node = node.step(tok)
        return segment.id_map[tuple(path)]

    triples: list[tuple[str, str]] = []
    sentinel: str | None = None
    first = run_segment(grammar.first)
    if first in grammar.sentinels:
        sentinel = str(first)
    elif first != NONE_LABEL:
        relation = str(first)
        while True:
            at_cap = len(triples) + 1 >= grammar.max_triples
            segment = grammar.target_last if at_cap else grammar.target
            target, more = run_segment(segment)  # type: ignore[misc]
            triples.append((relation, str(target)))
            if not more:
                break
            relation = str(run_segment(grammar.rel))

    return TripleDecodeResult(
        ids=tuple(generated),
        triples=tuple(triples),
        logprob=total_logprob,
        n_choices=n_choices,
        sentinel=sentinel,
    )


def encode_triples_target(
    triples: Sequence[tuple[str, str]], grammar: TripleGrammar
) -> list[int]:
    """Canonical token ids for a gold triple list — the SFT target.

    Produces exactly the id sequence :func:`decode_triples` emits for the same
    triples, including the final eos. Raises ``ValueError`` for relations/targets
    outside the grammar or more than ``max_triples`` items (such an example could
    never be decoded, so training on it would be silent label noise).
    """
    items = list(triples)
    if not items:
        return list(grammar.enc_none)
    if len(items) > grammar.max_triples:
        raise ValueError(f"{len(items)} triples exceeds grammar max_triples={grammar.max_triples}")
    ids: list[int] = []
    for i, (relation, target) in enumerate(items):
        if relation not in grammar.enc_rel:
            raise ValueError(f"relation {relation!r} not in grammar taxonomy {grammar.relations}")
        if target not in grammar.enc_target:
            raise ValueError(f"target {target!r} not in the grammar's target vocabulary")
        ids.extend(grammar.enc_rel[relation])
        ids.extend(grammar.enc_target[target])
        ids.extend(grammar.enc_sep if i < len(items) - 1 else (grammar.eos_id,))
    return ids


__all__ = [
    "STOP_TOKEN",
    "NONE_LABEL",
    "REL_TARGET_SEP",
    "TRIPLE_SEP",
    "target_text",
    "TokenTrie",
    "build_relation_trie",
    "allowed_along",
    "DecodeResult",
    "constrained_decode",
    "TripleGrammar",
    "build_triple_grammar",
    "TripleDecodeResult",
    "decode_triples",
    "encode_triples_target",
]
