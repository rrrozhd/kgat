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
        lp = _logsoftmax(vals)

        if len(allowed) == 1:
            idx = 0
        elif temperature <= 0.0:
            idx = max(range(len(allowed)), key=lambda i: lp[i])
        else:
            scaled = _logsoftmax([v / temperature for v in vals])
            r = (rng or random).random()
            acc, idx = 0.0, len(allowed) - 1
            for i in range(len(allowed)):
                acc += math.exp(scaled[i])
                if r <= acc:
                    idx = i
                    break

        total_logprob += lp[idx]
        choice = allowed[idx]
        generated = (*generated, choice)
        node = node.step(choice)

    return DecodeResult(ids=generated, candidate=id_map[generated], logprob=total_logprob)


__all__ = [
    "STOP_TOKEN",
    "target_text",
    "TokenTrie",
    "build_relation_trie",
    "allowed_along",
    "DecodeResult",
    "constrained_decode",
]
