"""Objective edge judging for the phase-2 reward — rules first, distilled model second.

The routing reward's precision term needs a per-edge quality score WITHOUT calling
an LLM inside the training loop. Two properties of this setting make that sound:
the policy never generates the text the judge reads (it only picks (relation,
target) against an environment-given chunk, so the adversarial-text reward-hacking
channel does not exist), and alphina's production critic already logged ~490k
verdicts (accepted edges carry ``raw_faithfulness``; ``extraction_rejections``
stores full rejected payloads) — a ready-made distillation set.

The judge is layered, mirroring what "a proper KG edge" means:

1. **Hard gates** (binary, objective, pure Python — this module):
   * ``grounded`` — the target name (or a known alias) literally appears in the
     chunk; an edge whose object is not in its own evidence is unwritable
     (alphina's ``verify_and_locate_evidence`` enforces the same at commit time).
   * ``target_is_company`` — the target must not be a known PERSON (board-bio /
     executive-history edges are the critic's rejection class #1). Injectable
     name set; exported from alphina's people tables when available.
   * ``filer_is_party`` — the chunk must speak in the filer's first person
     ("we/our/us/the Company"); bystander industry passages fail this. Coarse
     by construction — a chunk-level cue, not clause-level parsing.
   A gate failure zeroes the edge: gates check *existence*, not degree.

2. **Type faithfulness** (the part rules cannot decide — is the RELATION right,
   in the filer's direction?): a scoring callable in [0, 1]. Offline/rules-only
   runs anchor it to the distant teacher labels (novel edges score 0 —
   conservative: rules alone cannot certify edges the teacher never saw). The
   distilled cross-encoder (``kgat.train.judge``) replaces that anchor and OPENS
   the exceed-the-teacher channel: a grounded, party, company edge the teacher
   missed earns reward iff the distilled critic scores it faithful.

Everything here is pure Python and exhaustively testable; the model only enters
through the injected callable.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

from kgat.data.backfill_export import ExtractionPair

# A judge scores one emitted edge against its chunk: (pair, relation, target) -> [0, 1].
EdgeJudgeFn = Callable[[ExtractionPair, str, str], float]

# First-person cues that mark the filer as a party to the text. Chunk-level and
# deliberately generous — the gate exists to kill pure third-party passages
# (industry landscape, peer-group lists), not to do clause-level attribution.
_FILER_PARTY_RE = re.compile(
    r"\b(we|our|us|the company|the corporation|the registrant)\b", re.IGNORECASE
)

# Corporate suffixes ignored when grounding a target name in the chunk — filings
# routinely drop them on later mentions ("Intel Corporation ... compete with Intel").
_CORP_SUFFIXES = (
    "incorporated", "corporation", "company", "holdings", "limited", "technologies",
    "inc", "corp", "co", "ltd", "llc", "plc", "lp", "sa", "ag", "nv", "group",
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _strip_suffixes(name: str) -> str:
    words = _normalize(name).split()
    while len(words) > 1 and words[-1] in _CORP_SUFFIXES:
        words.pop()
    return " ".join(words)


def grounded(target: str, text: str, aliases: Mapping[str, Collection[str]] | None = None) -> bool:
    """Is the target name (or an alias, or its suffix-stripped core) in the chunk?"""
    hay = " " + _normalize(text) + " "
    candidates = [target, *((aliases or {}).get(target, ()))]
    for cand in candidates:
        core = _strip_suffixes(cand)
        if core and f" {core} " in hay:
            return True
        full = _normalize(cand)
        if full and f" {full} " in hay:
            return True
    return False


def filer_is_party(text: str) -> bool:
    """Does the chunk speak in the filer's first person at all?"""
    return bool(_FILER_PARTY_RE.search(text))


@dataclass(frozen=True)
class GateReport:
    """Which existence gates an emitted edge passed."""

    grounded: bool
    is_company: bool
    filer_party: bool

    @property
    def passed(self) -> bool:
        return self.grounded and self.is_company and self.filer_party


@dataclass(frozen=True)
class RuleGates:
    """Configured hard gates. ``known_people`` and ``aliases`` come from alphina
    exports; both default empty (the corresponding gate then never fires falsely)."""

    known_people: frozenset[str] = frozenset()
    aliases: Mapping[str, Collection[str]] = field(default_factory=dict)

    def evaluate(self, pair: ExtractionPair, target: str) -> GateReport:
        return GateReport(
            grounded=grounded(target, pair.text, self.aliases),
            is_company=_normalize(target) not in self.known_people,
            filer_party=filer_is_party(pair.text),
        )


def normalize_people(names: Collection[str]) -> frozenset[str]:
    """Normalize a people-name export for ``RuleGates.known_people``."""
    return frozenset(_normalize(n) for n in names if n and n.strip())


def make_rule_judge(
    gates: RuleGates | None = None,
    *,
    type_score: Callable[[ExtractionPair, str, str], float] | None = None,
) -> EdgeJudgeFn:
    """Compose the layered judge: hard gates x type faithfulness.

    ``type_score`` decides how faithful the (relation, target) claim is once the
    edge passes the existence gates. ``None`` selects the conservative distant
    anchor: 1.0 iff the edge is among the chunk's teacher labels, else 0.0 —
    imitation-safe, cannot exceed the teacher. Plug the distilled cross-encoder
    (``kgat.train.judge.load_judge_scorer``) here to lift that ceiling.
    """
    rules = gates or RuleGates()

    def judge(pair: ExtractionPair, relation: str, target: str) -> float:
        if not rules.evaluate(pair, target).passed:
            return 0.0
        if type_score is None:
            return 1.0 if (relation, target) in set(pair.triples) else 0.0
        return max(0.0, min(1.0, float(type_score(pair, relation, target))))

    return judge


__all__ = [
    "EdgeJudgeFn",
    "grounded",
    "filer_is_party",
    "GateReport",
    "RuleGates",
    "normalize_people",
    "make_rule_judge",
]
