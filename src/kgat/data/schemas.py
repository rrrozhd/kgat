"""Core data contracts for kgat.

These types are imported across the whole codebase — controllers, the traversal
engine, evaluation, governance, and training all speak in terms of them. Keep them
small, immutable where possible, and fully type-hinted.

Note on forward references: ``TraversalState.budget`` is a ``BudgetLedger``
(``kgat.traversal.budget``) and ``Trajectory.cost`` / ``AuditCertificate.cost`` are
``CostRecord``s (``kgat.eval.cost``). Those modules import *from here*, so importing
them at runtime would create a cycle. They are therefore imported only under
``TYPE_CHECKING`` and referenced as string annotations; dataclasses do not evaluate
annotations at runtime, so nothing breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kgat.eval.cost import CostRecord
    from kgat.traversal.budget import BudgetLedger

# Type aliases: a KG entity id (e.g. a Freebase MID like "m.06w2sn5") and a
# relation label (e.g. "people.person.sibling_s"). Both are plain strings.
Entity = str
Relation = str


@dataclass(frozen=True)
class Triple:
    """A directed KG edge ``head --relation--> tail``."""

    head: Entity
    relation: Relation
    tail: Entity


@dataclass(frozen=True)
class Question:
    """A KGQA question with its linked topic entities and gold answers."""

    qid: str
    text: str
    topic_entities: tuple[Entity, ...]  # linked question entities (given by dataset)
    gold_answers: tuple[Entity, ...]
    dataset: str  # "webqsp" | "cwq" | "metaqa" | "sample"


@dataclass(frozen=True)
class Path:
    """A partial or complete reasoning path from a topic entity.

    ``triples`` is the ordered chain of expansions. ``root`` anchors the starting
    topic entity so that an *unexpanded* path (no triples yet) still knows where it
    sits on the graph — this is what lets the engine seed the frontier directly from
    ``Question.topic_entities`` while keeping ``Path`` self-describing (the brief's
    ``TraversalState.frontier`` is ``list[Path]``).
    """

    triples: tuple[Triple, ...] = ()
    root: Entity | None = None

    @property
    def current_node(self) -> Entity:
        """Return the tail of the last triple, else the root anchor.

        Raises ``ValueError`` for a truly empty path (no triples *and* no root),
        matching the contract "tail of last triple, or raise if empty".
        """
        if self.triples:
            return self.triples[-1].tail
        if self.root is not None:
            return self.root
        raise ValueError("current_node is undefined for an empty path with no root anchor")

    @property
    def nodes(self) -> tuple[Entity, ...]:
        """All nodes visited, in order: ``(root/head, ..., final tail)``."""
        if not self.triples:
            return (self.root,) if self.root is not None else ()
        return (self.triples[0].head, *(t.tail for t in self.triples))

    @property
    def relations(self) -> tuple[Relation, ...]:
        """The relation labels along the path, in order."""
        return tuple(t.relation for t in self.triples)

    def __len__(self) -> int:
        """Number of hops (triples) in the path."""
        return len(self.triples)


class ActionType(Enum):
    EXPAND = "expand"
    STOP = "stop"


@dataclass(frozen=True)
class Action:
    """A controller decision: expand along a relation, or stop.

    Invariant (enforced in ``__post_init__``): ``EXPAND <=> relation is not None``.
    """

    type: ActionType
    relation: Relation | None = None  # required iff EXPAND
    score: float = 0.0  # controller confidence

    def __post_init__(self) -> None:
        if self.type is ActionType.EXPAND and self.relation is None:
            raise ValueError("EXPAND action requires a non-None relation")
        if self.type is ActionType.STOP and self.relation is not None:
            raise ValueError("STOP action must not carry a relation")

    @classmethod
    def expand(cls, relation: Relation, score: float = 0.0) -> Action:
        return cls(ActionType.EXPAND, relation=relation, score=score)

    @classmethod
    def stop(cls, score: float = 0.0) -> Action:
        return cls(ActionType.STOP, relation=None, score=score)


@dataclass
class TraversalState:
    """Mutable state threaded through the traversal loop."""

    question: Question
    frontier: list[Path]  # active partial paths (the beam)
    step: int  # hop index, 0-based
    budget: BudgetLedger

    @property
    def frontier_nodes(self) -> list[Entity]:
        """Current node of every path on the frontier (deduped, order-preserving)."""
        seen: dict[Entity, None] = {}
        for path in self.frontier:
            seen.setdefault(path.current_node, None)
        return list(seen)


@dataclass
class TrajectoryStep:
    """One controller decision, recorded for SFT / analysis."""

    state_repr: str  # serialized controller input (for SFT)
    candidates: tuple[Relation, ...]  # valid relations offered at this step
    action: Action
    # Frontier node set at decision time — lets training compute exact graph
    # signals (distance-to-gold shaping) without replaying the engine.
    frontier_nodes: tuple[Entity, ...] = ()


@dataclass
class Trajectory:
    """A full traversal for one question: the steps taken, the answer, the cost."""

    qid: str
    steps: list[TrajectoryStep]
    predicted_answers: tuple[Entity, ...]
    hit: bool
    cost: CostRecord
    final_frontier: tuple[Entity, ...] = ()  # frontier when traversal ended


@dataclass
class HopAudit:
    """Per-hop governance record."""

    step: int
    relation: Relation
    checks_passed: dict[str, bool]  # policy_name -> pass/fail
    confidence: float
    provenance: tuple[str, ...] = ()  # source ids for the edge, if any


@dataclass
class AuditCertificate:
    """The auditable certificate for a completed traversal."""

    qid: str
    hops: list[HopAudit]
    final_verdict: bool  # all mandatory policies satisfied
    cost: CostRecord


__all__ = [
    "Entity",
    "Relation",
    "Triple",
    "Question",
    "Path",
    "ActionType",
    "Action",
    "TraversalState",
    "TrajectoryStep",
    "Trajectory",
    "HopAudit",
    "AuditCertificate",
]
