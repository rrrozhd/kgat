"""The traversal engine — the controller-agnostic main loop.

Given a question, a ``KGStore`` scoped to that question, a controller, a synthesizer,
and (optionally) governance policies, the engine expands a beam of reasoning paths
hop by hop until the controller stops, the budget is exhausted, a mandatory policy
hard-blocks, or the frontier dead-ends. It returns the ``Trajectory`` (steps +
predicted answers + cost) and the ``AuditCertificate``.

Works end-to-end with the ``DummyController`` + ``DummySynthesizer`` — no model deps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from kgat.controller.base import TraversalController
from kgat.data.schemas import (
    ActionType,
    Path,
    Question,
    Trajectory,
    TrajectoryStep,
    TraversalState,
    Triple,
)
from kgat.eval.cost import CostRecord
from kgat.eval.metrics import hit as _hit
from kgat.governance.audit import AuditCertificate, AuditCertificateBuilder, run_hop_policies
from kgat.governance.policy import HopPolicy
from kgat.graph.store import KGStore
from kgat.synthesis.base import AnswerSynthesizer
from kgat.traversal.budget import BudgetCaps, BudgetLedger


@dataclass
class TraversalResult:
    """What the engine returns for one question."""

    trajectory: Trajectory
    certificate: AuditCertificate


# Frontier chains shown in the serialized state (prompt-size guard; the beam itself
# is not truncated). Mining, SFT, and inference all read prompts built from this, so
# the cap must stay identical across them — hence it lives here, not in the caller.
MAX_STATE_CHAINS = 8


def serialize_state(state: TraversalState) -> str:
    """Serialize a state to the controller-input string recorded in ``TrajectoryStep``.

    This is the SFT-facing representation; keep it stable and human-readable.
    """
    chains: list[str] = []
    for path in state.frontier:
        if not path.triples:
            chains.append(path.current_node if path.root is not None else "?")
            continue
        parts = [path.triples[0].head]
        parts.extend(f"-[{t.relation}]-> {t.tail}" for t in path.triples)
        chains.append(" ".join(parts))
    if len(chains) > MAX_STATE_CHAINS:
        hidden = len(chains) - MAX_STATE_CHAINS
        chains = chains[:MAX_STATE_CHAINS] + [f"... (+{hidden} more)"]
    frontier_repr = " | ".join(chains) if chains else "(empty)"
    return f"Q: {state.question.text} || step={state.step} || frontier: {frontier_repr}"


def _union_candidates(store: KGStore, nodes: list[str]) -> list[str]:
    """Deduped, order-preserving union of the relations leaving any frontier node."""
    seen: dict[str, None] = {}
    for node in nodes:
        for relation in store.relations_of(node):
            seen.setdefault(relation, None)
    return list(seen)


def _expand(store: KGStore, frontier: list[Path], relation: str, beam_size: int) -> list[Path]:
    """Extend every frontier path by one hop along ``relation`` (capped to beam_size)."""
    new_frontier: list[Path] = []
    for path in frontier:
        node = path.current_node
        for tail in store.neighbors(node, relation):
            new_frontier.append(Path(triples=(*path.triples, Triple(node, relation, tail))))
            if len(new_frontier) >= beam_size:
                return new_frontier
    return new_frontier


class TraversalEngine:
    """Runs the traversal loop for one question at a time."""

    def __init__(
        self,
        store: KGStore,
        controller: TraversalController,
        synthesizer: AnswerSynthesizer,
        *,
        policies: list[HopPolicy] | None = None,
        budget_caps: BudgetCaps | None = None,
        beam_size: int = 16,
        max_steps: int = 32,
    ) -> None:
        self.store = store
        self.controller = controller
        self.synthesizer = synthesizer
        self.policies = policies or []
        self.budget_caps = budget_caps or BudgetCaps()
        self.beam_size = beam_size
        # Absolute loop guard independent of the (possibly uncapped) budget.
        self.max_steps = max_steps

    def run(self, question: Question) -> TraversalResult:
        """Traverse the subgraph for ``question`` and return the result."""
        self.store.load_question_subgraph(question.qid)
        self.controller.bind_store(self.store)

        ledger = BudgetLedger(caps=self.budget_caps)
        frontier: list[Path] = [Path(root=e) for e in question.topic_entities]
        state = TraversalState(question=question, frontier=frontier, step=0, budget=ledger)
        audit = AuditCertificateBuilder(question.qid)
        steps: list[TrajectoryStep] = []

        start = time.perf_counter()
        while state.step < self.max_steps and not ledger.exhausted():
            if not state.frontier:
                break
            candidates = _union_candidates(self.store, state.frontier_nodes)
            if not candidates:
                break

            # One controller decision == one "LLM call" on the cost axis.
            action = self.controller.select(state, candidates)
            ledger.charge(llm_calls=1)

            checks_passed, hard_block = run_hop_policies(self.policies, state, action)
            audit.record_hop(
                step=state.step,
                action=action,
                checks_passed=checks_passed,
                confidence=action.score,
                provenance=(),
            )
            steps.append(
                TrajectoryStep(
                    state_repr=serialize_state(state),
                    candidates=tuple(candidates),
                    action=action,
                    frontier_nodes=tuple(state.frontier_nodes),
                )
            )

            if hard_block or action.type is ActionType.STOP:
                break

            # EXPAND: move the beam one hop along the chosen relation.
            assert action.relation is not None  # invariant guaranteed by Action
            new_frontier = _expand(self.store, state.frontier, action.relation, self.beam_size)
            if not new_frontier:  # dead end — keep the pre-expansion frontier for synthesis
                break
            state.frontier = new_frontier
            state.step += 1
            ledger.charge(hops=1)
            if ledger.exhausted():
                break

        ledger.charge(wall_ms=(time.perf_counter() - start) * 1000.0)

        predicted = self.synthesizer.synthesize(question, state.frontier)
        cost = CostRecord.from_ledger(ledger)
        hit = _hit(predicted, question.gold_answers)
        trajectory = Trajectory(
            qid=question.qid,
            steps=steps,
            predicted_answers=predicted,
            hit=hit,
            cost=cost,
            final_frontier=tuple(state.frontier_nodes),
        )
        certificate = audit.build(cost)
        return TraversalResult(trajectory=trajectory, certificate=certificate)


__all__ = ["TraversalEngine", "TraversalResult", "serialize_state"]
