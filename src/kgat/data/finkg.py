"""FinKG — financial KGQA training data generator (alphina-schema).

Generates multi-hop KGQA records in the project's standard on-disk schema
(``id/question/q_entity/a_entity/graph``) over a financial knowledge graph whose
relation vocabulary mirrors alphina's (the ``market_analysis`` platform): the seven
filer-centric relationship types from its SEC-filing extraction taxonomy, stored as
typed directed edge pairs exactly like its Neo4j sync
(``SUPPLIER_TO``/``CUSTOMER_OF``, symmetric ``COMPETITOR_OF``, ...), plus sector,
officer, and Form-4 insider-purchase edges.

Because answers are computed from the graph by construction, no human labeling is
needed — sample a relation path, verbalize it from per-relation noun phrases, and
the gold set is every entity reachable by that exact relation sequence (the MetaQA
construction). Depth mix is a knob, which is exactly what budget-adaptive stopping
needs from its training distribution.

Two triple sources, same generator:

* ``build_synthetic_kg`` — a deterministic, seeded synthetic financial KG for
  offline development (this module's default).
* ``load_triples_jsonl`` — an export of a real graph (e.g. alphina's; see
  docs/DESIGN-FINKG.md for the export contract), one
  ``{"head", "relation", "tail"}`` object per line.

CLI::

    python -m kgat.data.finkg --out data/finkg --companies 120 --questions 600 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from kgat.data.schemas import Entity, Relation, Triple

# ---------------------------------------------------------------------------
# Relation vocabulary — mirrors alphina's taxonomy + graph sync edge labels.
# Directed labels; asymmetric types are emitted as forward/reverse pairs the way
# alphina's Neo4j sync does. Traversal-facing semantics:
#   X -customer_of-> S   : S supplies X   ("suppliers of X" = follow customer_of)
#   X -supplier_to-> C   : X supplies C   ("customers of X" = follow supplier_to)
# ---------------------------------------------------------------------------
REVERSE_EDGE: dict[Relation, Relation] = {
    "supplier_to": "customer_of",
    "customer_of": "supplier_to",
    "competitor_of": "competitor_of",  # symmetric
    "partner_of": "partner_of",  # symmetric
    "acquired": "acquired_by",
    "acquired_by": "acquired",
    "subsidiary_of": "has_subsidiary",
    "has_subsidiary": "subsidiary_of",
    "invested_in": "has_investor",
    "has_investor": "invested_in",
    "in_sector": "sector_contains",
    "sector_contains": "in_sector",
    # People edges — alphina's people_sync projects (:Person)-[:OFFICER_OF|DIRECTOR_OF]->
    # (:Company) with reverses HAS_OFFICER|HAS_DIRECTOR. Directors on multiple boards
    # create the interlock paths alphina's people-centrality PageRank scores.
    "officer_of": "has_officer",
    "has_officer": "officer_of",
    "director_of": "has_director",
    "has_director": "director_of",
    "insider_bought": "bought_by_insider",
    "bought_by_insider": "insider_bought",
}

# Noun phrase per directed relation, composed right-to-left into questions:
# seq [competitor_of, customer_of] -> "suppliers of competitors of {X}".
_NOUN: dict[Relation, str] = {
    "supplier_to": "customers",
    "customer_of": "suppliers",
    "competitor_of": "competitors",
    "partner_of": "partners",
    "acquired": "acquisitions",
    "acquired_by": "acquirers",
    "subsidiary_of": "parent companies",
    "has_subsidiary": "subsidiaries",
    "invested_in": "portfolio companies",
    "has_investor": "investors",
    "in_sector": "sectors",
    "sector_contains": "companies in the sector",
    "officer_of": "companies led by",
    "has_officer": "executives",
    "director_of": "companies with board seats held by",
    "has_director": "board members",
    "insider_bought": "companies with insider purchases by",
    "bought_by_insider": "insider buyers",
}

# Relations a question hop may traverse (all directed labels above).
_HOP_RELATIONS: tuple[Relation, ...] = tuple(_NOUN)

_SECTORS = (
    "semiconductors",
    "software",
    "fintech",
    "biotech",
    "energy",
    "industrials",
    "consumer",
    "telecom",
)

_NAME_A = ("Vex", "Nor", "Alt", "Quen", "Bry", "Cor", "Del", "Fen", "Gal", "Hel")
_NAME_B = ("tra", "dian", "mont", "lex", "vio", "dara", "quin", "sor", "beck", "lune")
_NAME_C = ("Systems", "Labs", "Holdings", "Dynamics", "Industries", "Capital", "Group", "Tech")


@dataclass
class FinKG:
    """A directed financial KG plus entity-type metadata."""

    triples: list[Triple] = field(default_factory=list)
    companies: list[Entity] = field(default_factory=list)
    people: list[Entity] = field(default_factory=list)
    sectors: list[Entity] = field(default_factory=list)

    def add_pair(self, head: Entity, relation: Relation, tail: Entity) -> None:
        """Add a directed edge and its typed reverse (alphina's Neo4j convention)."""
        self.triples.append(Triple(head, relation, tail))
        self.triples.append(Triple(tail, REVERSE_EDGE[relation], head))


def _company_name(rng: random.Random, taken: set[str]) -> str:
    while True:
        name = rng.choice(_NAME_A) + rng.choice(_NAME_B) + " " + rng.choice(_NAME_C)
        if name not in taken:
            taken.add(name)
            return name


def build_synthetic_kg(n_companies: int = 120, seed: int = 42, edges_per_company: int = 5) -> FinKG:
    """Deterministic synthetic financial KG in alphina's relation vocabulary.

    Structure sketch: every company gets a sector; company-company edges are sampled
    over the seven filer-centric types; people hold 1-3 officer/director roles at
    DIFFERENT companies (mirroring alphina's Person->OFFICER_OF|DIRECTOR_OF model),
    so interlocking-directorate paths exist — "board members of X" -> "their other
    boards" is a real 2-hop, the pattern alphina's people-centrality scores. A
    subset of role-holders have Form-4-style insider purchases at their companies.
    Names are synthetic — this KG is for pipeline development, not facts.
    """
    rng = random.Random(seed)
    kg = FinKG()
    taken: set[str] = set()

    kg.sectors = [f"sector:{s}" for s in _SECTORS]
    kg.companies = [_company_name(rng, taken) for _ in range(n_companies)]
    for company in kg.companies:
        kg.add_pair(company, "in_sector", rng.choice(kg.sectors))

    n_people = max(2, int(n_companies * 1.2))
    for i in range(n_people):
        person = f"{rng.choice(('A.', 'J.', 'M.', 'R.', 'S.'))} {rng.choice(_NAME_B).title()}{i}"
        kg.people.append(person)
        n_roles = 1 + rng.randrange(3)  # 1-3 roles; >1 creates board interlocks
        seats = rng.sample(kg.companies, min(n_roles, len(kg.companies)))
        for j, company in enumerate(seats):
            role = "officer_of" if j == 0 and rng.random() < 0.6 else "director_of"
            kg.add_pair(person, role, company)
            if rng.random() < 0.25:  # Form-4-style open-market purchase
                kg.add_pair(person, "insider_bought", company)

    company_edge_types = (
        "supplier_to",
        "customer_of",
        "competitor_of",
        "partner_of",
        "acquired",
        "subsidiary_of",
        "invested_in",
    )
    seen_pairs: set[tuple[str, str, str]] = set()
    for company in kg.companies:
        for _ in range(edges_per_company):
            other = rng.choice(kg.companies)
            if other == company:
                continue
            relation = rng.choice(company_edge_types)
            key = (company, relation, other)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            kg.add_pair(company, relation, other)
    return kg


def load_triples_jsonl(path: str | Path) -> FinKG:
    """Load an exported real graph (one ``{"head","relation","tail"}`` per line).

    Relations must use the directed labels in ``REVERSE_EDGE`` (lowercased alphina
    edge labels); reverse edges are added automatically when absent. Entity typing
    is inferred from relation usage (sector/officer/insider edges).
    """
    kg = FinKG()
    present: set[tuple[str, str, str]] = set()
    rows: list[Triple] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            relation = str(raw["relation"]).lower()
            if relation not in REVERSE_EDGE:
                raise ValueError(f"unknown relation {relation!r}; known: {sorted(REVERSE_EDGE)}")
            t = Triple(str(raw["head"]), relation, str(raw["tail"]))
            rows.append(t)
            present.add((t.head, t.relation, t.tail))

    companies: set[str] = set()
    people: set[str] = set()
    sectors: set[str] = set()
    for t in rows:
        kg.triples.append(t)
        rev = (t.tail, REVERSE_EDGE[t.relation], t.head)
        if rev not in present:
            kg.triples.append(Triple(*rev))
            present.add(rev)
        if t.relation == "in_sector":
            companies.add(t.head), sectors.add(t.tail)
        elif t.relation == "sector_contains":
            sectors.add(t.head), companies.add(t.tail)
        elif t.relation in ("officer_of", "director_of", "insider_bought"):
            people.add(t.head), companies.add(t.tail)
        elif t.relation in ("has_officer", "has_director", "bought_by_insider"):
            companies.add(t.head), people.add(t.tail)
        else:
            companies.add(t.head), companies.add(t.tail)
    kg.companies, kg.people, kg.sectors = sorted(companies), sorted(people), sorted(sectors)
    return kg


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------


def _adjacency(triples: list[Triple]) -> dict[Entity, dict[Relation, list[Entity]]]:
    adj: dict[Entity, dict[Relation, list[Entity]]] = {}
    for t in triples:
        tails = adj.setdefault(t.head, {}).setdefault(t.relation, [])
        if t.tail not in tails:
            tails.append(t.tail)
    return adj


def follow_sequence(
    adj: dict[Entity, dict[Relation, list[Entity]]],
    topic: Entity,
    sequence: tuple[Relation, ...],
) -> set[Entity]:
    """All entities reachable from ``topic`` via exactly ``sequence`` — the gold set."""
    frontier = {topic}
    for relation in sequence:
        nxt: set[Entity] = set()
        for node in frontier:
            nxt.update(adj.get(node, {}).get(relation, ()))
        frontier = nxt
        if not frontier:
            return set()
    return frontier


def verbalize(topic: Entity, sequence: tuple[Relation, ...]) -> str:
    """Compose the question right-to-left: [r1, r2] -> 'noun(r2) of noun(r1) of X'.

    Nouns already ending in a preposition ("companies led by", "... purchases by")
    join without the extra "of".
    """
    topic_name = topic.split(":", 1)[-1]
    phrase = topic_name
    for relation in sequence:
        noun = _NOUN[relation]
        joiner = " " if noun.endswith(" by") else " of "
        phrase = f"{noun}{joiner}{phrase}"
    return f"who are the {phrase}?"


def _question_subgraph(
    adj: dict[Entity, dict[Relation, list[Entity]]],
    topic: Entity,
    depth: int,
    rng: random.Random,
    max_edges: int = 200,
) -> list[Triple]:
    """BFS neighborhood of ``topic`` to depth+1 (deduped, deterministic, capped)."""
    edges: list[Triple] = []
    seen_edges: set[tuple[str, str, str]] = set()
    frontier = [topic]
    visited = {topic}
    for _ in range(depth + 1):
        nxt: list[Entity] = []
        for node in frontier:
            for relation, tails in adj.get(node, {}).items():
                for tail in tails:
                    key = (node, relation, tail)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    edges.append(Triple(node, relation, tail))
                    if tail not in visited:
                        visited.add(tail)
                        nxt.append(tail)
        frontier = nxt
    if len(edges) > max_edges:
        # Keep deterministic: rng is seeded per question.
        edges = rng.sample(edges, max_edges)
    return edges


@dataclass(frozen=True)
class GeneratedQuestion:
    qid: str
    question: str
    topic: Entity
    sequence: tuple[Relation, ...]
    golds: tuple[Entity, ...]
    graph: tuple[Triple, ...]

    def to_record(self) -> dict:
        return {
            "id": self.qid,
            "question": self.question,
            "q_entity": [self.topic],
            "a_entity": sorted(self.golds),
            "graph": [[t.head, t.relation, t.tail] for t in self.graph],
            "path": list(self.sequence),  # extra field: the generating sequence
        }


def generate_questions(
    kg: FinKG,
    n_questions: int,
    *,
    seed: int = 42,
    depth_mix: dict[int, float] | None = None,
    max_golds: int = 12,
) -> list[GeneratedQuestion]:
    """Sample relation paths and turn them into QA records.

    ``depth_mix`` maps hop count -> probability (default 45% 1-hop, 35% 2-hop,
    20% 3-hop — deeper than WebQSP on purpose, to exercise adaptive stopping).
    Questions with empty or oversized gold sets are resampled; oversized golds
    (> ``max_golds``) usually mean a degenerate hub path that verbalizes badly.
    """
    depth_mix = depth_mix or {1: 0.45, 2: 0.35, 3: 0.20}
    rng = random.Random(seed)
    adj = _adjacency(kg.triples)
    anchors = kg.companies + kg.people
    depths, weights = zip(*sorted(depth_mix.items()), strict=True)

    out: list[GeneratedQuestion] = []
    seen_questions: set[str] = set()
    attempts = 0
    while len(out) < n_questions and attempts < n_questions * 60:
        attempts += 1
        depth = rng.choices(depths, weights=weights)[0]
        topic = rng.choice(anchors)

        # Random walk to pick a plausible relation sequence (each hop must exist).
        sequence: list[Relation] = []
        node = topic
        for _ in range(depth):
            options = [r for r in adj.get(node, {}) if r in _HOP_RELATIONS]
            if not options:
                break
            relation = rng.choice(options)
            sequence.append(relation)
            node = rng.choice(adj[node][relation])
        if len(sequence) != depth:
            continue

        golds = follow_sequence(adj, topic, tuple(sequence))
        if not golds or len(golds) > max_golds or topic in golds:
            continue
        question = verbalize(topic, tuple(sequence))
        if question in seen_questions:
            continue
        seen_questions.add(question)

        qrng = random.Random(seed * 100_003 + len(out))
        graph = _question_subgraph(adj, topic, depth, qrng)
        # Guarantee the reasoning paths survive any subgraph cap.
        graph_keys = {(t.head, t.relation, t.tail) for t in graph}
        frontier = {topic}
        for relation in sequence:
            nxt: set[Entity] = set()
            for n in frontier:
                for tail in adj.get(n, {}).get(relation, ()):
                    if (n, relation, tail) not in graph_keys:
                        graph.append(Triple(n, relation, tail))
                        graph_keys.add((n, relation, tail))
                    nxt.add(tail)
            frontier = nxt

        out.append(
            GeneratedQuestion(
                qid=f"finkg-{len(out)}",
                question=question,
                topic=topic,
                sequence=tuple(sequence),
                golds=tuple(sorted(golds)),
                graph=tuple(graph),
            )
        )
    if len(out) < n_questions:
        raise RuntimeError(
            f"generated only {len(out)}/{n_questions} questions — KG too small/sparse "
            f"(raise n_companies or edges_per_company)"
        )
    return out


def write_splits(
    questions: list[GeneratedQuestion],
    out_dir: str | Path,
    *,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, int]:
    """Write train/dev/test JSONL splits (question-disjoint, same KG — MetaQA-style)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(questions)
    n_train = int(n * fractions[0])
    n_dev = int(n * fractions[1])
    splits = {
        "train": questions[:n_train],
        "dev": questions[n_train : n_train + n_dev],
        "test": questions[n_train + n_dev :],
    }
    counts: dict[str, int] = {}
    for split, items in splits.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for q in items:
                fh.write(json.dumps(q.to_record(), ensure_ascii=False) + "\n")
        counts[split] = len(items)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FinKG (alphina-schema) KGQA data.")
    parser.add_argument("--out", default="data/finkg", help="output dir for split JSONLs")
    parser.add_argument("--companies", type=int, default=120)
    parser.add_argument("--edges-per-company", type=int, default=5)
    parser.add_argument("--questions", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--triples",
        default=None,
        help="optional real-graph export (triples JSONL) instead of the synthetic KG",
    )
    args = parser.parse_args()

    if args.triples:
        kg = load_triples_jsonl(args.triples)
        print(
            f"loaded real graph: {len(kg.triples)} directed edges, "
            f"{len(kg.companies)} companies, {len(kg.people)} people"
        )
    else:
        kg = build_synthetic_kg(
            args.companies, seed=args.seed, edges_per_company=args.edges_per_company
        )
        print(
            f"synthetic KG: {len(kg.triples)} directed edges, "
            f"{len(kg.companies)} companies, {len(kg.people)} people"
        )

    questions = generate_questions(kg, args.questions, seed=args.seed)
    depth_hist: dict[int, int] = {}
    for q in questions:
        depth_hist[len(q.sequence)] = depth_hist.get(len(q.sequence), 0) + 1
    counts = write_splits(questions, args.out)
    print(f"depth histogram: {dict(sorted(depth_hist.items()))}")
    print(f"wrote {counts} -> {args.out}")


if __name__ == "__main__":
    main()


__all__ = [
    "REVERSE_EDGE",
    "FinKG",
    "build_synthetic_kg",
    "load_triples_jsonl",
    "follow_sequence",
    "verbalize",
    "generate_questions",
    "write_splits",
    "GeneratedQuestion",
]
