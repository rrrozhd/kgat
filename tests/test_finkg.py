"""Tests for the FinKG generator (kgat.data.finkg)."""

from __future__ import annotations

import json

from kgat.data.finkg import (
    REVERSE_EDGE,
    build_synthetic_kg,
    follow_sequence,
    generate_questions,
    load_triples_jsonl,
    verbalize,
    write_splits,
)
from kgat.data.subgraph import record_from_raw
from kgat.train.mine_trajectories import mine_dataset


def test_synthetic_kg_is_deterministic():
    a = build_synthetic_kg(n_companies=30, seed=7)
    b = build_synthetic_kg(n_companies=30, seed=7)
    assert a.triples == b.triples
    c = build_synthetic_kg(n_companies=30, seed=8)
    assert a.triples != c.triples


def test_every_edge_has_typed_reverse():
    kg = build_synthetic_kg(n_companies=30, seed=7)
    edge_set = {(t.head, t.relation, t.tail) for t in kg.triples}
    for head, relation, tail in edge_set:
        assert (tail, REVERSE_EDGE[relation], head) in edge_set


def test_generated_golds_match_sequence_replay():
    kg = build_synthetic_kg(n_companies=60, seed=7)
    questions = generate_questions(kg, 40, seed=7)
    for q in questions:
        # Replaying the generating relation sequence over the question's own
        # SUBGRAPH must reach every gold (the subgraph keeps all reasoning paths).
        adj: dict = {}
        for t in q.graph:
            adj.setdefault(t.head, {}).setdefault(t.relation, []).append(t.tail)
        reached = follow_sequence(adj, q.topic, q.sequence)
        assert set(q.golds) <= reached
        assert q.golds  # non-empty by construction


def test_depth_mix_is_respected():
    kg = build_synthetic_kg(n_companies=120, seed=42)
    questions = generate_questions(kg, 200, seed=42, depth_mix={1: 0.5, 2: 0.3, 3: 0.2})
    hist: dict[int, int] = {}
    for q in questions:
        hist[len(q.sequence)] = hist.get(len(q.sequence), 0) + 1
    assert set(hist) == {1, 2, 3}
    assert hist[1] > hist[3]  # mix roughly honored (sampling, not exact)


def test_verbalize_composes_right_to_left():
    q = verbalize("Vextra Systems", ("competitor_of", "customer_of"))
    # traversal: competitors first, then their suppliers
    assert q == "who are the suppliers of competitors of Vextra Systems?"


def test_board_interlocks_exist():
    # People hold roles at multiple companies (alphina's Person model), so the
    # interlocking-directorate 2-hop — has_director -> director_of landing on a
    # DIFFERENT company — must be reachable in the synthetic KG.
    kg = build_synthetic_kg(n_companies=60, seed=7)
    adj: dict = {}
    for t in kg.triples:
        adj.setdefault(t.head, {}).setdefault(t.relation, []).append(t.tail)
    interlocked = 0
    for company in kg.companies:
        others = follow_sequence(adj, company, ("has_director", "director_of")) - {company}
        interlocked += bool(others)
    assert interlocked >= 5  # plenty of interlock structure to sample questions from


def test_interlock_question_verbalizes():
    q = verbalize("Vextra Systems", ("has_director", "director_of"))
    assert q == ("who are the companies with board seats held by board members of Vextra Systems?")


def test_records_parse_and_mine_end_to_end(tmp_path):
    kg = build_synthetic_kg(n_companies=80, seed=11)
    questions = generate_questions(kg, 60, seed=11)
    counts = write_splits(questions, tmp_path)
    assert counts == {"train": 42, "dev": 9, "test": 9}

    # Standard-schema round trip: our normal loader path must parse it...
    records = []
    for line in (tmp_path / "train.jsonl").read_text().splitlines():
        records.append(record_from_raw(json.loads(line), dataset="finkg"))
    assert all(r.question.topic_entities and r.question.gold_answers for r in records)

    # ...and the UNCHANGED mining pipeline must produce trajectories from it.
    mined, stats = mine_dataset(records, beam_size=64, max_hops=4)
    assert stats.n_mined / stats.n_questions >= 0.9  # almost all oracle-reachable
    assert all(t.hit for t in mined)
    assert len(stats.depth_histogram) >= 2  # multiple depths -> adaptive-stop signal


def test_load_real_triples_export(tmp_path):
    path = tmp_path / "export.jsonl"
    rows = [
        {"head": "NVIDIA", "relation": "competitor_of", "tail": "AMD"},
        {"head": "TSMC", "relation": "supplier_to", "tail": "NVIDIA"},
        {"head": "NVIDIA", "relation": "in_sector", "tail": "sector:semiconductors"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    kg = load_triples_jsonl(path)
    edge_set = {(t.head, t.relation, t.tail) for t in kg.triples}
    assert ("NVIDIA", "customer_of", "TSMC") in edge_set  # auto reverse
    assert "NVIDIA" in kg.companies and "sector:semiconductors" in kg.sectors
    # "suppliers of competitors of AMD" -> TSMC
    adj: dict = {}
    for t in kg.triples:
        adj.setdefault(t.head, {}).setdefault(t.relation, []).append(t.tail)
    assert follow_sequence(adj, "AMD", ("competitor_of", "customer_of")) == {"TSMC"}
