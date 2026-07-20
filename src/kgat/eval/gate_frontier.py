"""Evaluate an escalation gate: trace the cost/quality curve it induces.

The gate scores each test chunk with its predicted extractor F1; escalating the
LOWEST-scoring chunks first traces a curve directly comparable to the cascade's
tau-curve and to the routing policy's P(ESCALATE) sweep — same accounting
(escalated chunks contribute their gold edges and no false positives), same axes.

Also reports the reference orderings on the SAME outcomes, so the comparison is
matched by construction rather than assembled across files:

* ``confidence`` — ``exp(mean logprob)``, the deployed tau-cascade heuristic
* ``oracle``     — actual per-chunk error count (unreachable ceiling)
* ``random``     — floor

CLI::

    python -m kgat.eval.gate_frontier --gate outputs/gates/escalation-gate \\
      --outcomes outputs/backfill/cascade-markers/outcomes.jsonl \\
      --pairs data/backfill/real-strat-v2/test.jsonl --out outputs/gate/curve.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

__all__ = ["curve_for_ordering", "evaluate_gate", "main"]


def curve_for_ordering(rows: list[dict], order: list[int], fractions: list[float]) -> list[dict]:
    """(escalation, P, R, F1) at each escalation fraction for one ordering."""
    n = len(rows)
    out = []
    for frac in fractions:
        k = int(frac * n)
        esc = set(order[:k])
        tp = fp = fn = 0
        for i, r in enumerate(rows):
            gold, pred = r["gold"], r["pred"]
            if i in esc:
                tp += len(gold)
                continue
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        out.append({
            "escalation": k / n if n else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        })
    return out


def evaluate_gate(
    rows: list[dict], gate_scores: list[float], *, fractions: list[float] | None = None
) -> dict:
    """Gate curve plus the matched reference orderings."""
    n = len(rows)
    if len(gate_scores) != n:
        raise ValueError(f"{len(gate_scores)} gate scores for {n} chunks")
    fractions = fractions or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    def err(r):
        return len(r["gold"] - r["pred"]) + len(r["pred"] - r["gold"])

    rng = random.Random(42)
    shuffled = list(range(n))
    rng.shuffle(shuffled)

    orderings = {
        # lowest predicted F1 escalates first
        "gate": sorted(range(n), key=lambda i: gate_scores[i]),
        "confidence": sorted(range(n), key=lambda i: rows[i].get("confidence", 1.0)),
        "oracle": sorted(range(n), key=lambda i: -err(rows[i])),
        "random": shuffled,
    }
    # Grammar-agreement orderings (extractor_cascade SIGNALS) when the decode
    # pass logged them; old outcomes files simply lack the columns.
    if any(r.get("agreement") is not None for r in rows):

        def agree(i: int) -> float:
            a = rows[i].get("agreement")
            return 1.0 if a is None else a

        def conf_x_agree(i: int) -> float:
            return rows[i].get("confidence", 1.0) * agree(i)

        orderings["agreement"] = sorted(range(n), key=agree)
        orderings["conf_x_agree"] = sorted(range(n), key=conf_x_agree)
    return {name: curve_for_ordering(rows, o, fractions) for name, o in orderings.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Escalation-gate cost/quality curve.")
    p.add_argument("--gate", required=True, help="trained gate model dir")
    p.add_argument("--outcomes", required=True, help="extractor outcomes.jsonl for the eval split")
    p.add_argument("--pairs", required=True, help="the split jsonl (for chunk text + filer)")
    p.add_argument("--out", required=True)
    p.add_argument("--loose-match", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    from kgat.data.backfill_export import read_pairs_jsonl
    from kgat.train.escalation_gate import load_gate_scorer

    raw = [json.loads(x) for x in Path(args.outcomes).read_text(encoding="utf-8").splitlines() if x.strip()]
    pairs = read_pairs_jsonl(Path(args.pairs))
    if len(raw) != len(pairs):
        raise ValueError(f"{len(raw)} outcomes vs {len(pairs)} pairs — misaligned split")

    normalize: Any = None
    if args.loose_match:
        from kgat.data.chunk_targets import normalize_name

        normalize = normalize_name

    def norm(trips):
        if normalize is None:
            return {tuple(t) for t in trips}
        return {(r, normalize(t)) for r, t in trips}

    rows = [
        {
            "gold": norm(d.get("gold") or []),
            "pred": norm(d.get("pred") or []),
            "confidence": d.get("confidence", 1.0),
            "agreement": d.get("agreement"),
        }
        for d in raw
    ]

    score_many = load_gate_scorer(args.gate, device=args.device)
    print(f"gate eval: scoring {len(pairs)} chunks")
    # confidence must match what the gate saw in training (render_gate_input)
    scores = score_many(
        [(p_.filer, p_.text, r.get("confidence")) for p_, r in zip(pairs, rows, strict=True)]
    )

    curves = evaluate_gate(rows, scores)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_chunks": len(rows), "curves": curves}, indent=2), encoding="utf-8")
    (out.with_suffix(".scores.json")).write_text(json.dumps(scores), encoding="utf-8")

    names = list(curves)
    print("\n escal | " + " | ".join(f"{k:>12}" for k in names))
    for i in range(len(curves["gate"])):
        esc = curves["gate"][i]["escalation"]
        print(f" {esc:5.1%} | " + " | ".join(f"{curves[k][i]['f1']:12.3f}" for k in names))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
