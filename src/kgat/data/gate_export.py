"""Build training data for a standalone ESCALATION GATE.

Motivation (measured, not assumed):

* Teaching ``ESCALATE`` *into* the extractor costs ~0.08 F1 of extraction and the
  loss is irrecoverable — warm-starting from the markers extractor recovered only
  0.018 of 0.082, across two seeds (``results/escalate-ordering-2026-07-19``). A
  SEPARATE gate leaves the extractor at its full 0.631.
* A joint routing reward is hackable: GRPO drove extraction from 20.8% of chunks
  to 3.2% while reward rose, by exploiting vacuous precision
  (``results/grpo-routing-2026-07-19``). A supervised gate has no such attractor.
* The two signals we already have are **complementary** — ``P(ESCALATE)`` and
  ``exp(mean logprob)`` overlap on only 44% of their top-15% picks, and a trivial
  50/50 rank blend already matches 252 GRPO updates at 20% escalation. A learned
  combiner should do better than either.

The gate's target is the extractor's **per-chunk F1 against teacher edges** — i.e.
"how badly will I get this chunk wrong?", which is exactly the quantity a router
needs and exactly what the fixed routing reward now scores (``chunk_quality`` with
``QUALITY_F1``). Labels are free: run the existing extractor over a split and read
its own outcomes.

Escalate the chunks with the LOWEST predicted F1.

CLI::

    python -m kgat.data.gate_export --outcomes outputs/backfill/train-markers/outcomes.jsonl \\
      --pairs data/backfill/real-strat-v2/train.jsonl --out data/gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

__all__ = ["chunk_f1", "hardness_target", "gate_examples", "write_gate_splits", "main"]


def chunk_f1(gold: set, pred: set) -> float:
    """Per-chunk F1 with the conventions the routing reward uses.

    * nothing to find and nothing claimed -> 1.0 (a correct skip; 56.4% of chunks)
    * something to find but nothing claimed -> 0.0
    * nothing to find but something claimed -> 0.0 (pure false positives)
    * otherwise the usual harmonic mean

    Matches ``train.backfill_routing.chunk_quality(..., QUALITY_F1)`` so the gate
    predicts the same quantity the reward scores.
    """
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    if not tp:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


def hardness_target(pair: Any, *, min_edges: int = 4) -> float:
    """1.0 = easy (keep), 0.0 = hard (escalate). The SFT warmup's difficulty proxy.

    Measured 2026-07-20: a gate trained on the extractor's REALIZED per-chunk F1
    ranks poorly (dev Spearman 0.110) because that target is 90% at the extremes —
    18% at 0.0, 72% at 1.0 — so pointwise regression collapses to the mean. The
    structural proxy is a cleaner, lower-variance signal about the chunk itself,
    and it is what ``P(ESCALATE)`` was trained on, which outranked everything else.

    Convention: the gate always predicts QUALITY (higher = the extractor will cope
    = do not escalate), so ``eval.gate_frontier`` can escalate lowest-first in
    every mode without a sign flag.
    """
    from kgat.train.routing_warmup import is_hard_chunk

    return 0.0 if is_hard_chunk(pair, min_edges=min_edges) else 1.0


def gate_examples(
    outcomes: list[dict],
    texts: list[str],
    filers: list[str],
    *,
    normalize: Any = None,
) -> list[dict]:
    """Pair each chunk's text with the extractor's realized F1 on it.

    ``outcomes`` rows carry ``gold``/``pred`` (and optionally ``confidence`` and
    ``p_escalate``, which ride along as extra gate features). Lengths must match —
    a misalignment here would silently train the gate on the wrong chunk, so it is
    an error rather than a zip-truncation.
    """
    if not (len(outcomes) == len(texts) == len(filers)):
        raise ValueError(
            f"length mismatch: {len(outcomes)} outcomes, {len(texts)} texts, {len(filers)} filers"
        )

    def norm(trips):
        if normalize is None:
            return {tuple(t) for t in trips}
        return {(r, normalize(t)) for r, t in trips}

    out = []
    for row, text, filer in zip(outcomes, texts, filers, strict=True):
        gold = norm(row.get("gold") or [])
        pred = norm(row.get("pred") or [])
        example = {
            "text": text,
            "filer": filer,
            "target": chunk_f1(gold, pred),
            "n_gold": len(gold),
            "n_pred": len(pred),
        }
        # Extra signals, when the outcomes came from the escalate sweep. Kept as
        # features rather than folded in, so the gate can learn the combination.
        for k in ("confidence", "p_escalate"):
            if k in row:
                example[k] = row[k]
        out.append(example)
    return out


def write_gate_splits(
    examples: list[dict], out_dir: str | Path, *, dev_frac: float = 0.1, seed: int = 42
) -> dict[str, int]:
    """Write train/dev JSONL. Split is by FILER where available, else by index.

    Splitting on the filer keeps chunks from one company out of both halves — the
    same leakage discipline the extractor splits use.
    """
    import random

    rng = random.Random(seed)
    filers = sorted({e.get("filer") or "" for e in examples})
    if len(filers) > 1:
        rng.shuffle(filers)
        n_dev = max(1, int(len(filers) * dev_frac))
        dev_filers = set(filers[:n_dev])
        dev = [e for e in examples if (e.get("filer") or "") in dev_filers]
        train = [e for e in examples if (e.get("filer") or "") not in dev_filers]
    else:
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        n_dev = max(1, int(len(examples) * dev_frac))
        dev_idx = set(idx[:n_dev])
        dev = [e for i, e in enumerate(examples) if i in dev_idx]
        train = [e for i, e in enumerate(examples) if i not in dev_idx]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in (("train", train), ("dev", dev)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        counts[name] = len(rows)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="Build escalation-gate training data.")
    p.add_argument("--outcomes", required=True, help="extractor outcomes.jsonl (gold/pred per chunk)")
    p.add_argument("--pairs", required=True, help="the split jsonl those outcomes were produced from")
    p.add_argument("--out", default="data/gate")
    p.add_argument("--loose-match", action="store_true")
    p.add_argument("--dev-frac", type=float, default=0.1)
    p.add_argument(
        "--target", default="hardness", choices=["hardness", "f1"],
        help="hardness = structural proxy (default; needs no decode pass); "
             "f1 = the extractor's realized per-chunk F1 (ranks poorly, see "
             "results/escalation-gate-2026-07-20)",
    )
    p.add_argument("--hard-min-edges", type=int, default=4)
    args = p.parse_args()

    from kgat.data.backfill_export import read_pairs_jsonl

    outcomes = [json.loads(line) for line in open(args.outcomes, encoding="utf-8") if line.strip()]
    pairs = read_pairs_jsonl(Path(args.pairs))
    normalize = None
    if args.loose_match:
        from kgat.data.chunk_targets import normalize_name

        normalize = normalize_name

    examples = gate_examples(
        outcomes, [p_.text for p_ in pairs], [p_.filer for p_ in pairs], normalize=normalize
    )
    if args.target == "hardness":
        # Overwrite the target; the decode-derived fields (confidence) stay as
        # FEATURES, which is what the f1-target run failed to actually use.
        for ex, pr in zip(examples, pairs, strict=True):
            ex["target"] = hardness_target(pr, min_edges=args.hard_min_edges)
    counts = write_gate_splits(examples, args.out, dev_frac=args.dev_frac)
    targets = [e["target"] for e in examples]
    perfect = sum(1 for t in targets if t >= 0.999)
    zero = sum(1 for t in targets if t <= 0.001)
    print(
        f"{len(examples)} gate examples: mean target {sum(targets) / len(targets):.3f}, "
        f"{perfect} perfect ({perfect / len(targets):.1%}), {zero} total-miss "
        f"({zero / len(targets):.1%})"
    )
    print(f"wrote {counts} -> {args.out}")


if __name__ == "__main__":
    main()
