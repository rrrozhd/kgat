"""Distillation data for the edge judge — alphina's ~490k logged critic verdicts.

Every past backfill logged the LLM critic's per-edge rulings on both sides:
accepted edges keep the critic's ``raw_faithfulness`` on ``company_relationships``;
rejected candidates land in ``extraction_rejections`` with the full payload
(claimed type/target/quote, verdict, faithfulness). Distilling that into a ~150M
cross-encoder gives the phase-2 reward a judge with zero marginal cost and no
LLM in the training loop (``kgat.train.judge`` trains it; ``kgat.train.edge_judge``
consumes it as the ``type_score``).

Export contract (read-only; column names verified against alphina's
``models/filing.py`` and the rejection payload written by
``relationship_critic.critique_and_recalibrate``, 2026-07-15). Accepted::

    SELECT cr.evidence_text,
           fc.text AS chunk_text,
           cr.source_company_name AS filer,
           cr.relationship_type,
           cr.target_company_name AS target,
           cr.raw_faithfulness AS faithfulness,
           'accept' AS verdict,
           pf.accession_number
    FROM company_relationships cr
    JOIN processed_filings pf ON cr.filing_id = pf.id
    JOIN filing_chunks fc ON cr.chunk_id = fc.id
    WHERE cr.raw_faithfulness IS NOT NULL;

Rejected (the critic's own rejects only — other ``reason`` values are
deterministic rule rejects with differently-shaped payloads)::

    SELECT er.claimed_quote AS evidence_text,
           fc.text AS chunk_text,
           er.raw_payload->>'source_company' AS filer,
           er.raw_payload->>'relationship_type' AS relationship_type,
           er.raw_payload->>'target_company' AS target,
           (er.raw_payload->>'faithfulness')::float AS faithfulness,
           'reject' AS verdict,
           pf.accession_number
    FROM extraction_rejections er
    JOIN processed_filings pf ON er.filing_id = pf.id
    JOIN filing_chunks fc ON er.chunk_id = fc.id
    WHERE er.reason = 'critic_reject';

The judge trains on the CHUNK text (that is what it will see inside the RL loop),
not the critic's chosen quote; the quote is kept as provenance. Known label
noise: the critic judged the QUOTE — a chunk could in principle support a claim
its quote did not. Splits are by filing (accession), as everywhere else.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from kgat.data.backfill_export import RELATIONSHIP_TYPES


@dataclass(frozen=True)
class JudgeExample:
    """One logged critic ruling: (chunk, claim) -> verdict + faithfulness."""

    text: str  # chunk text (falls back to the evidence quote in old exports)
    filer: str
    relation: str
    target: str
    verdict: str  # "accept" | "reject"
    faithfulness: float  # the critic's 0-1 score (rejects are low by instruction)
    filing: str

    def to_record(self) -> dict:
        return {
            "text": self.text,
            "filer": self.filer,
            "relation": self.relation,
            "target": self.target,
            "verdict": self.verdict,
            "faithfulness": self.faithfulness,
            "filing": self.filing,
        }


def render_judge_input(filer: str, relation: str, target: str, text: str) -> str:
    """The cross-encoder's input text — single-sourced for training AND scoring."""
    return (
        f"filer: {filer}\n"
        f"claim: the filer is the {relation} of {target}\n"
        f"evidence: {text}"
    )


# Directional relation pairs whose swap is a genuine error (not just a different
# relation): the filer selling-to vs buying-from (supplier↔customer), and owning vs
# owned-by (acquirer↔subsidiary). partner/competitor/investor have no such
# same-target inverse in the taxonomy, so they are excluded.
DIRECTION_FLIP = {
    "supplier": "customer",
    "customer": "supplier",
    "acquirer": "subsidiary",
    "subsidiary": "acquirer",
}


def _claim_key(filer: str, relation: str, target: str) -> tuple[str, str, str]:
    return (filer.strip().lower(), relation, target.strip().lower())


def mint_direction_negatives(
    examples: list[JudgeExample],
    *,
    min_source_faithfulness: float = 0.5,
    low_faithfulness: float = 0.0,
) -> list[JudgeExample]:
    """Mint direction-flipped hard negatives from ACCEPTED directional edges.

    Weak-supervision contrastive negatives (LITERATURE-ALIGNMENT §2): for each
    confidently-accepted ``(filer, r, target)`` with ``r`` in ``DIRECTION_FLIP``,
    emit a ``reject`` for ``(filer, flip(r), target)`` on the SAME chunk — the
    wrong-direction claim, false by that evidence. This targets the judge's weak
    reject-agreement axis (64% at 0.65) and the supplier/subsidiary flip buckets.

    Guards (a synthetic negative must be a RELIABLE reject):
      - source must be ``accept`` with faithfulness ≥ ``min_source_faithfulness``;
      - skip if the flipped claim is itself accepted anywhere (genuinely
        bidirectional relationships — e.g. two firms that both supply and buy);
      - skip if that exact (chunk, flipped-claim) already exists in the data;
      - dedup within the minted batch.

    Returns ONLY the new negatives (each inherits its source's ``filing`` so a
    filing-disjoint split keeps them with their positive). Apply to the TRAIN
    split only — minting over dev/test would contaminate the eval.
    """
    accepted = {
        _claim_key(e.filer, e.relation, e.target) for e in examples if e.verdict == "accept"
    }
    existing = {(e.text, _claim_key(e.filer, e.relation, e.target)) for e in examples}
    low = max(0.0, min(1.0, low_faithfulness))
    seen: set[tuple[str, tuple[str, str, str]]] = set()
    out: list[JudgeExample] = []
    for e in examples:
        if e.verdict != "accept" or e.relation not in DIRECTION_FLIP:
            continue
        if e.faithfulness < min_source_faithfulness:
            continue
        flip = DIRECTION_FLIP[e.relation]
        fk = _claim_key(e.filer, flip, e.target)
        if fk in accepted:  # the reverse direction is genuinely asserted somewhere
            continue
        marker = (e.text, fk)
        if marker in existing or marker in seen:
            continue
        seen.add(marker)
        out.append(
            JudgeExample(
                text=e.text,
                filer=e.filer,
                relation=flip,
                target=e.target,
                verdict="reject",
                faithfulness=low,
                filing=e.filing,
            )
        )
    return out


def load_judge_export_jsonl(path: str | Path) -> tuple[list[JudgeExample], int]:
    """Load export rows into ``JudgeExample``s. Returns ``(examples, n_skipped)``.

    Rows with missing/NULL claim fields or an off-taxonomy type are SKIPPED (and
    counted), not fatal — rejection payloads are LLM output and occasionally
    malformed; a distillation set tolerates dropped rows, silent corruption not.
    """
    examples: list[JudgeExample] = []
    skipped = 0
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("chunk_text") or row.get("evidence_text")
            filer = row.get("filer")
            relation = row.get("relationship_type")
            target = row.get("target")
            verdict = row.get("verdict")
            faith = row.get("faithfulness")
            if (
                not text
                or not filer
                or not target
                or relation not in RELATIONSHIP_TYPES
                or verdict not in ("accept", "reject")
            ):
                skipped += 1
                continue
            faith = float(faith) if faith is not None else (1.0 if verdict == "accept" else 0.0)
            examples.append(
                JudgeExample(
                    text=str(text),
                    filer=str(filer),
                    relation=str(relation),
                    target=str(target),
                    verdict=str(verdict),
                    faithfulness=max(0.0, min(1.0, faith)),
                    filing=str(row["accession_number"]),
                )
            )
    if not examples:
        raise ValueError(f"no usable judge examples in {path}")
    return examples, skipped


def split_judge_examples(
    examples: list[JudgeExample],
    *,
    fractions: tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 42,
) -> dict[str, list[JudgeExample]]:
    """Filing-disjoint train/dev/test split (same leakage guard as the pairs)."""
    filings = sorted({e.filing for e in examples})
    random.Random(seed).shuffle(filings)
    n = len(filings)
    n_train = int(n * fractions[0])
    n_dev = int(n * fractions[1])
    assign = {
        f: ("train" if i < n_train else "dev" if i < n_train + n_dev else "test")
        for i, f in enumerate(filings)
    }
    out: dict[str, list[JudgeExample]] = {"train": [], "dev": [], "test": []}
    for e in examples:
        out[assign[e.filing]].append(e)
    return out


def export_judge_dataset(
    examples: list[JudgeExample],
    out_dir: str | Path,
    *,
    fractions: tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 42,
) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = split_judge_examples(examples, fractions=fractions, seed=seed)
    counts: dict[str, int] = {}
    for split, items in splits.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for e in items:
                fh.write(json.dumps(e.to_record(), ensure_ascii=False) + "\n")
        counts[split] = len(items)
    return counts


def read_judge_jsonl(path: str | Path, max_examples: int | None = None) -> list[JudgeExample]:
    examples: list[JudgeExample] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(JudgeExample(**row))
            if max_examples is not None and len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError(f"no judge examples in {path}")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a judge-verdict export for training.")
    parser.add_argument("--export", required=True, help="raw export JSONL (docstring SQL)")
    parser.add_argument("--out", required=True, help="output dir for train/dev/test JSONLs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples, skipped = load_judge_export_jsonl(args.export)
    n_accept = sum(1 for e in examples if e.verdict == "accept")
    counts = export_judge_dataset(examples, args.out, seed=args.seed)
    print(
        f"{len(examples)} examples ({n_accept} accept / {len(examples) - n_accept} reject), "
        f"{skipped} malformed rows skipped; {len({e.filing for e in examples})} filings"
    )
    print(f"wrote {counts} -> {args.out}")


if __name__ == "__main__":
    main()


__all__ = [
    "JudgeExample",
    "render_judge_input",
    "DIRECTION_FLIP",
    "mint_direction_negatives",
    "load_judge_export_jsonl",
    "split_judge_examples",
    "export_judge_dataset",
    "read_judge_jsonl",
]
