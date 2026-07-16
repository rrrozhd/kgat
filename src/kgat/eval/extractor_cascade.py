"""Cascade frontier for the backfill extractor (DESIGN-BACKFILL.md step 3).

On held-out filings the small constrained extractor reads every chunk; a chunk
escalates to the big teacher pipeline iff the extractor's decode confidence
(``exp(mean logprob)``, from ``TripleDecodeResult``) falls below a threshold tau.
Sweeping tau traces the write-path cost/quality frontier: quality vs the teacher's
edges against the fraction of chunks escalated (~ API $ per filing). Escalated
chunks score as teacher output by construction (the teacher IS the label source),
so tau high enough to escalate everything recovers 100% teacher quality at 100%
cost — the headline is "X% of teacher quality at Y% escalation".

Per-tau summaries follow the ``kgat.eval.harness`` format, so the standard
``kgat.eval.frontier`` machinery plots them with
``accuracy_metric=recall`` / ``cost_axis=escalation_rate``.

The threshold sweep is pure Python (testable without torch); only the CLI loads a
model. Like ``eval.frontier``, the CLI is argparse, not Hydra.

CLI::

    python -m kgat.eval.extractor_cascade --model-id Qwen/Qwen3-0.6B \\
        --adapter outputs/adapters/qwen3-0.6b-extractor \\
        --data-dir data/backfill/synthetic --out-dir outputs/backfill/cascade
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kgat.controller.constrained_decoding import build_triple_grammar, decode_triples
from kgat.controller.prompting import format_extraction_prompt
from kgat.data.backfill_export import ExtractionPair, read_pairs_jsonl

Triple = tuple[str, str]


@dataclass(frozen=True)
class ExtractionOutcome:
    """One evaluated chunk: teacher triples, small-model triples, its confidence.

    ``uncertain`` carries the chunk's sub-floor teacher edges: predictions that
    match them are removed before scoring (they cannot be true positives — not
    gold — but counting them as FALSE positives would punish the model for
    agreeing with a low-confidence teacher).
    """

    gold: tuple[Triple, ...]
    pred: tuple[Triple, ...]
    confidence: float
    uncertain: tuple[Triple, ...] = ()


def micro_prf(items: list[tuple[set[Triple], set[Triple]]]) -> dict[str, float]:
    """Micro precision/recall/F1 over (pred, gold) triple sets, plus exact-match.

    Micro counts make multi-triple chunks weigh by their edges; ``exact`` (pred set
    == gold set per chunk) is the metric where correct NONEs show up — negatives
    contribute no TP/FP/FN, so P/R alone would render the skip class invisible.
    """
    tp = fp = fn = 0
    exact = 0
    for pred, gold in items:
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += int(pred == gold)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": exact / len(items) if items else 1.0,
    }


def cascade_rows(outcomes: list[ExtractionOutcome], taus: list[float]) -> list[dict]:
    """Sweep the escalation threshold; one metrics row per tau.

    A chunk escalates iff ``confidence < tau``; its prediction is then the
    teacher's (gold). ``escalation_rate`` is the cost axis.
    """
    rows: list[dict] = []
    for tau in taus:
        items: list[tuple[set[Triple], set[Triple]]] = []
        escalated = 0
        for o in outcomes:
            escalate = o.confidence < tau
            escalated += int(escalate)
            pred = set(o.gold) if escalate else set(o.pred) - set(o.uncertain)
            items.append((pred, set(o.gold)))
        metrics = micro_prf(items)
        rows.append(
            {
                "tau": tau,
                "escalation_rate": escalated / len(outcomes) if outcomes else 0.0,
                **metrics,
            }
        )
    return rows


def cascade_rows_2d(
    outcomes: list[ExtractionOutcome],
    taus_none: list[float],
    taus_extract: list[float],
) -> list[dict]:
    """Per-route threshold sweep: separate bars for NONE decodes vs extractions.

    ``exp(mean logprob)`` is computed over very different decision counts for a
    NONE decode (~1 step) vs a multi-triple one (~15), so a single tau conflates
    output length with uncertainty. Escalate iff ``confidence < tau_none`` when
    the extractor predicted nothing, else iff ``confidence < tau_extract``.
    Returns one row per (tau_none, tau_extract) with the same metric fields as
    :func:`cascade_rows`.
    """
    rows: list[dict] = []
    for tn in taus_none:
        for te in taus_extract:
            items: list[tuple[set[Triple], set[Triple]]] = []
            escalated = 0
            for o in outcomes:
                tau = tn if not o.pred else te
                escalate = o.confidence < tau
                escalated += int(escalate)
                pred = set(o.gold) if escalate else set(o.pred) - set(o.uncertain)
                items.append((pred, set(o.gold)))
            metrics = micro_prf(items)
            rows.append(
                {
                    "tau_none": tn,
                    "tau_extract": te,
                    "escalation_rate": escalated / len(outcomes) if outcomes else 0.0,
                    **metrics,
                }
            )
    return rows


def pareto_front(
    rows: list[dict], *, cost_key: str = "escalation_rate", quality_key: str = "recall"
) -> list[dict]:
    """Upper envelope: rows not dominated by any cheaper-or-equal, better-or-equal row."""
    front = [
        r
        for r in rows
        if not any(
            (o[cost_key] <= r[cost_key] and o[quality_key] > r[quality_key])
            or (o[cost_key] < r[cost_key] and o[quality_key] >= r[quality_key])
            for o in rows
        )
    ]
    return sorted(front, key=lambda r: (r[cost_key], -r[quality_key]))


def write_summaries(rows: list[dict], out_dir: str | Path, *, n_questions: int) -> list[Path]:
    """Write one harness-format ``*.summary.json`` per tau (frontier input)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in rows:
        summary = {
            "run_label": f"tau={row['tau']:.2f}",
            "dataset": "backfill",
            "tau": row["tau"],
            "n_questions": n_questions,
            "metrics": {
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "exact": row["exact"],
            },
            "mean_cost": {
                "escalation_rate": row["escalation_rate"],
                "llm_calls": row["escalation_rate"],  # 1 big-LLM call per escalated chunk
            },
        }
        path = out_dir / f"tau_{row['tau']:.2f}.summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def headline(rows: list[dict], *, min_recall: float = 0.8) -> dict | None:
    """Cheapest tau meeting the pilot's recall gate (DESIGN-BACKFILL success gate)."""
    eligible = [r for r in rows if r["recall"] >= min_recall]
    if not eligible:
        return None
    return min(eligible, key=lambda r: (r["escalation_rate"], r["tau"]))


# ---------------------------------------------------------------------------
# Model-backed decoding (CLI only; everything above is torch-free)
# ---------------------------------------------------------------------------


def decode_pairs(
    pairs: list[ExtractionPair],
    model: Any,
    tokenizer: Any,
    device: str,
    grammar: Any,
    *,
    max_prompt_tokens: int = 1024,
    grammar_for: Any = None,
    normalize: Any = None,
) -> list[ExtractionOutcome]:
    """Greedy grammar-constrained extraction over pairs with the loaded model.

    ``grammar_for(pair)`` overrides the shared grammar per pair (chunk-local
    candidates); ``normalize(name)`` maps gold AND predicted target names to a
    comparison key before scoring (resolver-style loose matching — without it,
    a chunk-local surface form like "NVIDIA Corporation" would count as a miss
    against the teacher label "NVIDIA Corp").
    """
    import torch

    from kgat.utils.hf import forward_last_logits

    def norm_triples(triples):
        if normalize is None:
            return tuple(triples)
        return tuple((r, normalize(t)) for r, t in triples)

    outcomes: list[ExtractionOutcome] = []
    for i, pair in enumerate(pairs):
        g = grammar_for(pair) if grammar_for is not None else grammar
        prompt = format_extraction_prompt(pair.filer, pair.text, g.relations)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_tokens:  # keep the tail ("...extraction:")
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        def logits_fn(generated: tuple[int, ...], allowed: list[int], _prompt=prompt_ids):
            input_ids = torch.tensor([_prompt + list(generated)], dtype=torch.long, device=device)
            with torch.no_grad():
                row = forward_last_logits(model, input_ids, keep=1)[0, -1]
            # Structured-decoding contract: gather ONLY the allowed continuations
            # on-device (see DecoderPolicyController.logits_fn).
            idx = torch.tensor(allowed, dtype=torch.long, device=row.device)
            return row.index_select(0, idx).float().cpu().tolist()

        result = decode_triples(logits_fn, g)
        outcomes.append(
            ExtractionOutcome(
                gold=norm_triples(pair.triples),
                pred=norm_triples(result.triples),
                confidence=result.confidence,
                uncertain=norm_triples(pair.uncertain),
            )
        )
        if (i + 1) % 25 == 0:
            print(f"  decoded {i + 1}/{len(pairs)}")
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the extractor-vs-teacher escalation threshold into a frontier."
    )
    parser.add_argument("--model-id", required=True, help="HF id of the small extractor base")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (sft_extractor output)")
    parser.add_argument("--data-dir", required=True, help="backfill_export output dir")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--out-dir", required=True, help="where summaries + frontier land")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-triples", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--four-bit",
        default="false",
        choices=["auto", "true", "false"],
        help="4-bit base for the eval; default fp16 (4-bit eval jitter masked real "
        "effects in the GRPO sweep v1 — see STATUS.md)",
    )
    parser.add_argument(
        "--targets",
        default="vocab",
        choices=["vocab", "chunk"],
        help="target constraint: closed vocab.json list, or open-vocabulary "
        "chunk-local capitalized spans",
    )
    parser.add_argument(
        "--loose-match",
        action="store_true",
        help="score with resolver-style normalized names (recommended with --targets chunk)",
    )
    parser.add_argument("--min-recall", type=float, default=0.8, help="success-gate recall floor")
    parser.add_argument(
        "--taus", default=None, help="comma-separated thresholds (default 0.00..1.05 step 0.05)"
    )
    args = parser.parse_args()

    from kgat.utils.hf import load_causal_lm

    data_dir = Path(args.data_dir)
    pairs = read_pairs_jsonl(data_dir / f"{args.split}.jsonl", max_examples=args.max_examples)
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))

    four_bit = {"auto": "auto", "true": True, "false": False}[args.four_bit]
    model, tokenizer, device = load_causal_lm(
        args.model_id, adapter_path=args.adapter, device=args.device, four_bit=four_bit
    )
    grammar = None
    grammar_for = None
    if args.targets == "chunk":
        from kgat.data.chunk_targets import chunk_target_candidates

        def grammar_for(pair):
            return build_triple_grammar(
                vocab["relations"],
                chunk_target_candidates(pair.text, filer=pair.filer),
                tokenizer,
                eos_id=tokenizer.eos_token_id,
                max_triples=args.max_triples,
            )

        print(f"cascade eval: {len(pairs)} {args.split} pairs, chunk-local targets")
    else:
        grammar = build_triple_grammar(
            vocab["relations"],
            vocab["targets"],
            tokenizer,
            eos_id=tokenizer.eos_token_id,
            max_triples=args.max_triples,
        )
        print(f"cascade eval: {len(pairs)} {args.split} pairs, {len(grammar.targets)} targets")

    normalize = None
    if args.loose_match:
        from kgat.data.chunk_targets import normalize_name

        normalize = normalize_name
    outcomes = decode_pairs(
        pairs,
        model,
        tokenizer,
        device,
        grammar,
        max_prompt_tokens=args.max_prompt_tokens,
        grammar_for=grammar_for,
        normalize=normalize,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "outcomes.jsonl").open("w", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(
                json.dumps(
                    {
                        "gold": [list(t) for t in o.gold],
                        "pred": [list(t) for t in o.pred],
                        "confidence": o.confidence,
                    }
                )
                + "\n"
            )

    if args.taus:
        taus = [float(t) for t in args.taus.split(",")]
    else:
        taus = [round(i * 0.05, 2) for i in range(22)]  # 0.00 .. 1.05
    rows = cascade_rows(outcomes, taus)
    write_summaries(rows, out_dir, n_questions=len(outcomes))

    from kgat.eval.frontier import build_frontier

    df, csv_path, png_path = build_frontier(
        out_dir, out_dir, accuracy_metric="recall", cost_axis="escalation_rate"
    )
    print(df.to_string(index=False))
    print(f"wrote {csv_path}\nwrote {png_path}")

    best = headline(rows, min_recall=args.min_recall)
    if best is None:
        max_r = max((r["recall"] for r in rows), default=0.0)
        print(
            f"GATE MISSED: no tau reaches recall >= {args.min_recall:.2f} "
            f"(best {max_r:.3f} at full small-model coverage sweep)"
        )
    else:
        print(
            f"HEADLINE: {best['recall']:.1%} of teacher recall "
            f"(F1 {best['f1']:.3f}, exact {best['exact']:.3f}) at "
            f"{best['escalation_rate']:.1%} escalation (tau={best['tau']:.2f})"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "ExtractionOutcome",
    "micro_prf",
    "cascade_rows",
    "cascade_rows_2d",
    "pareto_front",
    "write_summaries",
    "headline",
    "decode_pairs",
]
