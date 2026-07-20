"""Cascade frontier for the backfill extractor (DESIGN-BACKFILL.md step 3).

On held-out filings the small constrained extractor reads every chunk; a chunk
escalates to the big teacher pipeline iff its escalation signal falls below a
threshold tau. The baseline signal is the decode confidence (``exp(mean
logprob)``, from ``TripleDecodeResult``); the ``SIGNALS`` registry adds
grammar-agreement variants that fold in the unconstrained mass the trie clipped
(the masked confidence is blind to it — see ``TripleDecodeResult.agreement``),
and ``routing_auroc`` scores each signal as an error detector independent of tau.
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
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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

    ``agreement`` / ``min_agreement`` are the grammar-agreement signals from the
    decode (``TripleDecodeResult.agreement``): the unconstrained probability mass
    the trie retained, mean and worst-step. ``None`` on decodes that predate the
    extended logits contract.
    """

    gold: tuple[Triple, ...]
    pred: tuple[Triple, ...]
    confidence: float
    uncertain: tuple[Triple, ...] = ()
    agreement: float | None = None
    min_agreement: float | None = None


def _effective_pred(o: ExtractionOutcome) -> set[Triple]:
    """The prediction as scored: sub-floor teacher matches masked out."""
    return set(o.pred) - set(o.uncertain)


# Escalation signals: ExtractionOutcome -> score in [0, 1]; a chunk escalates iff
# its score falls below tau. ``confidence`` is the masked-logprob baseline; the
# agreement variants fold in the mass the grammar clipped (unmeasured decodes
# count as full agreement, so old outcome logs degrade to the baseline).
SIGNALS: dict[str, Any] = {
    "confidence": lambda o: o.confidence,
    "agreement": lambda o: 1.0 if o.agreement is None else o.agreement,
    "min_agreement": lambda o: 1.0 if o.min_agreement is None else o.min_agreement,
    "conf_x_agree": lambda o: o.confidence * (1.0 if o.agreement is None else o.agreement),
    "conf_x_min_agree": lambda o: o.confidence
    * (1.0 if o.min_agreement is None else o.min_agreement),
}


def configure_determinism(torch_module: Any, *, seed: int = 42) -> None:
    """Enable the seeded deterministic contract recorded by canonical manifests."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True)


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


def cascade_rows(
    outcomes: list[ExtractionOutcome],
    taus: list[float],
    *,
    signal: Any = None,
) -> list[dict]:
    """Sweep the escalation threshold; one metrics row per tau.

    A chunk escalates iff ``signal(outcome) < tau`` (default signal: the decode
    ``confidence``); its prediction is then the teacher's (gold).
    ``escalation_rate`` is the cost axis.
    """
    score = signal or SIGNALS["confidence"]
    rows: list[dict] = []
    for tau in taus:
        items: list[tuple[set[Triple], set[Triple]]] = []
        escalated = 0
        for o in outcomes:
            escalate = score(o) < tau
            escalated += int(escalate)
            pred = set(o.gold) if escalate else _effective_pred(o)
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


def quantile_taus(
    outcomes: list[ExtractionOutcome], signal: Any, *, n: int = 41
) -> list[float]:
    """Thresholds that sweep the signal's own value range.

    Grammar agreement clusters near 1.0, so the fixed 0..1.05 grid collapses to
    a handful of distinct escalation sets. Midpoints between consecutive distinct
    observed values (subsampled to ~``n``) trace the full frontier of any signal;
    endpoints 0.0 (escalate nothing) and max+eps (escalate everything) included.
    """
    values = sorted({signal(o) for o in outcomes})
    if not values:
        return [0.0]
    mids = [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)]
    if len(mids) > n - 2:
        step = len(mids) / (n - 2)
        mids = [mids[int(i * step)] for i in range(n - 2)]
    return [0.0, *mids, values[-1] + 1e-9]


def routing_auroc(outcomes: list[ExtractionOutcome], signal: Any = None) -> float | None:
    """AUROC of ``signal`` as a detector of small-model error — escalation accuracy.

    Error = the scored prediction differs from gold (the chunk SHOULD escalate).
    AUROC = P(signal(error chunk) < signal(correct chunk)), ties at 0.5 — the
    threshold-free routing-quality metric that makes signal improvements
    attributable independently of any tau choice. ``None`` when the split is
    degenerate (no errors, or nothing but errors).
    """
    score = signal or SIGNALS["confidence"]
    errors = [score(o) for o in outcomes if _effective_pred(o) != set(o.gold)]
    correct = [score(o) for o in outcomes if _effective_pred(o) == set(o.gold)]
    if not errors or not correct:
        return None
    wins = 0.0
    for e in errors:
        for c in correct:
            if e < c:
                wins += 1.0
            elif e == c:
                wins += 0.5
    return wins / (len(errors) * len(correct))


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
            "run_label": f"tau={row['tau']:.4f}",
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
        path = out_dir / f"tau_{row['tau']:.4f}.summary.json"
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
    mark_filer: bool = False,
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
    prompt_lens, n_truncated = [], 0
    for i, pair in enumerate(pairs):
        g = grammar_for(pair) if grammar_for is not None else grammar
        prompt = format_extraction_prompt(
            pair.filer, pair.text, g.relations, mark_filer=mark_filer
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_lens.append(len(prompt_ids))
        if len(prompt_ids) > max_prompt_tokens:  # keep the tail ("...extraction:")
            prompt_ids = prompt_ids[-max_prompt_tokens:]
            n_truncated += 1

        def logits_fn(generated: tuple[int, ...], allowed: list[int], _prompt=prompt_ids):
            input_ids = torch.tensor([_prompt + list(generated)], dtype=torch.long, device=device)
            with torch.no_grad():
                row = forward_last_logits(model, input_ids, keep=1)[0, -1]
            # Structured-decoding contract: gather ONLY the allowed continuations
            # on-device (see DecoderPolicyController.logits_fn). The full-row
            # logsumexp rides along (one on-device reduction) so the decode can
            # record grammar agreement — the mass the trie clipped.
            idx = torch.tensor(allowed, dtype=torch.long, device=row.device)
            vals = row.index_select(0, idx).float().cpu().tolist()
            full_lse = torch.logsumexp(row.float(), dim=0).item()
            return vals, full_lse

        result = decode_triples(logits_fn, g)
        outcomes.append(
            ExtractionOutcome(
                gold=norm_triples(pair.triples),
                pred=norm_triples(result.triples),
                confidence=result.confidence,
                uncertain=norm_triples(pair.uncertain),
                agreement=result.agreement,
                min_agreement=result.min_agreement,
            )
        )
        if (i + 1) % 25 == 0:
            print(f"  decoded {i + 1}/{len(pairs)}")
    if prompt_lens:
        mean_len = sum(prompt_lens) / len(prompt_lens)
        # Marked prompts run a few tokens longer per mention; at fixed
        # max_prompt_tokens that costs left-truncated chunk text — log it so a
        # marked-vs-unmarked delta can be checked against a truncation confound.
        print(
            f"  prompt tokens: mean={mean_len:.0f} max={max(prompt_lens)} "
            f"truncated={n_truncated}/{len(prompt_lens)} (mark_filer={mark_filer})"
        )
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
    parser.add_argument(
        "--entity-markers",
        action="store_true",
        help="wrap the filer's chunk mentions with [F]…[/F] markers (must match the "
        "adapter's SFT setting — entity_markers: true)",
    )
    parser.add_argument("--min-recall", type=float, default=0.8, help="success-gate recall floor")
    parser.add_argument(
        "--taus", default=None, help="comma-separated thresholds (default 0.00..1.05 step 0.05)"
    )
    parser.add_argument(
        "--signals",
        default="confidence",
        help=f"comma-separated escalation signals to sweep, or 'all' ({sorted(SIGNALS)}); "
        "the first one is the headline and keeps the legacy root output layout",
    )
    args = parser.parse_args()

    import torch

    from kgat.utils.hf import load_causal_lm

    configure_determinism(torch, seed=42)

    data_dir = Path(args.data_dir)
    pairs = read_pairs_jsonl(data_dir / f"{args.split}.jsonl", max_examples=args.max_examples)
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))

    four_bit = {"auto": "auto", "true": True, "false": False}[args.four_bit]
    model, tokenizer, device = load_causal_lm(
        args.model_id, adapter_path=args.adapter, device=args.device, four_bit=four_bit
    )

    # Prompt construction MUST match how the adapter was trained. The adapter stamps
    # its settings in extractor_meta.json; that file wins over the CLI flag (a marked
    # adapter eval'd unmarked, or vice-versa, silently produces a garbage comparison).
    mark_filer = args.entity_markers
    if args.adapter is not None:
        meta_path = Path(args.adapter) / "extractor_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_mark = bool(meta.get("entity_markers", False))
            if meta_mark != args.entity_markers:
                print(
                    f"cascade eval: overriding --entity-markers={args.entity_markers} with "
                    f"entity_markers={meta_mark} from adapter meta ({meta_path})"
                )
            mark_filer = meta_mark
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
        mark_filer=mark_filer,
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
                        "agreement": o.agreement,
                        "min_agreement": o.min_agreement,
                    }
                )
                + "\n"
            )

    from kgat.eval.frontier import build_frontier

    if args.signals == "all":
        signals = ["confidence", *[s for s in SIGNALS if s != "confidence"]]
    else:
        signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    unknown = [s for s in signals if s not in SIGNALS]
    if unknown:
        parser.error(f"unknown signals {unknown}; choose from {sorted(SIGNALS)}")

    report: list[tuple[str, float | None, dict | None]] = []
    first_rows: list[dict] = []
    for name in signals:
        signal = SIGNALS[name]
        if args.taus:
            taus = [float(t) for t in args.taus.split(",")]
        elif name == "confidence":
            taus = [round(i * 0.05, 2) for i in range(22)]  # 0.00 .. 1.05 (legacy grid)
        else:
            taus = quantile_taus(outcomes, signal)
        rows = cascade_rows(outcomes, taus, signal=signal)
        if name == signals[0]:
            first_rows = rows
        # `confidence` keeps the legacy root layout; other signals get subdirs.
        signal_dir = out_dir if name == "confidence" else out_dir / name
        write_summaries(rows, signal_dir, n_questions=len(outcomes))
        df, csv_path, png_path = build_frontier(
            signal_dir, signal_dir, accuracy_metric="recall", cost_axis="escalation_rate"
        )
        if name == signals[0]:
            print(df.to_string(index=False))
        print(f"wrote {csv_path}\nwrote {png_path}")
        best = headline(rows, min_recall=args.min_recall)
        report.append((name, routing_auroc(outcomes, signal), best))

    print(f"\nsignal comparison (recall floor {args.min_recall:.2f}):")
    for name, auroc, best in report:
        auroc_s = f"{auroc:.3f}" if auroc is not None else "  n/a"
        if best is None:
            print(f"  {name:<18} auroc {auroc_s}  GATE MISSED")
        else:
            print(
                f"  {name:<18} auroc {auroc_s}  "
                f"{best['recall']:.1%} recall (F1 {best['f1']:.3f}, "
                f"exact {best['exact']:.3f}) at {best['escalation_rate']:.1%} "
                f"escalation (tau={best['tau']:.4f})"
            )

    best = report[0][2]
    if best is None:
        max_r = max((r["recall"] for r in first_rows), default=0.0)
        print(
            f"GATE MISSED: no tau reaches recall >= {args.min_recall:.2f} "
            f"(best {max_r:.3f} at full small-model coverage sweep)"
        )
    else:
        print(
            f"HEADLINE ({report[0][0]}): {best['recall']:.1%} of teacher recall "
            f"(F1 {best['f1']:.3f}, exact {best['exact']:.3f}) at "
            f"{best['escalation_rate']:.1%} escalation (tau={best['tau']:.2f})"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "ExtractionOutcome",
    "SIGNALS",
    "micro_prf",
    "cascade_rows",
    "cascade_rows_2d",
    "pareto_front",
    "quantile_taus",
    "routing_auroc",
    "write_summaries",
    "headline",
    "decode_pairs",
]
