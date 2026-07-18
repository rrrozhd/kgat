"""Evaluate a GRPO-trained routing policy — ONE frontier point per policy.

The cascade (``eval.extractor_cascade``) sweeps a confidence threshold tau over one
fixed model, tracing a curve from a single policy. A routing policy is different:
``ESCALATE`` is a decode ACTION it learned, so a given policy escalates whatever
fraction it learned to at its lambda — it yields exactly ONE (recall, escalation)
point. The frontier is traced by sweeping lambda across RETRAINED policies, then
overlaying those points on the cascade's tau-curve (the dominance test).

Escalated chunks score as teacher output by construction (the teacher IS the label
source), identical to the cascade's accounting, so the two are directly comparable
on the same axes.

Usage::

    python -m kgat.eval.routing_frontier --model-id Qwen/Qwen3-0.6B \\
      --adapter outputs/adapters/routing-lam0.1 --data-dir data/backfill/real-strat-v2 \\
      --split test --out outputs/routing/lam0.1.json --label lam=0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kgat.controller.constrained_decoding import build_triple_grammar
from kgat.data.backfill_export import ExtractionPair, read_pairs_jsonl
from kgat.train.backfill_routing import (
    ESCALATE_LABEL,
    ROUTE_ESCALATE,
    ROUTE_EXTRACT,
    decision_from_result,
)

__all__ = [
    "evaluate_routing_policy",
    "escalate_logprob",
    "collect_routing_outcomes",
    "sweep_escalate_threshold",
    "main",
]


def escalate_logprob(logits_fn, grammar) -> float:
    """Exact ``log P(ESCALATE)`` under the grammar's first segment.

    ESCALATE lives in the SAME first decision segment as ``NONE`` and every
    relation (``build_triple_grammar(..., sentinels=(ESCALATE_LABEL,))``), so its
    path probability is a free continuous routing signal on every forward pass the
    eval already makes. Greedy decoding collapses it to a hard argmax and reports
    ONE frontier point; keeping the scalar lets a threshold sweep trace a whole
    curve from a single policy — the routing analog of the cascade's ``tau``.

    Walks the sentinel's token path, summing mask-renormalized (temperature-1)
    step logprobs, exactly as ``decode_triples`` scores the path it takes.
    """
    import math

    ids = grammar.enc_sentinel[ESCALATE_LABEL]
    node = grammar.first.trie
    generated: list[int] = []
    total = 0.0
    for tok in ids:
        allowed = node.allowed()
        vals = list(logits_fn(tuple(generated), allowed))
        m = max(vals)
        log_z = m + math.log(sum(math.exp(v - m) for v in vals))
        total += vals[allowed.index(tok)] - log_z
        generated.append(tok)
        node = node.step(tok)
    return total


def sweep_escalate_threshold(rows: list[dict], *, n_points: int = 41) -> list[dict]:
    """Trace (escalation, P, R, F1) vs the escalate-probability threshold.

    ``rows`` come from :func:`collect_routing_outcomes`: each carries the policy's
    ``p_escalate`` plus the extract/skip decode it would fall back to. A chunk
    escalates iff ``p_escalate >= theta`` and then contributes its gold edges (the
    teacher solves it, no false positives) — identical accounting to
    ``eval.extractor_cascade``, so the two curves are directly comparable.
    """
    thetas = sorted({r["p_escalate"] for r in rows} | {0.0, 1.0 + 1e-9})
    if len(thetas) > n_points:
        step = len(thetas) / n_points
        thetas = [thetas[min(int(i * step), len(thetas) - 1)] for i in range(n_points)]
        thetas = sorted(set(thetas))
    out = []
    for theta in thetas:
        tp = fp = fn = n_esc = 0
        for r in rows:
            gold = {tuple(t) for t in r["gold"]}
            if r["p_escalate"] >= theta:
                n_esc += 1
                tp += len(gold)
                continue
            pred = {tuple(t) for t in r["pred"]}
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(
            {
                "theta": theta,
                "escalation_rate": n_esc / len(rows) if rows else 0.0,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_escalated": n_esc,
            }
        )
    return out


def evaluate_routing_policy(
    pairs: list[ExtractionPair],
    model: Any,
    tokenizer: Any,
    device: str,
    *,
    grammar_for: Any,
    max_prompt_tokens: int = 1024,
    mark_filer: bool = False,
    normalize: Any = None,
) -> dict:
    """Greedy-decode each chunk, follow the policy's own route, score micro P/R.

    Escalated chunks contribute their GOLD triples (the teacher would have solved
    them) and count toward ``escalation_rate`` — the cost axis. Skips contribute
    nothing. Extractions are scored against gold.
    """
    import torch

    from kgat.controller.constrained_decoding import decode_triples
    from kgat.controller.prompting import format_extraction_prompt
    from kgat.utils.hf import forward_last_logits

    def nz(trips):
        if normalize is None:
            return set(trips)
        return {(r, normalize(t)) for r, t in trips}

    tp = fp = fn = 0
    n_escalated = n_extract = n_skip = 0
    for i, pair in enumerate(pairs):
        g = grammar_for(pair)
        prompt = format_extraction_prompt(
            pair.filer, pair.text, g.relations, mark_filer=mark_filer
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        def logits_fn(generated, allowed, _p=prompt_ids):
            input_ids = torch.tensor([_p + list(generated)], dtype=torch.long, device=device)
            with torch.no_grad():
                row = forward_last_logits(model, input_ids, keep=1)[0, -1]
            idx = torch.tensor(allowed, dtype=torch.long, device=row.device)
            return row.index_select(0, idx).float().cpu().tolist()

        result = decode_triples(logits_fn, g)  # greedy (temperature 0)
        decision = decision_from_result(result)
        gold = nz(pair.triples)

        if decision.route == ROUTE_ESCALATE:
            n_escalated += 1
            tp += len(gold)  # teacher solves it -> all gold recovered, no FP
        elif decision.route == ROUTE_EXTRACT:
            n_extract += 1
            pred = nz(decision.triples) - nz(pair.uncertain)
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
        else:
            n_skip += 1
            fn += len(gold)
        if (i + 1) % 100 == 0:
            print(f"  routed {i + 1}/{len(pairs)}")

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(pairs)
    return {
        "n_chunks": n,
        "escalation_rate": n_escalated / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_escalated": n_escalated,
        "n_extract": n_extract,
        "n_skip": n_skip,
    }


def collect_routing_outcomes(
    pairs: list[ExtractionPair],
    model: Any,
    tokenizer: Any,
    device: str,
    *,
    grammar_for: Any,
    grammar_noesc_for: Any,
    max_prompt_tokens: int = 1024,
    mark_filer: bool = False,
    normalize: Any = None,
) -> list[dict]:
    """Per-chunk ``p_escalate`` + the extract/skip decode the policy falls back to.

    Two passes per chunk: score the ESCALATE path under the routing grammar, then
    decode greedily under a grammar with the sentinel REMOVED — so the fallback is
    the policy's genuine extract/skip choice, never a masked-out escalate. Feed the
    rows to :func:`sweep_escalate_threshold`.
    """
    import math

    import torch

    from kgat.controller.constrained_decoding import decode_triples
    from kgat.controller.prompting import format_extraction_prompt
    from kgat.utils.hf import forward_last_logits

    def nz(trips):
        if normalize is None:
            return [list(t) for t in trips]
        return [[r, normalize(t)] for r, t in trips]

    rows: list[dict] = []
    for i, pair in enumerate(pairs):
        g = grammar_for(pair)
        prompt = format_extraction_prompt(
            pair.filer, pair.text, g.relations, mark_filer=mark_filer
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        def logits_fn(generated, allowed, _p=prompt_ids):
            input_ids = torch.tensor([_p + list(generated)], dtype=torch.long, device=device)
            with torch.no_grad():
                row = forward_last_logits(model, input_ids, keep=1)[0, -1]
            idx = torch.tensor(allowed, dtype=torch.long, device=row.device)
            return row.index_select(0, idx).float().cpu().tolist()

        p_esc = math.exp(escalate_logprob(logits_fn, g))
        fallback = decode_triples(logits_fn, grammar_noesc_for(pair))  # greedy, no sentinel
        uncertain = {tuple(t) for t in nz(pair.uncertain)}
        rows.append(
            {
                "p_escalate": p_esc,
                "pred": [t for t in nz(fallback.triples) if tuple(t) not in uncertain],
                "gold": nz(pair.triples),
                "confidence": fallback.confidence,
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(pairs)}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="One frontier point for a GRPO routing policy.")
    p.add_argument("--model-id", required=True)
    p.add_argument("--adapter", required=True, help="GRPO routing adapter dir")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--out", required=True, help="write the point JSON here")
    p.add_argument("--label", default="policy", help="e.g. lam=0.1")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--max-triples", type=int, default=28)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--targets", default="vocab", choices=["vocab", "chunk"])
    p.add_argument("--loose-match", action="store_true")
    p.add_argument("--entity-markers", action="store_true")
    p.add_argument(
        "--escalate-sweep",
        action="store_true",
        help="trace a full curve by sweeping P(ESCALATE) instead of reporting one greedy point",
    )
    args = p.parse_args()

    from kgat.utils.hf import load_causal_lm

    data_dir = Path(args.data_dir)
    pairs = read_pairs_jsonl(data_dir / f"{args.split}.jsonl", max_examples=args.max_examples)
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))

    model, tokenizer, device = load_causal_lm(
        args.model_id, adapter_path=args.adapter, device=args.device, four_bit=False
    )

    # Prompt construction must match how the policy was trained.
    mark_filer = args.entity_markers
    meta_path = Path(args.adapter) / "extractor_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mark_filer = bool(meta.get("entity_markers", mark_filer))
        print(f"routing eval: entity_markers={mark_filer} (from {meta_path})")

    # The routing grammar MUST carry the ESCALATE sentinel or the policy cannot
    # express the escalate action at all.
    if args.targets == "chunk":
        from kgat.data.chunk_targets import chunk_target_candidates

        def grammar_for(pair):
            return build_triple_grammar(
                vocab["relations"],
                chunk_target_candidates(pair.text, filer=pair.filer),
                tokenizer,
                eos_id=tokenizer.eos_token_id,
                max_triples=args.max_triples,
                sentinels=(ESCALATE_LABEL,),
            )
    else:
        shared = build_triple_grammar(
            vocab["relations"],
            vocab["targets"],
            tokenizer,
            eos_id=tokenizer.eos_token_id,
            max_triples=args.max_triples,
            sentinels=(ESCALATE_LABEL,),
        )

        def grammar_for(pair):
            return shared

    normalize = None
    if args.loose_match:
        from kgat.data.chunk_targets import normalize_name

        normalize = normalize_name

    print(f"routing eval: {len(pairs)} {args.split} chunks, targets={args.targets}")

    if args.escalate_sweep:
        # Same grammar minus the sentinel: the fallback must be the policy's real
        # extract/skip decode, not a masked escalate.
        if args.targets == "chunk":
            from kgat.data.chunk_targets import chunk_target_candidates

            def grammar_noesc_for(pair):
                return build_triple_grammar(
                    vocab["relations"],
                    chunk_target_candidates(pair.text, filer=pair.filer),
                    tokenizer,
                    eos_id=tokenizer.eos_token_id,
                    max_triples=args.max_triples,
                )
        else:
            shared_noesc = build_triple_grammar(
                vocab["relations"],
                vocab["targets"],
                tokenizer,
                eos_id=tokenizer.eos_token_id,
                max_triples=args.max_triples,
            )

            def grammar_noesc_for(pair):
                return shared_noesc

        rows = collect_routing_outcomes(
            pairs,
            model,
            tokenizer,
            device,
            grammar_for=grammar_for,
            grammar_noesc_for=grammar_noesc_for,
            max_prompt_tokens=args.max_prompt_tokens,
            mark_filer=mark_filer,
            normalize=normalize,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        outcomes_path = out.with_name(out.stem + ".outcomes.jsonl")
        with outcomes_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        curve = sweep_escalate_threshold(rows)
        out.write_text(
            json.dumps(
                {
                    "label": args.label,
                    "adapter": args.adapter,
                    "n_chunks": len(rows),
                    "entity_markers": mark_filer,
                    "loose_match": bool(args.loose_match),
                    "targets": args.targets,
                    "curve": curve,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {outcomes_path} and {out}")
        print("  escal   P      R      F1     theta")
        for c in curve:
            print(
                f"  {c['escalation_rate']:5.1%}  {c['precision']:.3f}  {c['recall']:.3f}  "
                f"{c['f1']:.3f}  {c['theta']:.4g}"
            )
        return

    point = evaluate_routing_policy(
        pairs,
        model,
        tokenizer,
        device,
        grammar_for=grammar_for,
        max_prompt_tokens=args.max_prompt_tokens,
        mark_filer=mark_filer,
        normalize=normalize,
    )
    point["label"] = args.label
    point["adapter"] = args.adapter
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(point, indent=2), encoding="utf-8")
    print(
        f"[{args.label}] escalation={point['escalation_rate']:.3f} "
        f"P={point['precision']:.3f} R={point['recall']:.3f} F1={point['f1']:.3f} "
        f"(esc={point['n_escalated']} extract={point['n_extract']} skip={point['n_skip']})"
    )


if __name__ == "__main__":
    main()
