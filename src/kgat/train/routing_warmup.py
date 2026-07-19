"""Teach the extractor the ESCALATE action, so GRPO can then learn WHEN to use it.

Measured 2026-07-18: a plain SFT extractor assigns P(ESCALATE) ~ 6e-9 at step 0 —
about nine orders of magnitude under the other actions. GRPO can only reinforce what
it samples, so the routing policy could never discover escalation by exploration
(~170M decodes to sample it once; a +16 logit bias would be *forcing*, not exploring,
and would silently break the on-policy ratio). The action has to be TAUGHT.

This is that warmup: the same extraction SFT, but with the ESCALATE-sentinel grammar
and a subset of chunks whose target IS ``ESCALATE``. The result is a policy that both
extracts and can express escalation, which GRPO then tunes with the λ cost knob.

The warmup only has to make the action *reachable and sensibly-prior'd* — not
correct. Which chunks get labelled ESCALATE uses a free difficulty proxy (below);
GRPO learns the real policy from reward.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from kgat.controller.constrained_decoding import (
    build_triple_grammar,
    encode_triples_target,
)
from kgat.controller.prompting import format_extraction_prompt
from kgat.data.backfill_export import ExtractionPair, read_pairs_jsonl
from kgat.train.backfill_routing import ESCALATE_LABEL
from kgat.utils.hf import attach_lora, load_causal_lm, require_ml

__all__ = ["is_hard_chunk", "select_escalate_ids", "encode_warmup_example", "run_routing_warmup"]


def is_hard_chunk(pair: ExtractionPair, *, min_edges: int = 4) -> bool:
    """Free difficulty proxy — chunks where self-extraction is most error-prone.

    Two signals, no GPU pass required:

    * **non-verbatim target** — a gold target that cannot be matched to a span in
      the chunk. The model must recall the name from the global vocab rather than
      copy it from evidence, which is exactly where ungrounded targets came from
      (29% of novel edges in the markers run).
    * **many gold edges** (``>= min_edges``) — more chances to miss or mistype one.

    Escalation is the *right* answer on these, so the warmed policy starts with a
    prior that earns reward — otherwise GRPO would learn escalation is bad on
    average and collapse it straight back to zero.
    """
    from kgat.data.chunk_targets import chunk_target_candidates, match_candidate

    gold = list(pair.triples)
    if not gold:
        return False
    if len(gold) >= min_edges:
        return True
    cands = chunk_target_candidates(pair.text, filer=pair.filer)
    return any(match_candidate(t, cands) is None for _, t in gold)


def select_escalate_ids(
    pairs: list[ExtractionPair],
    *,
    min_edges: int = 4,
    extra_random: float = 0.05,
    seed: int = 42,
) -> set[int]:
    """Indices to label ESCALATE: the hard chunks plus a little random coverage.

    ``extra_random`` sprinkles ESCALATE across easy chunks too so the action is seen
    in varied contexts (and the policy does not tie it to one surface cue).
    """
    hard = {i for i, p in enumerate(pairs) if is_hard_chunk(p, min_edges=min_edges)}
    if extra_random > 0:
        rng = random.Random(seed)
        rest = [i for i in range(len(pairs)) if i not in hard]
        rng.shuffle(rest)
        hard |= set(rest[: int(len(rest) * extra_random)])
    return hard


def encode_warmup_example(
    pair: ExtractionPair,
    tokenizer: Any,
    grammar: Any,
    *,
    max_seq_len: int,
    escalate: bool,
    mark_filer: bool = False,
) -> dict[str, list[int]]:
    """Tokenize one warmup pair; target is ESCALATE or the normal triple target."""
    prompt = format_extraction_prompt(
        pair.filer, pair.text, grammar.relations, mark_filer=mark_filer
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if escalate:
        target_ids = list(grammar.enc_sentinel[ESCALATE_LABEL])
    else:
        target_ids = encode_triples_target(pair.triples, grammar)

    room = max_seq_len - len(target_ids)
    if room <= 0:
        raise ValueError(f"max_seq_len={max_seq_len} too small for target")
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def run_routing_warmup(cfg: Any) -> Path:
    """SFT the ESCALATE action into the extractor; returns the adapter dir."""
    require_ml()

    from kgat.train.sft import fit_lora
    from kgat.utils.paths import resolve_path
    from kgat.utils.seed import set_seed

    if "routing_warmup" not in cfg.train:
        raise ValueError("warmup config missing — run with train=routing_warmup")
    w = cfg.train.routing_warmup
    set_seed(int(cfg.seed))

    data_dir = resolve_path(w.data_dir)
    pairs = read_pairs_jsonl(data_dir / f"{w.split}.jsonl", max_examples=w.get("max_examples"))
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    output_dir = resolve_path(w.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally CONTINUE an existing extractor adapter instead of starting a fresh
    # LoRA on the bare base. Measured 2026-07-19: training ESCALATE from base costs
    # ~0.082 F1 of extraction quality (0.549 vs the markers extractor's 0.631,
    # reproduced across two seeds) — the model spends capacity relearning extraction
    # it could have inherited. Warm-starting keeps that quality and teaches only the
    # routing action on top.
    init_adapter = w.get("init_adapter", None)
    adapter_path = str(resolve_path(init_adapter)) if init_adapter else None

    model, tokenizer, device = load_causal_lm(
        cfg.model.hf_id,
        adapter_path=adapter_path,
        device=cfg.get("device", "auto"),
        four_bit=w.get("four_bit", "auto"),
        train_mode=True,
        gradient_checkpointing=bool(w.get("gradient_checkpointing", False)),
    )
    if adapter_path is None:
        model = attach_lora(
            model, r=int(w.lora_r), alpha=int(w.lora_alpha), dropout=float(w.lora_dropout)
        )
    else:
        print(f"routing warmup: continuing adapter {adapter_path}")
    model.print_trainable_parameters()

    mark_filer = bool(w.get("entity_markers", False))
    # A warm start inherits the extractor's PROMPT contract; a mismatch silently
    # fine-tunes on a shifted distribution (same guard as grpo_routing).
    if adapter_path is not None:
        meta_path = Path(adapter_path) / "extractor_meta.json"
        if meta_path.exists():
            meta_mark = bool(
                json.loads(meta_path.read_text(encoding="utf-8")).get("entity_markers", False)
            )
            if meta_mark != mark_filer:
                raise ValueError(
                    f"entity_markers mismatch: config={mark_filer} but init_adapter "
                    f"{adapter_path} was trained with entity_markers={meta_mark}. Set "
                    f"train.routing_warmup.entity_markers={meta_mark} to match."
                )
    # The ESCALATE sentinel MUST be in the grammar or the target is unencodable.
    grammar = build_triple_grammar(
        vocab["relations"],
        vocab["targets"],
        tokenizer,
        eos_id=tokenizer.eos_token_id,
        max_triples=int(w.max_triples),
        sentinels=(ESCALATE_LABEL,),
    )
    esc_ids = select_escalate_ids(
        pairs,
        min_edges=int(w.get("hard_min_edges", 4)),
        extra_random=float(w.get("extra_random", 0.05)),
        seed=int(cfg.seed),
    )
    encoded = [
        encode_warmup_example(
            p, tokenizer, grammar,
            max_seq_len=int(w.max_seq_len), escalate=(i in esc_ids), mark_filer=mark_filer,
        )
        for i, p in enumerate(pairs)
    ]
    print(
        f"routing warmup: {len(encoded)} pairs, {len(esc_ids)} labelled ESCALATE "
        f"({len(esc_ids) / max(len(pairs), 1):.1%}), entity_markers={mark_filer}"
    )

    result_dir = fit_lora(
        model, tokenizer, device,
        encoded=encoded, sft_cfg=w, output_dir=output_dir, seed=int(cfg.seed),
    )
    (result_dir / "extractor_meta.json").write_text(
        json.dumps(
            {
                "entity_markers": mark_filer,
                "targets_mode": "vocab",
                "routing_warmup": True,
                "escalate_fraction": round(len(esc_ids) / max(len(pairs), 1), 4),
                "init_adapter": adapter_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result_dir


def _main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_routing_warmup(cfg)

    main()


if __name__ == "__main__":
    _main()
