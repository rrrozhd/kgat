"""GRPO over the write-path routing policy: per-chunk {skip | extract | escalate}.

DESIGN-BACKFILL.md phase 2, wired onto the existing GRPO machinery
(``kgat.train.grpo``): an episode is one FILING; each rollout routes every chunk
with one grammar-constrained decode (``NONE`` = skip, ``ESCALATE`` sentinel =
escalate, triples = extract) sampled at temperature>0; the episode gets ONE
terminal reward — ``kgat.train.backfill_routing.routing_reward`` (judged
precision + distant recall − λ·tokens) — and Dr. GRPO group advantages over
``group_size`` rollouts of the same filing. The gradient pass recomputes each
chunk-decode's logprob under the SAME grammar mask the rollout sampled from
(``triples_allowed_along``), forms per-chunk importance ratios, and applies the
DAPO-style asymmetric clip. Loss discipline (dr_grpo token-sum over a constant,
backward per chunk) is inherited verbatim from the read path.

The judge is config-selected and never an LLM: ``distant`` (teacher labels),
``rules`` (objective gates + distant anchor), or a distilled-critic model dir
(gates + thresholded cross-encoder — the exceed-the-teacher reward).

Known cold-start limitation (documented, not solved here): the SFT extractor has
never emitted ``ESCALATE``, so its rollout probability starts near zero and the
route is rarely explored. The planned fix is a cheap warm-start SFT pass that
labels the extractor's own low-confidence chunks as ``ESCALATE`` targets
(initializing the policy AT the cascade behavior), or branch-forced groups
(DESIGN-GRAPH-RL.md §C). The smoke run validates mechanics, not routing gains.

CLI::

    python -m kgat.train.grpo_routing train=grpo_routing model=qwen3-0.6b
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kgat.controller.constrained_decoding import (
    TripleGrammar,
    build_triple_grammar,
    decode_triples,
    triples_allowed_along,
)
from kgat.controller.prompting import format_extraction_prompt
from kgat.data.backfill_export import ExtractionPair, read_pairs_jsonl
from kgat.train.backfill_routing import (
    ESCALATE_LABEL,
    QUALITY_F1,
    ROUTE_EXTRACT,
    ROUTE_SKIP,
    decision_from_result,
    per_chunk_rewards,
    routing_reward,
)
from kgat.train.edge_judge import EdgeJudgeFn, make_rule_judge
from kgat.train.grpo import compute_advantages
from kgat.utils.hf import attach_lora, load_causal_lm, require_ml


def group_by_filing(
    pairs: list[ExtractionPair],
    *,
    max_chunks: int | None = None,
    max_filings: int | None = None,
    seed: int = 42,
) -> list[list[ExtractionPair]]:
    """Episodes: one list of chunks per filing (deterministic order and caps).

    ``max_chunks`` caps the episode length by sampling chunks WITHIN a filing
    (wall-clock control — rollout cost is G × chunks × decode); ``max_filings``
    caps the dataset. Pure and seeded: same inputs, same episodes.
    """
    rng = random.Random(seed)
    by_filing: dict[str, list[ExtractionPair]] = defaultdict(list)
    for p in pairs:
        by_filing[p.filing].append(p)
    filings = sorted(by_filing)
    rng.shuffle(filings)
    if max_filings is not None:
        filings = filings[:max_filings]
    episodes = []
    for f in filings:
        chunks = by_filing[f]
        if max_chunks is not None and len(chunks) > max_chunks:
            chunks = rng.sample(chunks, max_chunks)
        episodes.append(chunks)
    return episodes


def build_judge(spec: str, *, threshold: float = 0.5, device: str = "auto") -> EdgeJudgeFn | None:
    """Config-selected reward judge: ``distant`` | ``rules`` | model dir path."""
    if spec == "distant":
        return None
    if spec == "rules":
        return make_rule_judge()
    from kgat.train.judge import load_judge_scorer

    return make_rule_judge(
        type_score=load_judge_scorer(spec, device=device, threshold=threshold)
    )


@dataclass
class EpisodeRollout:
    """One sampled routing of a filing, reduced to what the gradient pass needs."""

    # per chunk: (prompt_ids, gen_ids, lp_old_sum, grammar) — lp_old is the
    # rollout policy's temp-1 grammar-masked SUM logprob (decode_triples.logprob);
    # the grammar rides along so the gradient pass renormalizes over the SAME
    # constraint (chunk-local grammars differ per chunk).
    chunks: list[tuple[list[int], tuple[int, ...], float, TripleGrammar]]
    reward: float  # macro mean of chunk_rewards (logging + group skip check)
    n_escalated: int
    # Route census — the degenerate-policy tripwire. escalate-everything and
    # skip-everything are BOTH reward attractors (see routing_reward's λ* note);
    # logging only escalation would leave a skip-collapse invisible.
    n_skip: int = 0
    n_extract: int = 0
    chunk_rewards: list[float] = field(default_factory=list)
    chunk_advantages: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def rollout_episode(
    pairs: list[ExtractionPair],
    model: Any,
    tokenizer: Any,
    grammars: list[TripleGrammar],
    *,
    device: str,
    temperature: float,
    rng: random.Random,
    max_prompt_tokens: int,
    judge: EdgeJudgeFn | None,
    reward_kwargs: dict,
    mark_filer: bool = False,
) -> EpisodeRollout:
    """Sample one routing of a filing and score it (no grad).

    ``grammars`` aligns with ``pairs`` — a shared grammar repeated (vocab mode)
    or one per chunk (chunk-local candidates).
    """
    import torch

    from kgat.utils.hf import forward_last_logits

    chunk_records = []
    decisions = []
    for pair, grammar in zip(pairs, grammars, strict=True):
        prompt = format_extraction_prompt(
            pair.filer, pair.text, grammar.relations, mark_filer=mark_filer
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:]

        def logits_fn(generated, allowed, _prompt=prompt_ids):
            input_ids = torch.tensor([_prompt + list(generated)], dtype=torch.long, device=device)
            with torch.no_grad():
                row = forward_last_logits(model, input_ids, keep=1)[0, -1]
            idx = torch.tensor(allowed, dtype=torch.long, device=row.device)
            return row.index_select(0, idx).float().cpu().tolist()

        result = decode_triples(logits_fn, grammar, temperature=temperature, rng=rng)
        decisions.append(decision_from_result(result))
        chunk_records.append((prompt_ids, result.ids, result.logprob, grammar))

    chunk_rs, mean_r = per_chunk_rewards(pairs, decisions, judge=judge, **reward_kwargs)
    r = routing_reward(pairs, decisions, judge=judge, **reward_kwargs)  # micro diagnostics
    return EpisodeRollout(
        chunks=chunk_records,
        reward=mean_r,
        n_escalated=r.n_escalated,
        n_skip=sum(1 for d in decisions if d.route == ROUTE_SKIP),
        n_extract=sum(1 for d in decisions if d.route == ROUTE_EXTRACT),
        chunk_rewards=chunk_rs,
        metrics={"precision": r.precision, "recall": r.recall, "cost": r.normalized_cost},
    )


def grammar_logprob(
    model: Any,
    grammar: TripleGrammar,
    prompt_ids: list[int],
    gen_ids: tuple[int, ...],
    *,
    device: str,
):
    """Grammar-masked log-probability of a decode path (grads flow).

    The routing analog of ``grpo.action_logprob``: renormalizes each position
    over exactly the tokens the grammar allowed during the rollout
    (``triples_allowed_along``), so the optimized distribution is the sampled
    one. Returns ``(lp_sum tensor, n_tokens)``.
    """
    import torch

    from kgat.utils.hf import forward_last_logits

    allowed = triples_allowed_along(grammar, gen_ids)
    input_ids = torch.tensor([prompt_ids + list(gen_ids)], dtype=torch.long, device=device)
    logits = forward_last_logits(model, input_ids, keep=len(gen_ids) + 1)[0]

    total = None
    for j, (tok_id, allowed_j) in enumerate(zip(gen_ids, allowed, strict=True)):
        row = logits[j].float()
        allowed_idx = torch.tensor(allowed_j, dtype=torch.long, device=device)
        lse = torch.logsumexp(row[allowed_idx], dim=0)
        lp = row[tok_id] - lse
        total = lp if total is None else total + lp
    return total, len(gen_ids)


def run_grpo_routing(cfg: Any) -> Path:
    """Train the routing policy with filing-level GRPO; returns the adapter dir."""
    require_ml()
    import torch

    from kgat.utils.logging import JSONLLogger
    from kgat.utils.paths import resolve_path
    from kgat.utils.seed import set_seed

    if "grpo_routing" not in cfg.train:
        raise ValueError("routing config missing — run with the train=grpo_routing override")
    g = cfg.train.grpo_routing
    set_seed(int(cfg.seed))
    rng = random.Random(int(cfg.seed))

    data_dir = resolve_path(g.data_dir)
    pairs = read_pairs_jsonl(data_dir / f"{g.split}.jsonl", max_examples=g.get("max_examples"))
    import json as _json

    vocab = _json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    episodes = group_by_filing(
        pairs,
        max_chunks=g.get("max_chunks_per_filing"),
        max_filings=g.get("max_filings"),
        seed=int(cfg.seed),
    )
    print(f"routing GRPO: {len(episodes)} filing episodes from {data_dir}")

    init_adapter = g.get("init_adapter")
    adapter_path = None
    if init_adapter and resolve_path(init_adapter).exists():
        adapter_path = str(resolve_path(init_adapter))
    model, tokenizer, device = load_causal_lm(
        cfg.model.hf_id,
        adapter_path=adapter_path,
        device=cfg.get("device", "auto"),
        four_bit=g.get("four_bit", "auto"),
        train_mode=True,
    )
    if adapter_path is None:
        print("routing GRPO: no extractor adapter found — attaching a fresh LoRA (cold start)")
        model = attach_lora(
            model, r=int(g.lora_r), alpha=int(g.lora_alpha), dropout=float(g.lora_dropout)
        )

    targets_mode = str(g.get("targets_mode", "vocab"))
    mark_filer = bool(g.get("entity_markers", False))
    # The warm-start extractor stamped its prompt settings; the GRPO prompt MUST
    # match or the policy fine-tunes a distribution shifted from the SFT one.
    if adapter_path is not None:
        meta_path = Path(adapter_path) / "extractor_meta.json"
        if meta_path.exists():
            meta_mark = bool(
                _json.loads(meta_path.read_text(encoding="utf-8")).get("entity_markers", False)
            )
            if meta_mark != mark_filer:
                raise ValueError(
                    f"entity_markers mismatch: config={mark_filer} but warm-start adapter "
                    f"{adapter_path} was trained with entity_markers={meta_mark}. Set "
                    f"train.grpo_routing.entity_markers={meta_mark} to match."
                )
    if targets_mode == "chunk":
        from kgat.data.chunk_targets import chunk_target_candidates

        def grammar_for(pair: ExtractionPair) -> TripleGrammar:
            return build_triple_grammar(
                vocab["relations"],
                chunk_target_candidates(pair.text, filer=pair.filer),
                tokenizer,
                eos_id=tokenizer.eos_token_id,
                max_triples=int(g.max_triples),
                sentinels=(ESCALATE_LABEL,),
            )
    elif targets_mode == "vocab":
        shared_grammar = build_triple_grammar(
            vocab["relations"],
            vocab["targets"],
            tokenizer,
            eos_id=tokenizer.eos_token_id,
            max_triples=int(g.max_triples),
            sentinels=(ESCALATE_LABEL,),
        )

        def grammar_for(pair: ExtractionPair) -> TripleGrammar:
            return shared_grammar
    else:
        raise ValueError(f"targets_mode must be 'vocab' or 'chunk', got {targets_mode!r}")
    judge = build_judge(
        str(g.get("judge", "distant")),
        threshold=float(g.get("judge_threshold", 0.5)),
        device=cfg.get("device", "auto"),
    )
    reward_kwargs = {
        "lam": float(g.lam),
        "escalation_cost_tokens": float(g.escalation_cost_tokens),
        "cost_cap_per_chunk": float(g.cost_cap_per_chunk),
        "precision_weight": float(g.precision_weight),
        "quality_mode": str(g.get("quality_mode", QUALITY_F1)),
        "recall_weight": float(g.recall_weight),
    }

    # KL anchor: a frozen copy of the WARM-START adapter (the SFT extractor is
    # the behavior to preserve — disable_adapter would anchor to the bare base).
    kl_coeff = float(g.get("kl_coeff", 0.0))
    can_kl = kl_coeff > 0 and adapter_path is not None and hasattr(model, "load_adapter")
    if can_kl:
        model.load_adapter(adapter_path, adapter_name="ref")
        model.set_adapter("default")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(g.learning_rate))
    output_dir = resolve_path(g.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = JSONLLogger(output_dir / "grpo_routing_log.jsonl")

    group_size = int(g.group_size)
    batch_filings = int(g.get("batch_filings", 2))
    max_prompt_tokens = int(g.get("max_prompt_tokens", 1024))
    loss_norm = str(g.get("loss_norm", "dr_grpo"))
    if loss_norm not in ("dr_grpo", "grpo"):
        raise ValueError(f"loss_norm must be 'dr_grpo' or 'grpo', got {loss_norm!r}")
    scale_rewards = bool(g.get("scale_rewards", False))
    norm_constant = float(g.get("norm_constant", 32.0))
    clip_lo = float(g.get("clip_eps_low", 0.2))
    clip_hi = float(g.get("clip_eps_high", 0.28))
    updates_per_batch = int(g.get("updates_per_batch", 1))
    updates = 0

    for epoch in range(int(g.epochs)):
        order = list(range(len(episodes)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_filings):
            batch = [episodes[i] for i in order[start : start + batch_filings]]

            rollouts: list[EpisodeRollout] = []
            model.eval()
            for ep_pairs in batch:
                ep_grammars = [grammar_for(p) for p in ep_pairs]  # built once per episode
                group = [
                    rollout_episode(
                        ep_pairs,
                        model,
                        tokenizer,
                        ep_grammars,
                        device=device,
                        temperature=float(g.temperature),
                        rng=rng,
                        max_prompt_tokens=max_prompt_tokens,
                        judge=judge,
                        reward_kwargs=reward_kwargs,
                        mark_filer=mark_filer,
                    )
                    for _ in range(group_size)
                ]
                # Per-chunk-position advantages: group members route the SAME
                # chunk list, so the group baseline at each position is a
                # matched-pair comparison — exact per-decision credit, the
                # write-path analog of exact shaping (DESIGN-GRAPH-RL §B).
                n_chunks = len(group[0].chunk_rewards)
                any_signal = False
                for r in group:
                    r.chunk_advantages = [0.0] * n_chunks
                for j in range(n_chunks):
                    advs = compute_advantages(
                        [r.chunk_rewards[j] for r in group], scale_rewards=scale_rewards
                    )
                    if max(abs(a) for a in advs) >= 1e-9:
                        any_signal = True
                    for r, adv in zip(group, advs, strict=True):
                        r.chunk_advantages[j] = adv
                if not any_signal:
                    continue  # every chunk uniform across the group — nothing to learn
                rollouts.extend(group)

            if not rollouts:
                continue

            # KL reference logprobs are constants of the frozen warm-start
            # adapter: compute once per rollout batch (grpo.py convention).
            ref_sums: list[list[float]] | None = None
            if can_kl:
                ref_sums = []
                model.set_adapter("ref")
                with torch.no_grad():
                    for rollout in rollouts:
                        ref_sums.append(
                            [
                                grammar_logprob(
                                    model, chunk_grammar, prompt_ids, gen_ids, device=device
                                )[0].item()
                                for prompt_ids, gen_ids, _, chunk_grammar in rollout.chunks
                            ]
                        )
                model.set_adapter("default")

            # Gradient pass stays in eval mode ON PURPOSE: train() enables LoRA
            # dropout, which perturbs the recomputed logprobs vs the rollout's
            # and makes importance ratios != 1 even on-policy. Grads flow fine
            # in eval; only dropout/batchnorm behavior differs.
            batch_loss = 0.0
            for _ in range(updates_per_batch):
                optimizer.zero_grad()
                batch_loss = 0.0
                for ri, rollout in enumerate(rollouts):
                    for ci, (prompt_ids, gen_ids, lp_old_sum, chunk_grammar) in enumerate(
                        rollout.chunks
                    ):
                        adv = rollout.chunk_advantages[ci]
                        if abs(adv) < 1e-12 and not can_kl:
                            continue  # zero-advantage chunk contributes no gradient
                        lp_sum, n_tok = grammar_logprob(
                            model, chunk_grammar, prompt_ids, gen_ids, device=device
                        )
                        ratio = torch.exp(lp_sum - lp_old_sum)
                        clipped = torch.clamp(ratio, 1.0 - clip_lo, 1.0 + clip_hi)
                        surrogate = torch.minimum(ratio * adv, clipped * adv)
                        step_loss = -surrogate

                        if ref_sums is not None:
                            step_loss = step_loss + kl_coeff * (lp_sum - ref_sums[ri][ci]) / n_tok

                        if loss_norm == "dr_grpo":
                            scale = n_tok / (norm_constant * len(rollouts))
                        else:
                            scale = 1.0 / (len(rollout.chunks) * len(rollouts))
                        (step_loss * scale).backward()
                        batch_loss += float(step_loss.detach()) * (
                            n_tok / norm_constant
                            if loss_norm == "dr_grpo"
                            else 1.0 / len(rollout.chunks)
                        )

                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                updates += 1

            mean_reward = sum(r.reward for r in rollouts) / len(rollouts)
            mean_esc = sum(r.n_escalated for r in rollouts) / len(rollouts)
            mean_skip = sum(r.n_skip for r in rollouts) / len(rollouts)
            mean_extract = sum(r.n_extract for r in rollouts) / len(rollouts)
            mean_prec = sum(r.metrics["precision"] for r in rollouts) / len(rollouts)
            mean_rec = sum(r.metrics["recall"] for r in rollouts) / len(rollouts)
            record = {
                "epoch": epoch,
                "update": updates,
                "n_rollouts": len(rollouts),
                "mean_reward": mean_reward,
                "mean_precision": mean_prec,
                "mean_recall": mean_rec,
                "mean_escalated": mean_esc,
                # degenerate-policy tripwires: escalate-all and skip-all are both
                # reward attractors; watch these collapse toward n_chunks.
                "mean_skip": mean_skip,
                "mean_extract": mean_extract,
                "loss": batch_loss / len(rollouts),
                "lam": float(g.lam),
                "quality_mode": reward_kwargs["quality_mode"],
            }
            log.log(record)
            print(
                f"[grpo_routing] up={updates} reward={mean_reward:+.3f} "
                f"P={mean_prec:.2f} R={mean_rec:.2f} esc={mean_esc:.1f} "
                f"loss={record['loss']:+.4f}"
            )

            if updates % int(g.get("save_every", 20)) == 0:
                model.save_pretrained(str(output_dir))

    # Only the trained adapter — a KL run also holds a frozen "ref" copy, which
    # would otherwise be written alongside and read back as a checkpoint.
    save_kw = {"selected_adapters": ["default"]} if can_kl else {}
    model.save_pretrained(str(output_dir), **save_kw)
    tokenizer.save_pretrained(str(output_dir))
    # Stamp the prompt contract, exactly as sft_extractor/routing_warmup do: the
    # eval (eval.routing_frontier) self-configures from this file, and without it
    # it falls back to its CLI default and silently scores a markers-trained
    # policy on unmarked prompts.
    (output_dir / "extractor_meta.json").write_text(
        _json.dumps(
            {
                "entity_markers": mark_filer,
                "targets_mode": targets_mode,
                "routing_policy": True,
                "lam": float(g.lam),
                "quality_mode": str(g.get("quality_mode", QUALITY_F1)),
                "init_adapter": str(adapter_path) if adapter_path else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.close()
    print(f"routing GRPO done: {updates} updates; adapter -> {output_dir}")
    return output_dir


def _main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_grpo_routing(cfg)

    main()


if __name__ == "__main__":
    _main()


__all__ = [
    "group_by_filing",
    "build_judge",
    "EpisodeRollout",
    "rollout_episode",
    "grammar_logprob",
    "run_grpo_routing",
]
