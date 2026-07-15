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
    decision_from_result,
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

    # per chunk: (prompt_ids, gen_ids, lp_old_sum) — lp_old is the rollout
    # policy's temp-1 grammar-masked SUM logprob (decode_triples.logprob).
    chunks: list[tuple[list[int], tuple[int, ...], float]]
    reward: float
    n_escalated: int
    advantage: float = 0.0
    metrics: dict = field(default_factory=dict)


def rollout_episode(
    pairs: list[ExtractionPair],
    model: Any,
    tokenizer: Any,
    grammar: TripleGrammar,
    *,
    device: str,
    temperature: float,
    rng: random.Random,
    max_prompt_tokens: int,
    judge: EdgeJudgeFn | None,
    reward_kwargs: dict,
) -> EpisodeRollout:
    """Sample one routing of a filing and score it (no grad)."""
    import torch

    from kgat.utils.hf import forward_last_logits

    chunk_records = []
    decisions = []
    for pair in pairs:
        prompt = format_extraction_prompt(pair.filer, pair.text, grammar.relations)
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
        chunk_records.append((prompt_ids, result.ids, result.logprob))

    r = routing_reward(pairs, decisions, judge=judge, **reward_kwargs)
    return EpisodeRollout(
        chunks=chunk_records,
        reward=r.reward,
        n_escalated=r.n_escalated,
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

    grammar = build_triple_grammar(
        vocab["relations"],
        vocab["targets"],
        tokenizer,
        eos_id=tokenizer.eos_token_id,
        max_triples=int(g.max_triples),
        sentinels=(ESCALATE_LABEL,),
    )
    judge = build_judge(
        str(g.get("judge", "distant")),
        threshold=float(g.get("judge_threshold", 0.5)),
        device=cfg.get("device", "auto"),
    )
    reward_kwargs = {
        "lam": float(g.lam),
        "escalation_cost_tokens": float(g.escalation_cost_tokens),
        "cost_cap": float(g.cost_cap),
        "precision_weight": float(g.precision_weight),
        "recall_weight": float(g.recall_weight),
    }

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
                group = [
                    rollout_episode(
                        ep_pairs,
                        model,
                        tokenizer,
                        grammar,
                        device=device,
                        temperature=float(g.temperature),
                        rng=rng,
                        max_prompt_tokens=max_prompt_tokens,
                        judge=judge,
                        reward_kwargs=reward_kwargs,
                    )
                    for _ in range(group_size)
                ]
                advantages = compute_advantages(
                    [r.reward for r in group], scale_rewards=scale_rewards
                )
                if max(abs(a) for a in advantages) < 1e-9:
                    continue  # uniform group — no learning signal
                for r, adv in zip(group, advantages, strict=True):
                    r.advantage = adv
                rollouts.extend(group)

            if not rollouts:
                continue

            model.train()
            batch_loss = 0.0
            for _ in range(updates_per_batch):
                optimizer.zero_grad()
                batch_loss = 0.0
                for rollout in rollouts:
                    for prompt_ids, gen_ids, lp_old_sum in rollout.chunks:
                        lp_sum, n_tok = grammar_logprob(
                            model, grammar, prompt_ids, gen_ids, device=device
                        )
                        ratio = torch.exp(lp_sum - lp_old_sum)
                        clipped = torch.clamp(ratio, 1.0 - clip_lo, 1.0 + clip_hi)
                        adv = rollout.advantage
                        surrogate = torch.minimum(ratio * adv, clipped * adv)
                        step_loss = -surrogate

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
                "loss": batch_loss / len(rollouts),
                "lam": float(g.lam),
            }
            log.log(record)
            print(
                f"[grpo_routing] up={updates} reward={mean_reward:+.3f} "
                f"P={mean_prec:.2f} R={mean_rec:.2f} esc={mean_esc:.1f} "
                f"loss={record['loss']:+.4f}"
            )

            if updates % int(g.get("save_every", 20)) == 0:
                model.save_pretrained(str(output_dir))

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
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
