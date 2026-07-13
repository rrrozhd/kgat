"""GRPO training of the traversal controller with the cost-penalized reward (M5).

Custom trajectory-level GRPO rather than trl's ``GRPOTrainer``: trl's trainer does
single-turn prompt->completion generation, but our episodes are multi-step
environment interactions (hop, observe new frontier, hop again), with ONE reward at
trajectory end. The loop here is the GRPO essence applied to that setting:

1. **Rollouts** — per question, sample ``group_size`` traversals through the real
   ``TraversalEngine`` with a temperature>0 controller (no grad), recording each
   action's rollout-time logprob.
2. **Rewards** — ``kgat.train.reward.compute_reward`` on each finished trajectory
   (correctness − λ·normalized cost). λ is the frontier-sweep knob.
3. **Group advantages** — reward minus the group mean (Dr. GRPO default;
   ``scale_rewards=true`` restores legacy z-scoring, which carries the documented
   difficulty bias: low-variance groups get upweighted).
4. **Clipped policy gradient** — recompute each taken action's log-probability
   under the *same* trie-masked distribution the rollout sampled from, form the
   step-level importance ratio vs the rollout logprob, and apply a PPO-style
   asymmetric clip (DAPO clip-higher: ``clip_eps_high > clip_eps_low``). With the
   default single update per rollout batch the ratio is ~1 and this reduces to
   REINFORCE; it makes ``updates_per_batch > 1`` (minibatch reuse) sound.
5. **Loss aggregation** — ``loss_norm="dr_grpo"`` (default): token-loss sums over a
   constant divisor (``norm_constant``), removing the 1/len length bias that made
   short trajectories get larger per-token gradients. ``loss_norm="grpo"`` keeps
   the legacy mean-per-token/mean-per-step aggregation for ablation.
6. **KL regularization** (optional) — the k1 estimator ``lp − lp_ref`` on taken
   actions against the adapter-disabled base model, weighted by ``kl_coeff``.

The bias fixes follow docs/LIT-REVIEW-2026-07.md (Dr. GRPO, COLM 2025; DAPO
clip-higher per arXiv 2509.24203, scoped to near-on-policy). GSPO-style
sequence-over-whole-trajectory ratios were reviewed and rejected there.

Status: implemented and unit-consistent with the rollout path (shared trie/prompt
code), but **not yet validated on a GPU run** — smoke-test on Colab before a real
sweep (see notebooks/colab_kgat.ipynb).

CLI::

    python -m kgat.train.grpo dataset=webqsp model=qwen3-0.6b train.grpo.lam=0.1
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kgat.controller.constrained_decoding import (
    STOP_TOKEN,
    allowed_along,
    build_relation_trie,
    target_text,
)
from kgat.controller.decoder_policy import DecoderPolicyController
from kgat.controller.prompting import format_prompt
from kgat.data.schemas import ActionType, Trajectory
from kgat.train.reward import compute_reward
from kgat.utils.hf import attach_lora, load_causal_lm, require_ml


@dataclass
class Rollout:
    """One sampled traversal, reduced to what the gradient pass needs.

    Each step is ``(prompt, candidates, target, lp_old_mean)`` where
    ``lp_old_mean`` is the rollout policy's mean-per-token logprob of the taken
    action — the denominator of the importance ratio. It is recovered from
    ``Action.score`` (the controller records ``exp(mean token logprob)``).
    """

    steps: list[tuple[str, tuple[str, ...], str, float]]
    reward: float
    advantage: float = 0.0


def trajectory_to_rollout(traj: Trajectory, reward: float) -> Rollout:
    steps = []
    for step in traj.steps:
        target = step.action.relation if step.action.type is ActionType.EXPAND else STOP_TOKEN
        lp_old_mean = math.log(max(step.action.score, 1e-12))
        steps.append(
            (format_prompt(step.state_repr, step.candidates), step.candidates, target, lp_old_mean)
        )
    return Rollout(steps=steps, reward=reward)


def compute_advantages(rewards: list[float], *, scale_rewards: bool) -> list[float]:
    """Group-relative advantages.

    ``scale_rewards=False`` (Dr. GRPO, default): reward minus group mean — every
    group contributes gradient proportional to its reward spread. ``True`` restores
    legacy z-scoring, which upweights low-variance (too easy / too hard) groups —
    the documented GRPO difficulty bias. Pure function; unit-tested without torch.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    centered = [r - mean for r in rewards]
    if not scale_rewards:
        return centered
    std = (sum(c * c for c in centered) / n) ** 0.5
    return [c / (std + 1e-6) for c in centered]


def action_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: tuple[str, ...],
    target: str,
    *,
    device: str,
    max_prompt_tokens: int,
):
    """Trie-masked log-probability of ``target`` given ``prompt`` (grads flow).

    Returns ``(lp_sum, n_tokens)``: the SUM of per-token masked logprobs (a tensor
    with grad) and the token count. Callers choose the normalization — dividing by
    ``n_tokens`` here is exactly the Dr. GRPO length bias, so this function no
    longer takes that decision.

    Renormalizes each position's logits over exactly the tokens the constraint
    allowed during the rollout, so the optimized distribution is the one that was
    sampled from. Prompt truncation mirrors the controller's (left-side).

    Memory: only the last ``len(target)+1`` positions' logits are materialized
    (``forward_last_logits``) — full-sequence logits over Qwen3's ~151k vocab are
    what OOMs a T4.
    """
    import torch

    from kgat.utils.hf import forward_last_logits

    trie, _ = build_relation_trie(list(candidates), tokenizer, end_id=tokenizer.eos_token_id)
    target_ids = tokenizer.encode(target_text(target), add_special_tokens=False)
    target_ids = target_ids + [tokenizer.eos_token_id]
    allowed = allowed_along(trie, target_ids)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]

    input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)
    # keep = |target|+1 tail positions; sliced[j] is the logit row that predicts
    # target_ids[j] (absolute position len(prompt)-1+j).
    logits = forward_last_logits(model, input_ids, keep=len(target_ids) + 1)[0]

    total = None
    for j, (tok_id, allowed_j) in enumerate(zip(target_ids, allowed, strict=True)):
        row = logits[j].float()
        allowed_idx = torch.tensor(allowed_j, dtype=torch.long, device=device)
        lse = torch.logsumexp(row[allowed_idx], dim=0)
        lp = row[tok_id] - lse
        total = lp if total is None else total + lp
    return total, len(target_ids)


def run_grpo(cfg: Any) -> Path:
    """Train with trajectory-level GRPO; returns the adapter output dir."""
    require_ml()
    import torch

    from kgat.data.loaders import load_records
    from kgat.graph.inmemory import InMemoryKGStore
    from kgat.synthesis.base import DummySynthesizer
    from kgat.traversal.budget import BudgetCaps
    from kgat.traversal.engine import TraversalEngine
    from kgat.utils.logging import JSONLLogger
    from kgat.utils.paths import resolve_path
    from kgat.utils.seed import set_seed

    if "grpo" not in cfg.train:
        raise ValueError("GRPO config missing — run with train=grpo (the default)")
    g = cfg.train.grpo
    set_seed(int(cfg.seed))
    rng = random.Random(int(cfg.seed))

    records = load_records(
        resolve_path(cfg.dataset.data_dir),
        split=cfg.dataset.split,
        dataset=cfg.dataset.name,
        limit=g.get("max_questions") or cfg.dataset.get("limit"),
    )
    store = InMemoryKGStore.from_records(records)

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
        print("GRPO: no SFT adapter found — attaching a fresh LoRA (cold start)")
        model = attach_lora(
            model, r=int(g.lora_r), alpha=int(g.lora_alpha), dropout=float(g.lora_dropout)
        )

    max_prompt_tokens = int(g.get("max_prompt_tokens", 1024))
    controller = DecoderPolicyController.from_model(
        model,
        tokenizer,
        device=device,
        temperature=float(g.temperature),
        max_prompt_tokens=max_prompt_tokens,
        seed=int(cfg.seed),
    )
    engine = TraversalEngine(
        store,
        controller,
        DummySynthesizer(),  # frontier tails; the reader is for eval, not RL speed
        budget_caps=BudgetCaps.from_config(dict(cfg.budget)),
        beam_size=int(cfg.engine.beam_size),
        max_steps=int(cfg.engine.max_steps),
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(g.learning_rate))
    kl_coeff = float(g.get("kl_coeff", 0.0))
    can_kl = kl_coeff > 0 and hasattr(model, "disable_adapter")

    output_dir = resolve_path(g.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = JSONLLogger(output_dir / "grpo_log.jsonl")

    group_size = int(g.group_size)
    batch_q = int(g.get("batch_questions", 4))
    loss_norm = str(g.get("loss_norm", "dr_grpo"))
    if loss_norm not in ("dr_grpo", "grpo"):
        raise ValueError(f"loss_norm must be 'dr_grpo' or 'grpo', got {loss_norm!r}")
    scale_rewards = bool(g.get("scale_rewards", False))
    norm_constant = float(g.get("norm_constant", 32.0))  # dr_grpo divisor; interacts with LR
    clip_lo = float(g.get("clip_eps_low", 0.2))
    clip_hi = float(g.get("clip_eps_high", 0.28))  # DAPO clip-higher: hi > lo
    updates_per_batch = int(g.get("updates_per_batch", 1))
    if float(g.temperature) != 1.0 and updates_per_batch > 1:
        # Recorded rollout logprobs are temp-1; the IS ratio is only exact at T=1.
        print(
            "[grpo] WARNING: temperature != 1.0 with updates_per_batch > 1 — "
            "importance ratios are approximate; prefer temperature=1.0."
        )
    updates = 0

    for epoch in range(int(g.epochs)):
        order = list(range(len(records)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_q):
            batch = [records[i] for i in order[start : start + batch_q]]

            # 1-3: rollouts, rewards, group advantages (no grad).
            rollouts: list[Rollout] = []
            model.eval()
            with torch.no_grad():
                for rec in batch:
                    q = rec.question
                    group: list[Rollout] = []
                    for _ in range(group_size):
                        traj = engine.run(q).trajectory
                        reward = compute_reward(
                            traj.predicted_answers,
                            q.gold_answers,
                            traj.cost,
                            lam=float(g.lam),
                            correctness=str(g.correctness),
                            cost_cap=float(g.cost_cap),
                            cost_axis=str(g.cost_axis),
                        )
                        if traj.steps:
                            group.append(trajectory_to_rollout(traj, reward))
                    if len(group) < 2:
                        continue
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

            # 4-6: clipped policy-gradient updates over the batch's rollouts.
            # Backward per STEP so each forward's graph frees immediately — peak
            # memory is one step's activations regardless of trajectory length.
            # lp_old (rollout policy) stays fixed across the inner updates; the
            # clipped ratio is what makes updates_per_batch > 1 sound.
            model.train()
            batch_loss = 0.0
            for _ in range(updates_per_batch):
                optimizer.zero_grad()
                batch_loss = 0.0
                for rollout in rollouts:
                    n_steps = len(rollout.steps)
                    adv = rollout.advantage
                    for prompt, candidates, target, lp_old_mean in rollout.steps:
                        lp_sum, n_tok = action_logprob(
                            model,
                            tokenizer,
                            prompt,
                            candidates,
                            target,
                            device=device,
                            max_prompt_tokens=max_prompt_tokens,
                        )
                        # Step-level importance ratio vs the rollout policy.
                        lp_old_sum = lp_old_mean * n_tok
                        ratio = torch.exp(lp_sum - lp_old_sum)
                        clipped = torch.clamp(ratio, 1.0 - clip_lo, 1.0 + clip_hi)
                        surrogate = torch.minimum(ratio * adv, clipped * adv)
                        step_loss = -surrogate

                        if can_kl:
                            with torch.no_grad(), model.disable_adapter():
                                ref_sum, _ = action_logprob(
                                    model,
                                    tokenizer,
                                    prompt,
                                    candidates,
                                    target,
                                    device=device,
                                    max_prompt_tokens=max_prompt_tokens,
                                )
                            step_loss = step_loss + kl_coeff * (lp_sum - ref_sum.detach()) / n_tok

                        if loss_norm == "dr_grpo":
                            # Token-weighted sum over a CONSTANT divisor: no 1/len.
                            scale = n_tok / (norm_constant * len(rollouts))
                        else:  # legacy "grpo": mean per step, mean per trajectory
                            scale = 1.0 / (n_steps * len(rollouts))
                        (step_loss * scale).backward()
                        batch_loss += float(step_loss.detach()) * (
                            n_tok / norm_constant if loss_norm == "dr_grpo" else 1.0 / n_steps
                        )

                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                updates += 1

            mean_reward = sum(r.reward for r in rollouts) / len(rollouts)
            mean_len = sum(len(r.steps) for r in rollouts) / len(rollouts)
            record = {
                "epoch": epoch,
                "update": updates,
                "n_rollouts": len(rollouts),
                "mean_reward": mean_reward,
                "mean_steps": mean_len,
                "loss": batch_loss / len(rollouts),
                "loss_norm": loss_norm,
                "lam": float(g.lam),
            }
            log.log(record)
            print(
                f"[grpo] up={updates} reward={mean_reward:+.3f} "
                f"steps={mean_len:.2f} loss={record['loss']:+.4f}"
            )

            if updates % int(g.get("save_every", 50)) == 0:
                model.save_pretrained(str(output_dir))

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.close()
    print(f"GRPO done: {updates} updates; adapter -> {output_dir}")
    return output_dir


def _main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_grpo(cfg)

    main()


if __name__ == "__main__":
    _main()


__all__ = [
    "run_grpo",
    "action_logprob",
    "compute_advantages",
    "trajectory_to_rollout",
    "Rollout",
]
