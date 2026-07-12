"""Decoder-based traversal controller (Arch A flagship).

A small causal LM (Qwen3-0.6B primary) that, given the serialized question +
frontier, generates either a relation drawn from the candidate set or ``[STOP]``.
Generation is constrained by the token trie in ``constrained_decoding`` — the model
physically cannot emit an off-candidate relation.

The model loads lazily on first ``select`` (importing this module needs no ML deps)
and can also be injected pre-loaded via ``from_model`` (the GRPO trainer does this
so rollouts and gradient passes share one model). Token usage is charged to the
traversal's ``BudgetLedger`` so the cost axis reflects real prompt/gen tokens.

``temperature=0`` decodes greedily (evaluation); ``temperature>0`` samples
(GRPO rollouts). ``Action.score`` is ``exp(mean token logprob)`` in [0, 1] — the
model's per-token confidence in its choice, comparable across candidates of
different token lengths.
"""

from __future__ import annotations

import math
import random
from typing import Any

from kgat.controller.base import TraversalController
from kgat.controller.constrained_decoding import (
    STOP_TOKEN,
    build_relation_trie,
    constrained_decode,
)
from kgat.controller.prompting import format_prompt
from kgat.data.schemas import Action, Relation, TraversalState


class DecoderPolicyController(TraversalController):
    """Constrained-decoding traversal policy over candidate relations."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        temperature: float = 0.0,
        device: str = "auto",
        four_bit: str | bool = "auto",
        max_prompt_tokens: int = 1024,
        seed: int = 0,
        **kwargs: object,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.temperature = float(temperature)
        self.device = device
        self.four_bit = four_bit
        self.max_prompt_tokens = int(max_prompt_tokens)
        self._rng = random.Random(seed)
        self._model: Any = None
        self._tokenizer: Any = None
        self._device_str: str | None = None
        self._extra = kwargs

    @classmethod
    def from_model(
        cls,
        model: Any,
        tokenizer: Any,
        *,
        device: str,
        temperature: float = 0.0,
        max_prompt_tokens: int = 1024,
        seed: int = 0,
    ) -> DecoderPolicyController:
        """Build a controller around an already-loaded model (used by GRPO)."""
        self = cls(
            model_name="<injected>",
            temperature=temperature,
            max_prompt_tokens=max_prompt_tokens,
            seed=seed,
        )
        self._model, self._tokenizer, self._device_str = model, tokenizer, device
        return self

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from kgat.utils.hf import load_causal_lm

            self._model, self._tokenizer, self._device_str = load_causal_lm(
                self.model_name,
                adapter_path=self.adapter_path,
                device=self.device,
                four_bit=self.four_bit,
            )

    def build_prompt(self, state: TraversalState, candidates: list[Relation]) -> str:
        from kgat.traversal.engine import serialize_state

        return format_prompt(serialize_state(state), candidates)

    def select(self, state: TraversalState, candidates: list[Relation]) -> Action:
        self._ensure_loaded()
        import torch

        tok = self._tokenizer
        prompt = self.build_prompt(state, candidates)
        prompt_ids: list[int] = tok.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > self.max_prompt_tokens:  # keep the tail ("...next:")
            prompt_ids = prompt_ids[-self.max_prompt_tokens :]

        trie, id_map = build_relation_trie(candidates, tok, end_id=tok.eos_token_id)

        model, device = self._model, self._device_str
        base = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        from kgat.utils.hf import forward_last_logits

        def logits_fn(generated: tuple[int, ...]):
            if generated:
                gen = torch.tensor([list(generated)], dtype=torch.long, device=device)
                input_ids = torch.cat([base, gen], dim=1)
            else:
                input_ids = base
            with torch.no_grad():
                logits = forward_last_logits(model, input_ids, keep=1)[0, -1]
            return logits.float().cpu().tolist()

        result = constrained_decode(
            logits_fn, trie, id_map, temperature=self.temperature, rng=self._rng
        )

        state.budget.charge(prompt_tokens=len(prompt_ids), gen_tokens=len(result.ids))

        score = math.exp(result.logprob / max(len(result.ids), 1))
        if result.candidate == STOP_TOKEN:
            return Action.stop(score=score)
        return Action.expand(result.candidate, score=score)


__all__ = ["DecoderPolicyController"]
