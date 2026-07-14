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
        use_kv_cache: bool = True,
        seed: int = 0,
        **kwargs: object,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.temperature = float(temperature)
        self.device = device
        self.four_bit = four_bit
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.use_kv_cache = bool(use_kv_cache)
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
        use_kv_cache: bool = True,
        seed: int = 0,
    ) -> DecoderPolicyController:
        """Build a controller around an already-loaded model (used by GRPO)."""
        self = cls(
            model_name="<injected>",
            temperature=temperature,
            max_prompt_tokens=max_prompt_tokens,
            use_kv_cache=use_kv_cache,
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

        from kgat.utils.hf import forward_last_logits

        # KV-cached incremental decoding: prefill the prompt once, then feed one
        # token per step. constrained_decode grows `generated` monotonically by one
        # token per call, so the cache never needs invalidation within a decode.
        # Without this, every generated token re-forwards the whole prompt —
        # ~(prompt_len/gen_len)x wasted FLOPs, the rollout-phase bottleneck.
        cache: dict[str, Any] = {"past": None, "n_fed": 0}

        def _stateless(full_ids: list[int]):
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                return forward_last_logits(model, input_ids, keep=1)[0, -1]

        def _cached(full_ids: list[int]):
            new_ids = full_ids[cache["n_fed"] :]
            input_ids = torch.tensor([new_ids], dtype=torch.long, device=device)
            kwargs: dict[str, Any] = {"use_cache": True}
            if cache["past"] is not None:
                kwargs["past_key_values"] = cache["past"]
            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, logits_to_keep=1, **kwargs)
                except TypeError:  # model doesn't accept logits_to_keep
                    out = model(input_ids=input_ids, **kwargs)
            past = getattr(out, "past_key_values", None)
            if past is None:
                raise RuntimeError("model returned no KV cache")
            cache["past"] = past
            cache["n_fed"] = len(full_ids)
            return out.logits[0, -1]

        def logits_fn(generated: tuple[int, ...], allowed: list[int]):
            full_ids = prompt_ids + list(generated)
            if self.use_kv_cache:
                try:
                    row = _cached(full_ids)
                except (RuntimeError, TypeError, AttributeError, NotImplementedError):
                    # Cache path unsupported for this model/backend — disable for
                    # the controller's lifetime and fall back to full re-forwards.
                    self.use_kv_cache = False
                    cache["past"], cache["n_fed"] = None, 0
                    row = _stateless(full_ids)
            else:
                row = _stateless(full_ids)
            # Structured-decoding contract: gather ONLY the allowed continuations
            # on-device; never materialize the ~151k-vocab row on the CPU.
            idx = torch.tensor(allowed, dtype=torch.long, device=row.device)
            return row.index_select(0, idx).float().cpu().tolist()

        result = constrained_decode(
            logits_fn, trie, id_map, temperature=self.temperature, rng=self._rng
        )

        state.budget.charge(prompt_tokens=len(prompt_ids), gen_tokens=len(result.ids))

        score = math.exp(result.logprob / max(len(result.ids), 1))
        if result.candidate == STOP_TOKEN:
            return Action.stop(score=score)
        return Action.expand(result.candidate, score=score)


__all__ = ["DecoderPolicyController"]
