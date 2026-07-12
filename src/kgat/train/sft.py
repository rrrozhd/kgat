"""Supervised fine-tuning of the traversal controller (M4).

LoRA/QLoRA SFT of the decoder controller on mined trajectories: the model learns to
emit the teacher's next relation (or ``[STOP]``) given the serialized state +
candidates. Loss is completion-only — prompt tokens are masked to ``-100`` so only
the action tokens are trained, matching what constrained decoding scores at
inference.

Built on the plain ``transformers.Trainer`` + ``peft`` (not trl's SFTTrainer) to
minimize version-churn risk. QLoRA (4-bit base) activates automatically on CUDA
with bitsandbytes; on MPS/CPU it falls back to fp16/fp32 LoRA — slow but correct,
which is what makes the tiny-model smoke test runnable anywhere.

CLI::

    python -m kgat.train.sft train=sft dataset=webqsp model=qwen3-0.6b
    # data_path defaults to the mining output for the selected dataset/split
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kgat.controller.constrained_decoding import target_text
from kgat.utils.hf import attach_lora, load_causal_lm, require_ml


def read_sft_examples(path: str | Path, max_examples: int | None = None) -> list[dict]:
    """Load mined SFT examples ({"prompt", "target", ...} per line)."""
    examples: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_examples is not None and len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError(f"no SFT examples in {path}; run kgat.train.mine_trajectories first")
    return examples


def encode_example(example: dict, tokenizer: Any, *, max_seq_len: int) -> dict[str, list[int]]:
    """Tokenize one example with completion-only labels.

    ``input_ids = prompt + target + eos``; ``labels`` mask the prompt with -100.
    Overlong prompts are truncated from the LEFT so the decision-relevant tail
    ("... candidates ... next:") and the full target always survive.
    """
    prompt_ids = tokenizer.encode(example["prompt"], add_special_tokens=False)
    target_ids = tokenizer.encode(target_text(example["target"]), add_special_tokens=False)
    target_ids = target_ids + [tokenizer.eos_token_id]

    room = max_seq_len - len(target_ids)
    if room <= 0:
        raise ValueError(f"max_seq_len={max_seq_len} too small for target {example['target']!r}")
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def run_sft(cfg: Any) -> Path:
    """Train the LoRA adapter and return its output directory."""
    require_ml()
    import torch
    from transformers import Trainer, TrainingArguments

    from kgat.utils.paths import resolve_path
    from kgat.utils.seed import set_seed

    if "sft" not in cfg.train:
        raise ValueError("SFT config missing — run with the train=sft override")
    sft = cfg.train.sft
    set_seed(int(cfg.seed))

    data_path = resolve_path(sft.data_path)
    examples = read_sft_examples(data_path, max_examples=sft.get("max_examples"))
    output_dir = resolve_path(sft.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_causal_lm(
        cfg.model.hf_id,
        device=cfg.get("device", "auto"),
        four_bit=sft.get("four_bit", "auto"),
        train_mode=True,
        gradient_checkpointing=bool(sft.get("gradient_checkpointing", False)),
    )
    model = attach_lora(
        model, r=int(sft.lora_r), alpha=int(sft.lora_alpha), dropout=float(sft.lora_dropout)
    )
    model.print_trainable_parameters()

    max_seq_len = int(sft.max_seq_len)
    encoded = [encode_example(ex, tokenizer, max_seq_len=max_seq_len) for ex in examples]
    print(f"SFT: {len(encoded)} examples from {data_path} (device={device})")

    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = width - n
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attention.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=int(sft.batch_size),
        gradient_accumulation_steps=int(sft.grad_accum),
        num_train_epochs=float(sft.epochs),
        learning_rate=float(sft.learning_rate),
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=int(cfg.seed),
        bf16=use_bf16,
        fp16=(device == "cuda" and not use_bf16),  # T4 has no bf16; fp32 would OOM
        remove_unused_columns=False,
        use_cpu=(device == "cpu"),
    )
    trainer = Trainer(model=model, args=args, train_dataset=encoded, data_collator=collate)
    result = trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = {"train_loss": result.training_loss, "n_examples": len(encoded)}
    (output_dir / "sft_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"SFT done: loss={result.training_loss:.4f}; adapter -> {output_dir}")
    return output_dir


def _main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_sft(cfg)

    main()


if __name__ == "__main__":
    _main()


__all__ = ["run_sft", "read_sft_examples", "encode_example"]
