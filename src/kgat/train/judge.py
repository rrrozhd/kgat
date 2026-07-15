"""Distill alphina's edge critic into a ~150M cross-encoder (the phase-2 judge).

Fine-tunes a sequence-regression encoder (ModernBERT-base by default —
``model=crossencoder-modernbert``) on the logged critic verdicts
(``kgat.data.judge_export``): input is ``render_judge_input(filer, relation,
target, chunk)``, target is the critic's 0-1 faithfulness (rejects are low by the
critic's own instruction). A single scalar head matches the ``EdgeJudgeFn``
contract exactly, so the trained model plugs into the routing reward as
``make_rule_judge(type_score=load_judge_scorer(model_dir))`` — objective gates
outside, distilled type-faithfulness inside, no LLM in the RL loop.

Reported at the end: dev MSE plus accept/reject accuracy at the 0.5 bar — the
number to compare against the LLM critic before trusting the distilled judge as
a reward.

CLI::

    python -m kgat.train.judge train=judge model=crossencoder-modernbert
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kgat.data.judge_export import JudgeExample, read_judge_jsonl, render_judge_input
from kgat.utils.hf import require_ml


def encode_judge_example(
    example: JudgeExample, tokenizer: Any, *, max_seq_len: int
) -> dict[str, list[int] | float]:
    """Tokenize one verdict for regression (truncates the evidence tail)."""
    enc = tokenizer(
        render_judge_input(example.filer, example.relation, example.target, example.text),
        truncation=True,
        max_length=max_seq_len,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": float(example.faithfulness),
    }


def run_judge_training(cfg: Any) -> Path:
    """Train the judge regressor and return its output directory."""
    require_ml()
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from kgat.utils.paths import resolve_path
    from kgat.utils.seed import set_seed

    if "judge" not in cfg.train:
        raise ValueError("judge config missing — run with the train=judge override")
    jcfg = cfg.train.judge
    set_seed(int(cfg.seed))

    data_dir = resolve_path(jcfg.data_dir)
    train_examples = read_judge_jsonl(
        data_dir / "train.jsonl", max_examples=jcfg.get("max_examples")
    )
    dev_examples = read_judge_jsonl(data_dir / "dev.jsonl", max_examples=jcfg.get("max_dev"))
    output_dir = resolve_path(jcfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.hf_id, num_labels=1, problem_type="regression"
    )

    max_seq_len = int(jcfg.max_seq_len)
    train_enc = [
        encode_judge_example(e, tokenizer, max_seq_len=max_seq_len) for e in train_examples
    ]
    dev_enc = [encode_judge_example(e, tokenizer, max_seq_len=max_seq_len) for e in dev_examples]
    n_accept = sum(1 for e in train_examples if e.verdict == "accept")
    print(
        f"judge: {len(train_enc)} train ({n_accept} accept), {len(dev_enc)} dev, "
        f"base={cfg.model.hf_id}"
    )

    pad_id = tokenizer.pad_token_id

    def collate(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, attention, labels = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            attention.append(b["attention_mask"] + [0] * pad)
            labels.append([b["labels"]])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=int(jcfg.batch_size),
        gradient_accumulation_steps=int(jcfg.grad_accum),
        num_train_epochs=float(jcfg.epochs),
        learning_rate=float(jcfg.learning_rate),
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=int(cfg.seed),
        bf16=use_bf16,
        fp16=(torch.cuda.is_available() and not use_bf16),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_enc, data_collator=collate)
    result = trainer.train()

    # Dev report: MSE + accept/reject accuracy at the 0.5 bar.
    preds = trainer.predict(dev_enc).predictions.reshape(-1)
    golds = [e.faithfulness for e in dev_examples]
    verdicts = [e.verdict for e in dev_examples]
    mse = sum((p - g) ** 2 for p, g in zip(preds, golds, strict=True)) / len(golds)
    acc = sum(
        1 for p, v in zip(preds, verdicts, strict=True) if (p >= 0.5) == (v == "accept")
    ) / len(verdicts)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = {
        "train_loss": result.training_loss,
        "dev_mse": float(mse),
        "dev_verdict_accuracy": float(acc),
        "n_train": len(train_enc),
        "n_dev": len(dev_enc),
    }
    (output_dir / "judge_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"judge done: loss={result.training_loss:.4f} dev_mse={mse:.4f} dev_acc={acc:.3f}")
    return output_dir


def load_judge_scorer(model_dir: str | Path, *, device: str = "auto", max_seq_len: int = 1024):
    """Load a trained judge as an ``edge_judge`` ``type_score`` callable.

    Returns ``(pair, relation, target) -> float`` — the bridge into
    ``make_rule_judge(type_score=...)``. Lazy/heavy imports stay inside.
    """
    require_ml()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from kgat.utils.hf import pick_device

    dev = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(dev).eval()

    def type_score(pair: Any, relation: str, target: str) -> float:
        enc = tokenizer(
            render_judge_input(pair.filer, relation, target, pair.text),
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        ).to(dev)
        with torch.no_grad():
            raw = model(**enc).logits.item()
        return max(0.0, min(1.0, float(raw)))

    return type_score


def _main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        run_judge_training(cfg)

    main()


if __name__ == "__main__":
    _main()


__all__ = ["run_judge_training", "encode_judge_example", "load_judge_scorer"]
