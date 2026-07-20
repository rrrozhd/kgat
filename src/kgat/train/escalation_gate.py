"""Train a standalone ESCALATION GATE: predict the extractor's per-chunk F1.

The gate is a small cross-encoder regressor over ``(filer, chunk)`` that answers
"how badly will the extractor get this chunk wrong?". Escalate the lowest-scoring
chunks. It never touches the extractor, which is the whole point:

* teaching ``ESCALATE`` INTO the extractor costs ~0.08 F1 of extraction and the loss
  is irrecoverable (two seeds + a warm start);
* a joint RL objective trades extraction against routing in both directions, and is
  hackable — GRPO drove extraction to 3.2% of chunks while reward rose;
* a supervised gate has no such attractor and leaves the extractor at full quality.

Target = ``kgat.data.gate_export.chunk_f1``, pinned by test to equal what the
routing reward scores. Labels come free from a decode pass over the train split.

CLI::

    python -m kgat.train.escalation_gate train=escalation_gate model=crossencoder-modernbert
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kgat.utils.hf import require_ml

__all__ = ["render_gate_input", "read_gate_jsonl", "load_gate_scorer", "run_gate_training"]


def render_gate_input(filer: str, text: str) -> str:
    """One string for the cross-encoder: who is filing, and the evidence.

    Mirrors ``judge.render_judge_input`` — the filer goes first so it survives the
    tail truncation that long chunks trigger.
    """
    return f"filer: {filer}\n\n{text}"


def read_gate_jsonl(path: str | Path, *, max_examples: int | None = None) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
            if max_examples and len(rows) >= max_examples:
                break
    return rows


def load_gate_scorer(model_dir: str | Path, *, device: str = "auto", batch_size: int = 32):
    """Return ``score(filer, text) -> predicted chunk F1 in [0, 1]`` (batched)."""
    require_ml()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from kgat.utils.hf import pick_device

    dev = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(dev).eval()

    def score_many(items: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            enc = tokenizer(
                [render_gate_input(f, t) for f, t in batch],
                truncation=True, max_length=1024, padding=True, return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                logits = model(**enc).logits.squeeze(-1)
            out.extend(float(max(0.0, min(1.0, v))) for v in logits.tolist())
        return out

    return score_many


def run_gate_training(cfg: Any) -> Path:
    """Fine-tune the gate regressor; returns its output dir."""
    require_ml()

    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from kgat.utils.paths import resolve_path
    from kgat.utils.seed import set_seed

    if "escalation_gate" not in cfg.train:
        raise ValueError("gate config missing — run with train=escalation_gate")
    g = cfg.train.escalation_gate
    set_seed(int(cfg.seed))

    data_dir = resolve_path(g.data_dir)
    train_rows = read_gate_jsonl(data_dir / "train.jsonl", max_examples=g.get("max_examples"))
    dev_rows = read_gate_jsonl(data_dir / "dev.jsonl", max_examples=g.get("max_dev"))
    output_dir = resolve_path(g.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.hf_id, num_labels=1, problem_type="regression"
    )

    def encode(rows):
        feats = []
        for r in rows:
            enc = tokenizer(
                render_gate_input(r.get("filer", ""), r["text"]),
                truncation=True, max_length=int(g.get("max_seq_len", 1024)),
            )
            feats.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": float(r["target"]),
            })
        return feats

    train_feats, dev_feats = encode(train_rows), encode(dev_rows)
    targets = [r["target"] for r in train_rows]
    print(
        f"gate: {len(train_feats)} train / {len(dev_feats)} dev; "
        f"mean target {sum(targets) / max(len(targets), 1):.3f}, "
        f"{sum(1 for t in targets if t <= 0.001) / max(len(targets), 1):.1%} total-miss chunks"
    )

    def metrics(eval_pred):
        preds = np.clip(eval_pred.predictions.squeeze(-1), 0.0, 1.0)
        labels = eval_pred.label_ids
        mae = float(np.mean(np.abs(preds - labels)))
        # Ranking quality is what a gate is actually for — a well-ranked gate with
        # a biased scale still routes correctly, an unbiased one that cannot rank
        # does not. Spearman over the dev split.
        pr = np.argsort(np.argsort(preds))
        lr = np.argsort(np.argsort(labels))
        spearman = float(np.corrcoef(pr, lr)[0, 1]) if len(preds) > 1 else 0.0
        return {"mae": mae, "spearman": spearman}

    from transformers import DataCollatorWithPadding

    args = TrainingArguments(
        output_dir=str(output_dir / "_hf"),
        num_train_epochs=float(g.get("epochs", 2)),
        per_device_train_batch_size=int(g.get("batch_size", 16)),
        gradient_accumulation_steps=int(g.get("grad_accum", 2)),
        learning_rate=float(g.get("learning_rate", 3e-5)),
        logging_steps=20,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_feats, eval_dataset=dev_feats,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    evaluation = trainer.evaluate()
    print(f"gate dev: {evaluation}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "gate_metrics.json").write_text(
        json.dumps(
            {
                "n_train": len(train_feats), "n_dev": len(dev_feats),
                "eval": {k: float(v) for k, v in evaluation.items() if isinstance(v, int | float)},
                "log_history": getattr(getattr(trainer, "state", None), "log_history", []) or [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"gate done -> {output_dir}")
    return output_dir


def _main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path="../../../configs", config_name="config")
    def _run(cfg) -> None:
        run_gate_training(cfg)

    _run()


if __name__ == "__main__":
    _main()
