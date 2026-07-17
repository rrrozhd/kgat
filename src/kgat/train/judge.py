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

from kgat.data.judge_export import (
    JudgeExample,
    mint_direction_negatives,
    read_judge_jsonl,
    render_judge_input,
)
from kgat.utils.hf import require_ml


def verdict_agreement(preds, verdicts, *, threshold: float = 0.5) -> dict[str, float]:
    """Accept/reject agreement at ``threshold``, overall AND per class.

    The aggregate hides asymmetry: a judge weak on REJECTS under-protects the
    graph even at high overall agreement, so both class rates are first-class
    metrics. Empty classes report 1.0 (vacuous).
    """
    pairs = list(zip(preds, verdicts, strict=True))
    if not pairs:
        raise ValueError("no predictions to score")

    def rate(cls: str) -> float:
        cls_pairs = [(p, v) for p, v in pairs if v == cls]
        if not cls_pairs:
            return 1.0
        return sum(1 for p, v in cls_pairs if (p >= threshold) == (v == "accept")) / len(cls_pairs)

    overall = sum(1 for p, v in pairs if (p >= threshold) == (v == "accept")) / len(pairs)
    return {
        "dev_verdict_accuracy": overall,
        "dev_accept_agreement": rate("accept"),
        "dev_reject_agreement": rate("reject"),
    }


def threshold_sweep(preds, verdicts, thresholds=None) -> list[dict[str, float]]:
    """Per-class agreement at each candidate threshold — the free operating-point knob.

    The regression head makes the decision bar post-hoc tunable: sweeping it on
    dev trades accept-agreement for reject-agreement without retraining. Rows
    are dicts with ``threshold`` + the ``verdict_agreement`` fields, plus
    ``balanced`` (mean of the two class rates) for picking a symmetric point.
    """
    rows = []
    for t in thresholds if thresholds is not None else [round(0.05 * i, 2) for i in range(1, 20)]:
        m = verdict_agreement(preds, verdicts, threshold=t)
        rows.append(
            {
                "threshold": t,
                **m,
                "balanced": (m["dev_accept_agreement"] + m["dev_reject_agreement"]) / 2,
            }
        )
    return rows


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

    # Direction-flipped hard negatives (judge v3): mint wrong-direction rejects from
    # confident accepts to fix the weak reject-agreement axis. TRAIN split ONLY —
    # dev stays a clean held-out mirror of the real critic distribution.
    n_dir_neg = 0
    if bool(jcfg.get("direction_negatives", False)):
        negs = mint_direction_negatives(
            train_examples,
            min_source_faithfulness=float(jcfg.get("dir_neg_min_faith", 0.5)),
        )
        train_examples = train_examples + negs
        n_dir_neg = len(negs)

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
        f"judge: {len(train_enc)} train ({n_accept} accept, +{n_dir_neg} direction-neg), "
        f"{len(dev_enc)} dev, base={cfg.model.hf_id}"
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
    save_steps = int(jcfg.get("save_steps") or 0)
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=int(jcfg.batch_size),
        gradient_accumulation_steps=int(jcfg.grad_accum),
        num_train_epochs=float(jcfg.epochs),
        learning_rate=float(jcfg.learning_rate),
        logging_steps=50,
        # Long runs on interruptible pods keep rolling checkpoints; smoke runs skip.
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        save_total_limit=2,
        report_to=[],
        seed=int(cfg.seed),
        bf16=use_bf16,
        fp16=(torch.cuda.is_available() and not use_bf16),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_enc, data_collator=collate)
    result = trainer.train()

    # Dev report: MSE + accept/reject agreement, overall and per class.
    preds = trainer.predict(dev_enc).predictions.reshape(-1)
    golds = [e.faithfulness for e in dev_examples]
    verdicts = [e.verdict for e in dev_examples]
    mse = sum((p - g) ** 2 for p, g in zip(preds, golds, strict=True)) / len(golds)
    agreement = verdict_agreement([float(p) for p in preds], verdicts)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = {
        "train_loss": result.training_loss,
        "dev_mse": float(mse),
        **{k: float(v) for k, v in agreement.items()},
        "n_train": len(train_enc),
        "n_dev": len(dev_enc),
    }
    (output_dir / "judge_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"judge done: loss={result.training_loss:.4f} dev_mse={mse:.4f} "
        f"acc={agreement['dev_verdict_accuracy']:.3f} "
        f"accept={agreement['dev_accept_agreement']:.3f} "
        f"reject={agreement['dev_reject_agreement']:.3f}"
    )
    return output_dir


def load_judge_scorer(
    model_dir: str | Path,
    *,
    device: str = "auto",
    max_seq_len: int = 1024,
    threshold: float | None = None,
):
    """Load a trained judge as an ``edge_judge`` ``type_score`` callable.

    Returns ``(pair, relation, target) -> float`` — the bridge into
    ``make_rule_judge(type_score=...)``. With ``threshold`` set, the score is
    BINARIZED at the dev-tuned operating point (``threshold_sweep``) — "audited
    precision" semantics: an edge is accepted or it is not. ``None`` keeps the
    raw regression score. Lazy/heavy imports stay inside.
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
        if threshold is not None:
            return 1.0 if raw >= threshold else 0.0
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


__all__ = [
    "run_judge_training",
    "encode_judge_example",
    "verdict_agreement",
    "threshold_sweep",
    "load_judge_scorer",
]
