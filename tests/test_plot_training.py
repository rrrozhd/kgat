"""Training-telemetry loaders (pure parts; plotting itself needs no test).

Mixed-vintage runs are the norm here: adapters trained before the curve was
persisted have no ``log_history``, so the loaders must degrade rather than raise.
"""

from __future__ import annotations

import json

from kgat.eval.plot_training import load_grpo_log, load_sft_curve


def test_sft_curve_reads_log_history(tmp_path):
    p = tmp_path / "sft_metrics.json"
    p.write_text(json.dumps({
        "train_loss": 0.3,
        "log_history": [
            {"loss": 0.9, "step": 10},
            {"loss": 0.5, "step": 20},
            {"train_runtime": 12.0},  # summary row, no loss -> filtered
        ],
    }))
    rows = load_sft_curve(p)
    assert [r["loss"] for r in rows] == [0.9, 0.5]


def test_sft_curve_missing_history_returns_empty(tmp_path):
    """Pre-fix runs saved only the final scalar — must not raise."""
    p = tmp_path / "sft_metrics.json"
    p.write_text(json.dumps({"train_loss": 0.29, "n_examples": 16736}))
    assert load_sft_curve(p) == []


def test_grpo_log_parses_rows_and_skips_blanks(tmp_path):
    p = tmp_path / "grpo_routing_log.jsonl"
    p.write_text(
        json.dumps({"update": 1, "mean_reward": 0.6, "mean_escalated": 0.0}) + "\n"
        + "\n"
        + json.dumps({"update": 2, "mean_reward": 0.7, "mean_escalated": 1.5}) + "\n"
    )
    rows = load_grpo_log(p)
    assert len(rows) == 2
    assert [r["update"] for r in rows] == [1, 2]


def test_grpo_log_empty_file(tmp_path):
    p = tmp_path / "grpo_routing_log.jsonl"
    p.write_text("")
    assert load_grpo_log(p) == []
