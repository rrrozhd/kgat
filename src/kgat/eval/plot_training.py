"""Plot training telemetry — SFT loss curves and GRPO routing runs.

Companion to ``eval.frontier`` (which plots the cost/quality frontier); this plots
the RUNS that produce those points, so a training pathology is visible rather than
inferred from a final scalar.

Two sources, both written automatically by the trainers:

* ``sft_metrics.json`` — ``log_history`` from the HF Trainer (per-``logging_steps``
  loss/lr/epoch). Saved next to the adapter.
* ``grpo_routing_log.jsonl`` — one row per GRPO update: reward, precision, recall,
  and the ROUTE CENSUS (escalate/skip/extract).

The route census matters: ``escalate-everything`` and ``skip-everything`` are both
reward attractors (see ``backfill_routing.routing_reward``), so a run collapsing into
either is a degenerate policy, not a good one. Plotting the census makes that
collapse visible at a glance instead of hiding behind a healthy-looking reward.

Headless ``Agg`` backend (set before pyplot import), matching ``eval.frontier``.

CLI::

    python -m kgat.eval.plot_training --sft outputs/adapters/*/sft_metrics.json \\
      --grpo outputs/routing/*.jsonl --out-dir outputs/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["load_sft_curve", "load_grpo_log", "plot_sft_losses", "plot_grpo_run", "main"]


def load_sft_curve(path: str | Path) -> list[dict]:
    """Per-step rows from an ``sft_metrics.json`` ``log_history``.

    Older runs saved only the final scalar (no ``log_history``); those return []
    rather than raising, so a mixed set of runs still plots what it can.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    hist = data.get("log_history") or []
    return [r for r in hist if "loss" in r]


def load_grpo_log(path: str | Path) -> list[dict]:
    """Rows from a ``grpo_routing_log.jsonl`` (one per update)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def plot_sft_losses(curves: dict[str, list[dict]], out_path: str | Path) -> Path:
    """Overlay SFT loss curves (x = optimizer step, or epoch when steps absent)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = 0
    for label, rows in curves.items():
        if not rows:
            continue
        xs = [r.get("step", r.get("epoch", i)) for i, r in enumerate(rows)]
        ys = [r["loss"] for r in rows]
        ax.plot(xs, ys, label=label, linewidth=1.4)
        plotted += 1
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title("SFT loss")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no log_history in these runs", ha="center", transform=ax.transAxes)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_grpo_run(rows: list[dict], out_path: str | Path, *, label: str = "") -> Path:
    """Reward/quality panel + route-census panel for one GRPO run."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    x = [r.get("update", i) for i, r in enumerate(rows)]

    for key, name in (
        ("mean_reward", "reward"),
        ("mean_precision", "precision"),
        ("mean_recall", "recall"),
    ):
        ys = [r.get(key) for r in rows]
        if any(v is not None for v in ys):
            ax1.plot(x, ys, label=name, linewidth=1.4)
    lam = rows[0].get("lam") if rows else None
    ax1.set_ylabel("reward / quality")
    ax1.set_title(f"GRPO routing{f' — {label}' if label else ''}"
                  f"{f'  (lam={lam})' if lam is not None else ''}")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    # Route census — the degenerate-collapse tripwire.
    for key, name in (
        ("mean_escalated", "escalate"),
        ("mean_extract", "extract"),
        ("mean_skip", "skip"),
    ):
        ys = [r.get(key) for r in rows]
        if any(v is not None for v in ys):
            ax2.plot(x, ys, label=name, linewidth=1.4)
    ax2.set_xlabel("update")
    ax2.set_ylabel("chunks / episode")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.text(
        0.99, 0.02,
        "collapse to one route = degenerate policy",
        transform=ax2.transAxes, ha="right", va="bottom", fontsize=7, alpha=0.6,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Plot SFT loss curves and GRPO routing runs.")
    p.add_argument("--sft", nargs="*", default=[], help="sft_metrics.json paths")
    p.add_argument("--grpo", nargs="*", default=[], help="grpo_routing_log.jsonl paths")
    p.add_argument("--out-dir", default="outputs/plots")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    written: list[Path] = []

    if args.sft:
        curves = {}
        for f in args.sft:
            rows = load_sft_curve(f)
            # label by the adapter dir, not the filename (all are sft_metrics.json)
            curves[Path(f).parent.name] = rows
            if not rows:
                print(f"note: {f} has no log_history (pre-dates curve saving) — skipped")
        written.append(plot_sft_losses(curves, out_dir / "sft_loss.png"))

    for f in args.grpo:
        rows = load_grpo_log(f)
        if not rows:
            print(f"note: {f} is empty — skipped")
            continue
        name = Path(f).stem if Path(f).stem != "grpo_routing_log" else Path(f).parent.name
        written.append(plot_grpo_run(rows, out_dir / f"grpo_{name}.png", label=name))

    for w in written:
        print(f"wrote {w}")
    if not written:
        print("nothing to plot — pass --sft and/or --grpo paths")


if __name__ == "__main__":
    main()
