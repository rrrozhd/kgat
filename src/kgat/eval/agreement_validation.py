"""Canonical validation helpers for grammar-agreement routing experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def legacy_chunk_key(raw_line: bytes) -> str:
    """Return the migration-only stable key for an exact legacy JSONL line."""
    return hashlib.sha256(raw_line).hexdigest()


def budget_order(scores: Sequence[float], tie_keys: Sequence[str], budget: float) -> list[int]:
    """Return the exact lowest-score indices selected at ``budget``.

    Ties are resolved by the stable content-addressed key, never row position.
    """
    if len(scores) != len(tie_keys):
        raise ValueError("scores and tie_keys must have equal length")
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget must be in [0, 1]")
    k = math.floor(budget * len(scores))
    return sorted(range(len(scores)), key=lambda i: (scores[i], tie_keys[i]))[:k]


def _triples(value: Sequence[Sequence[str]]) -> set[tuple[str, str]]:
    return {tuple(item) for item in value}  # type: ignore[misc]


def _metrics(rows: Sequence[Mapping[str, Any]], decisions: Sequence[bool]) -> dict[str, float]:
    tp = fp = fn = exact = 0
    for row, escalate in zip(rows, decisions, strict=True):
        gold = _triples(row.get("gold") or [])
        pred = gold if escalate else _triples(row.get("pred") or [])
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += int(pred == gold)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": exact / len(rows) if rows else 1.0,
    }


def _sha256_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    return hashlib.sha256(payload).hexdigest()


def freeze_confidence_baseline(
    rows: Sequence[Mapping[str, Any]],
    raw_lines: Sequence[bytes],
    *,
    threshold: float,
    input_token_counts: Sequence[int],
    output_token_counts: Sequence[int],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the migration confidence policy and its cost-driving decisions."""
    n = len(rows)
    if not (len(raw_lines) == len(input_token_counts) == len(output_token_counts) == n):
        raise ValueError("rows, raw lines, and token counts must align")
    decisions = [float(row.get("confidence", 1.0)) < threshold for row in rows]
    projection = [
        {
            "gold": row.get("gold") or [],
            "pred": row.get("pred") or [],
            "confidence": round(float(row.get("confidence", 1.0)), 12),
        }
        for row in rows
    ]
    decision_bytes = "".join("1" if value else "0" for value in decisions).encode("ascii")
    escalated = sum(decisions)
    return {
        "policy": {"signal": "confidence", "threshold": threshold},
        "n_chunks": n,
        "legacy_chunk_keys_sha256": _sha256_jsonl(
            [{"key": legacy_chunk_key(line)} for line in raw_lines]
        ),
        "decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "projection_sha256": _sha256_jsonl(projection),
        "escalated_calls": escalated,
        "escalation_rate": escalated / n if n else 0.0,
        "escalated_input_tokens": sum(
            count for count, decision in zip(input_token_counts, decisions, strict=True) if decision
        ),
        "escalated_output_tokens": sum(
            count
            for count, decision in zip(output_token_counts, decisions, strict=True)
            if decision
        ),
        "metrics": _metrics(rows, decisions),
        "legacy_contract": dict(contract),
    }


def validate_reproduction(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    """Return named failed confidence-reproduction gates."""
    failures: list[str] = []
    for field in (
        "decision_sha256",
        "projection_sha256",
        "escalated_calls",
        "legacy_contract",
    ):
        if candidate.get(field) != baseline.get(field):
            failures.append(field)
    for field in ("f1", "recall"):
        if abs(candidate["metrics"][field] - baseline["metrics"][field]) > 0.002:
            failures.append(field)
    if abs(candidate["escalation_rate"] - baseline["escalation_rate"]) > 0.002:
        failures.append("escalation_rate")
    for field in ("escalated_input_tokens", "escalated_output_tokens"):
        reference = baseline[field]
        tolerance = 0.005 * reference + 1e-12
        if abs(candidate[field] - reference) > tolerance:
            failures.append(field)
    return failures


def exact_frontier(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float], tie_keys: Sequence[str]
) -> list[dict[str, Any]]:
    """Return every stable ranked escalation point, including zero and all."""
    if not (len(rows) == len(scores) == len(tie_keys)):
        raise ValueError("rows, scores, and tie_keys must align")
    order = sorted(range(len(rows)), key=lambda i: (scores[i], tie_keys[i]))
    points: list[dict[str, Any]] = []
    decisions = [False] * len(rows)
    for k in range(len(rows) + 1):
        if k:
            decisions[order[k - 1]] = True
        metrics = _metrics(rows, decisions)
        points.append(
            {
                "escalated_calls": k,
                "escalation_rate": k / len(rows) if rows else 0.0,
                **metrics,
            }
        )
    return points


def routing_auroc(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> float | None:
    """AUROC for low signal detecting a non-exact extractor prediction."""
    if len(rows) != len(scores):
        raise ValueError("rows and scores must align")
    errors: list[float] = []
    correct: list[float] = []
    for row, score in zip(rows, scores, strict=True):
        target = (
            errors
            if _triples(row.get("gold") or []) != _triples(row.get("pred") or [])
            else correct
        )
        target.append(float(score))
    if not errors or not correct:
        return None
    wins = 0.0
    for error in errors:
        for good in correct:
            if error < good:
                wins += 1.0
            elif error == good:
                wins += 0.5
    return wins / (len(errors) * len(correct))


def signal_verdict(
    reproduction_failures: Sequence[str],
    confidence: Mapping[str, Mapping[str, float]],
    candidates: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> str:
    """Apply the deterministic nearby-budget signal gate."""
    if reproduction_failures:
        return "REPRODUCIBILITY_FAIL"
    for report in candidates.values():
        improved = False
        passed = True
        for budget in ("0.15", "0.20", "0.25"):
            base = confidence[budget]
            row = report[budget]
            if row["recall"] - base["recall"] < -0.002:
                passed = False
            if row["f1"] - base["f1"] < -0.002:
                passed = False
            if row["exact"] - base["exact"] < -0.005:
                passed = False
            improved |= row["recall"] - base["recall"] >= 0.005
            improved |= row["f1"] - base["f1"] >= 0.005
        if passed and improved:
            return "ECONOMICS_READY"
    return "SIGNAL_NO_GO"


def _bucket_metrics(
    rows: Sequence[Mapping[str, Any]], decisions: Sequence[bool], indices: Sequence[int]
) -> dict[str, Any]:
    subset_rows = [rows[i] for i in indices]
    subset_decisions = [decisions[i] for i in indices]
    return {
        "n": len(indices),
        "gating": len(indices) >= 30,
        **_metrics(subset_rows, subset_decisions),
    }


def bucket_report(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    tie_keys: Sequence[str],
    *,
    budget: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Report predeclared validation buckets at one exact escalation budget."""
    if not (len(rows) == len(pairs) == len(scores) == len(tie_keys)):
        raise ValueError("bucket inputs must align")
    selected = set(budget_order(scores, tie_keys, budget))
    decisions = [i in selected for i in range(len(rows))]

    buckets: dict[str, dict[str, list[int]]] = {
        "positive": defaultdict(list),
        "relation": defaultdict(list),
        "item_key": defaultdict(list),
        "chunk_length_quartile": defaultdict(list),
        "filer_volume_decile": defaultdict(list),
    }
    lengths = sorted((len(str(pair.get("text", ""))), i) for i, pair in enumerate(pairs))
    length_bucket = {
        i: f"q{min(3, rank * 4 // max(1, len(rows))) + 1}" for rank, (_, i) in enumerate(lengths)
    }
    filer_counts = Counter(str(pair.get("filer", "")) for pair in pairs)
    filer_order = {
        name: rank
        for rank, (name, _) in enumerate(sorted(filer_counts.items(), key=lambda x: (x[1], x[0])))
    }

    for i, (row, pair) in enumerate(zip(rows, pairs, strict=True)):
        gold = _triples(row.get("gold") or [])
        buckets["positive"][str(bool(gold)).lower()].append(i)
        for relation in sorted({triple[0] for triple in gold}):
            buckets["relation"][relation].append(i)
        buckets["item_key"][str(pair.get("item_key", ""))].append(i)
        buckets["chunk_length_quartile"][length_bucket[i]].append(i)
        filer = str(pair.get("filer", ""))
        decile = min(9, filer_order[filer] * 10 // max(1, len(filer_order))) + 1
        buckets["filer_volume_decile"][f"d{decile}"].append(i)

    return {
        dimension: {
            name: _bucket_metrics(rows, decisions, indices)
            for name, indices in sorted(groups.items())
        }
        for dimension, groups in buckets.items()
    }


def _budget_metrics(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    tie_keys: Sequence[str],
    budget: float,
) -> dict[str, Any]:
    selected = set(budget_order(scores, tie_keys, budget))
    decisions = [i in selected for i in range(len(rows))]
    return {
        "escalated_calls": len(selected),
        "escalation_rate": len(selected) / len(rows) if rows else 0.0,
        **_metrics(rows, decisions),
    }


def comparison_report(
    baseline: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    raw_lines: Sequence[bytes],
    *,
    input_token_counts: Sequence[int],
    output_token_counts: Sequence[int],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete canonical reproduction and signal report."""
    if not (len(rows) == len(pairs) == len(raw_lines)):
        raise ValueError("comparison inputs must align")
    candidate_baseline = freeze_confidence_baseline(
        rows,
        raw_lines,
        threshold=float(baseline["policy"]["threshold"]),
        input_token_counts=input_token_counts,
        output_token_counts=output_token_counts,
        contract=baseline["legacy_contract"],
    )
    failures = validate_reproduction(baseline, candidate_baseline)
    failures.extend(validate_run_manifest(manifest, expected_pairs=len(rows)))
    failures = sorted(set(failures))

    tie_keys = [legacy_chunk_key(line) for line in raw_lines]
    signal_values: dict[str, list[float]] = {
        "confidence": [float(row.get("confidence", 1.0)) for row in rows],
        "agreement": [float(row["agreement"]) for row in rows],
        "min_agreement": [float(row["min_agreement"]) for row in rows],
    }
    signal_values["conf_x_agree"] = [
        confidence * agreement
        for confidence, agreement in zip(
            signal_values["confidence"], signal_values["agreement"], strict=True
        )
    ]
    signal_values["conf_x_min_agree"] = [
        confidence * agreement
        for confidence, agreement in zip(
            signal_values["confidence"], signal_values["min_agreement"], strict=True
        )
    ]

    signals: dict[str, dict[str, Any]] = {}
    for name, scores in signal_values.items():
        budgets = {
            f"{budget:.2f}": _budget_metrics(rows, scores, tie_keys, budget)
            for budget in (0.15, 0.20, 0.25)
        }
        signals[name] = {
            "auroc": routing_auroc(rows, scores),
            "budgets": budgets,
            "exact_frontier": exact_frontier(rows, scores, tie_keys),
            "buckets_at_20pct": bucket_report(rows, pairs, scores, tie_keys, budget=0.20),
        }
    verdict = signal_verdict(
        failures,
        signals["confidence"]["budgets"],
        {name: value["budgets"] for name, value in signals.items() if name != "confidence"},
    )
    return {
        "verdict": verdict,
        "reproduction_failures": failures,
        "baseline": dict(baseline),
        "candidate_confidence": candidate_baseline,
        "signals": signals,
    }


_PAIRS_RE = re.compile(r"cascade eval: (\d+) \w+ pairs, (\d+) targets")
_PROMPT_RE = re.compile(
    r"prompt tokens: mean=([\d.]+) max=(\d+) truncated=(\d+)/(\d+) \(mark_filer=(True|False)\)"
)


def build_run_manifest(
    log_text: str,
    *,
    config: Mapping[str, Any],
    hashes: Mapping[str, str],
    environment: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    exit_code: int,
) -> dict[str, Any]:
    """Build the structured manifest the evaluator itself does not emit."""
    pair_match = _PAIRS_RE.search(log_text)
    prompt_match = _PROMPT_RE.search(log_text)
    if pair_match is None or prompt_match is None:
        raise ValueError("evaluator log is missing pair/target or prompt statistics")
    return {
        "exit_code": exit_code,
        "n_pairs": int(pair_match.group(1)),
        "n_targets": int(pair_match.group(2)),
        "prompt": {
            "mean_tokens": round(float(prompt_match.group(1))),
            "max_tokens": int(prompt_match.group(2)),
            "truncated": int(prompt_match.group(3)),
            "total": int(prompt_match.group(4)),
            "mark_filer": prompt_match.group(5) == "True",
        },
        "config": dict(config),
        "hashes": dict(hashes),
        "environment": dict(environment),
        "deterministic": dict(deterministic),
    }


def _required_path(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def validate_run_manifest(manifest: Mapping[str, Any], *, expected_pairs: int = 2193) -> list[str]:
    """Fail-closed validation of the canonical explicit run contract."""
    failures: list[str] = []
    exact = {
        "exit_code": 0,
        "n_pairs": expected_pairs,
        "n_targets": 8993,
        "prompt.truncated": 0,
        "prompt.total": expected_pairs,
        "prompt.mark_filer": True,
        "config.targets": "vocab",
        "config.entity_markers": True,
        "config.four_bit": False,
        "config.max_triples": 28,
        "config.max_prompt_tokens": 1024,
        "config.device": "cuda",
        "deterministic.seed": 42,
        "deterministic.deterministic_algorithms": True,
    }
    for path, expected in exact.items():
        if _required_path(manifest, path) != expected:
            failures.append(path)
    required = (
        "config.code_commit",
        "config.model_revision",
        "hashes.adapter",
        "hashes.test",
        "hashes.vocab",
        "hashes.outcomes",
        "environment.python",
        "environment.torch",
        "environment.transformers",
        "environment.cuda",
        "environment.driver",
        "environment.gpu",
        "prompt.mean_tokens",
        "prompt.max_tokens",
    )
    for path in required:
        if _required_path(manifest, path) in (None, ""):
            failures.append(path)
    return failures


def sha256_path(path: str | Path) -> str:
    """Hash a file or a directory tree including relative names."""
    root = Path(path)
    digest = hashlib.sha256()
    if root.is_file():
        return hashlib.sha256(root.read_bytes()).hexdigest()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def environment_manifest() -> dict[str, str]:
    """Capture validation-relevant runtime versions without requiring CUDA locally."""

    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    cuda = "unavailable"
    try:
        import torch

        cuda = str(torch.version.cuda or "unavailable")
    except ImportError:
        pass
    driver = gpu = "unavailable"
    try:
        result = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .split(", ", maxsplit=1)
        )
        driver, gpu = result[0], result[1]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        pass
    return {
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "cuda": cuda,
        "driver": driver,
        "gpu": gpu,
    }


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def _raw_lines(path: str | Path) -> list[bytes]:
    text = Path(path).read_text(encoding="utf-8")
    return [line.encode("utf-8") for line in text.splitlines(keepends=True)]


def _token_counts(
    tokenizer: Any,
    pairs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int]]:
    inputs = [
        len(tokenizer(str(pair.get("text", "")), add_special_tokens=False)["input_ids"])
        for pair in pairs
    ]
    outputs = [
        len(
            tokenizer(
                json.dumps(row.get("gold") or [], ensure_ascii=False, separators=(",", ":")),
                add_special_tokens=False,
            )["input_ids"]
        )
        for row in rows
    ]
    return inputs, outputs


def _comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Canonical agreement-routing comparison",
        "",
        f"**Verdict:** {report['verdict']}",
        "",
    ]
    failures = report["reproduction_failures"]
    lines.append(
        "**Reproduction:** PASS"
        if not failures
        else f"**Reproduction:** FAIL ({', '.join(failures)})"
    )
    lines.extend(
        [
            "",
            "| Signal | AUROC | 15% F1 | 20% F1 | 25% F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, signal in report["signals"].items():
        auroc = signal["auroc"]
        auroc_text = "n/a" if auroc is None else f"{auroc:.4f}"
        budgets = signal["budgets"]
        lines.append(
            f"| {name} | {auroc_text} | {budgets['0.15']['f1']:.4f} | "
            f"{budgets['0.20']['f1']:.4f} | {budgets['0.25']['f1']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("write-run-manifest")
    for name in ("log", "outcomes", "adapter", "test", "vocab", "out"):
        manifest.add_argument(f"--{name}", required=True)
    manifest.add_argument("--code-commit", required=True)
    manifest.add_argument("--model-revision", required=True)
    manifest.add_argument("--expected-pairs", type=int, default=2193)
    freeze = sub.add_parser("freeze-baseline")
    for name in ("pairs", "outcomes", "adapter", "vocab", "tokenizer", "out"):
        freeze.add_argument(f"--{name}", required=True)
    freeze.add_argument("--code-commit", required=True)
    freeze.add_argument("--model-revision", required=True)
    compare = sub.add_parser("compare")
    for name in (
        "baseline",
        "pairs",
        "outcomes",
        "tokenizer",
        "run-manifest",
        "out-json",
        "out-md",
    ):
        compare.add_argument(f"--{name}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write-run-manifest":
        manifest = build_run_manifest(
            Path(args.log).read_text(encoding="utf-8"),
            config={
                "code_commit": args.code_commit,
                "model_revision": args.model_revision,
                "targets": "vocab",
                "entity_markers": True,
                "four_bit": False,
                "max_triples": 28,
                "max_prompt_tokens": 1024,
                "device": "cuda",
            },
            hashes={
                "adapter": sha256_path(args.adapter),
                "test": sha256_path(args.test),
                "vocab": sha256_path(args.vocab),
                "outcomes": sha256_path(args.outcomes),
            },
            environment=environment_manifest(),
            deterministic={"seed": 42, "deterministic_algorithms": True},
            exit_code=0,
        )
        failures = validate_run_manifest(manifest, expected_pairs=args.expected_pairs)
        if failures:
            raise SystemExit(f"invalid run manifest: {', '.join(failures)}")
        _write_json(args.out, manifest)
        return 0
    if args.command == "freeze-baseline":
        rows = _read_jsonl(args.outcomes)
        pairs = _read_jsonl(args.pairs)
        raw_lines = _raw_lines(args.pairs)
        if not (len(rows) == len(pairs) == len(raw_lines)):
            raise SystemExit("pairs and outcomes are misaligned")
        tokenizer = _load_tokenizer(args.tokenizer)
        input_counts, output_counts = _token_counts(tokenizer, pairs, rows)
        contract = {
            "entity_markers": True,
            "targets": "vocab",
            "max_triples": 28,
            "max_prompt_tokens": 1024,
            "truncated": 0,
            "loose_match": False,
            "legacy_unstructured_contract": True,
            "evidence": [
                "docs/results/entity-markers-2026-07-17/README.md",
                "outputs/backfill/cascade-markers/outcomes.jsonl",
            ],
        }
        baseline = freeze_confidence_baseline(
            rows,
            raw_lines,
            threshold=0.85,
            input_token_counts=input_counts,
            output_token_counts=output_counts,
            contract=contract,
        )
        baseline.update(
            {
                "policy_id": "confidence-tau-0.85-markers-vocab-max28",
                "code_commit": args.code_commit,
                "model_revision": args.model_revision,
                "environment": environment_manifest(),
                "hashes": {
                    "pairs": sha256_path(args.pairs),
                    "outcomes": sha256_path(args.outcomes),
                    "adapter": sha256_path(args.adapter),
                    "vocab": sha256_path(args.vocab),
                    "tokenizer": sha256_path(args.tokenizer),
                },
            }
        )
        _write_json(args.out, baseline)
        return 0
    if args.command == "compare":
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        manifest = json.loads(Path(args.run_manifest).read_text(encoding="utf-8"))
        rows = _read_jsonl(args.outcomes)
        pairs = _read_jsonl(args.pairs)
        raw_lines = _raw_lines(args.pairs)
        if not (len(rows) == len(pairs) == len(raw_lines)):
            raise SystemExit("pairs and outcomes are misaligned")
        tokenizer = _load_tokenizer(args.tokenizer)
        input_counts, output_counts = _token_counts(tokenizer, pairs, rows)
        report = comparison_report(
            baseline,
            rows,
            pairs,
            raw_lines,
            input_token_counts=input_counts,
            output_token_counts=output_counts,
            manifest=manifest,
        )
        identity_failures = []
        actual_hashes = {
            "test": sha256_path(args.pairs),
            "outcomes": sha256_path(args.outcomes),
        }
        for name, actual in actual_hashes.items():
            if manifest.get("hashes", {}).get(name) != actual:
                identity_failures.append(f"hashes.{name}")
        baseline_pairs = baseline.get("hashes", {}).get("pairs")
        if baseline_pairs is not None and baseline_pairs != actual_hashes["test"]:
            identity_failures.append("baseline.hashes.pairs")
        for name in ("adapter", "vocab"):
            expected = baseline.get("hashes", {}).get(name)
            if expected is not None and manifest.get("hashes", {}).get(name) != expected:
                identity_failures.append(f"baseline.hashes.{name}")
        for name in ("code_commit", "model_revision"):
            expected = baseline.get(name)
            if expected is not None and manifest.get("config", {}).get(name) != expected:
                identity_failures.append(f"baseline.{name}")
        if identity_failures:
            report["reproduction_failures"] = sorted(
                set(report["reproduction_failures"] + identity_failures)
            )
            report["verdict"] = "REPRODUCIBILITY_FAIL"
        _write_json(args.out_json, report)
        markdown_path = Path(args.out_md)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_comparison_markdown(report), encoding="utf-8")
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
