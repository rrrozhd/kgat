import hashlib
import json
import math
from pathlib import Path

import kgat.eval.agreement_validation as agreement_validation
from kgat.eval.agreement_validation import (
    bucket_report,
    budget_order,
    build_run_manifest,
    comparison_report,
    exact_frontier,
    freeze_confidence_baseline,
    legacy_chunk_key,
    main,
    routing_auroc,
    signal_verdict,
    validate_reproduction,
    validate_run_manifest,
)


def test_legacy_chunk_key_hashes_exact_line_bytes():
    raw = b'{"text":"caf\xc3\xa9"}\n'

    assert legacy_chunk_key(raw) == hashlib.sha256(raw).hexdigest()
    assert legacy_chunk_key(raw) != legacy_chunk_key(raw.rstrip(b"\n"))


def test_budget_order_uses_floor_budget_and_stable_tie_keys():
    scores = [0.2, 0.1, 0.1, 0.9, 0.3]
    tie_keys = ["e", "d", "a", "b", "c"]

    assert budget_order(scores, tie_keys, 0.60) == [2, 1, 0]
    assert len(budget_order(scores, tie_keys, 0.60)) == math.floor(0.60 * len(scores))


def test_freeze_confidence_baseline_records_decisions_quality_and_token_totals():
    rows = [
        {"gold": [["r", "a"]], "pred": [], "confidence": 0.80},
        {"gold": [["r", "b"]], "pred": [["r", "b"]], "confidence": 0.90},
        {"gold": [], "pred": [["r", "c"]], "confidence": 0.70},
    ]
    raw_lines = [b'{"id":1}\n', b'{"id":2}\n', b'{"id":3}\n']
    contract = {
        "entity_markers": True,
        "targets": "vocab",
        "max_triples": 28,
        "max_prompt_tokens": 1024,
        "truncated": 0,
    }

    baseline = freeze_confidence_baseline(
        rows,
        raw_lines,
        threshold=0.85,
        input_token_counts=[10, 20, 30],
        output_token_counts=[2, 3, 4],
        contract=contract,
    )

    assert baseline["policy"] == {"signal": "confidence", "threshold": 0.85}
    assert baseline["n_chunks"] == 3
    assert baseline["escalated_calls"] == 2
    assert baseline["escalation_rate"] == 2 / 3
    assert baseline["escalated_input_tokens"] == 40
    assert baseline["escalated_output_tokens"] == 6
    assert baseline["metrics"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact": 1.0}
    assert baseline["legacy_contract"] == contract
    assert len(baseline["decision_sha256"]) == 64
    assert len(baseline["projection_sha256"]) == 64


def test_projection_hash_canonicalizes_sub_ulps_but_not_triples():
    raw_lines = [b'{"text":"x"}\n']
    kwargs = {
        "threshold": 0.85,
        "input_token_counts": [1],
        "output_token_counts": [1],
        "contract": {"max_triples": 28},
    }
    baseline = freeze_confidence_baseline(
        [{"gold": [["r", "Target"]], "pred": [], "confidence": 0.8}],
        raw_lines,
        **kwargs,
    )
    ulp_variant = freeze_confidence_baseline(
        [{"gold": [["r", "Target"]], "pred": [], "confidence": 0.8000000000000002}],
        raw_lines,
        **kwargs,
    )
    triple_variant = freeze_confidence_baseline(
        [{"gold": [["r", "target"]], "pred": [], "confidence": 0.8}],
        raw_lines,
        **kwargs,
    )

    assert baseline["projection_sha256"] == ulp_variant["projection_sha256"]
    assert baseline["projection_sha256"] != triple_variant["projection_sha256"]


def test_validate_reproduction_enforces_token_tolerance_and_exact_decisions():
    baseline = {
        "decision_sha256": "same",
        "escalated_calls": 10,
        "escalated_input_tokens": 1000,
        "escalated_output_tokens": 200,
        "escalation_rate": 0.2,
        "metrics": {"f1": 0.8, "recall": 0.8},
        "legacy_contract": {"max_triples": 28, "truncated": 0},
    }
    candidate = {
        **baseline,
        "escalated_input_tokens": 1004,
        "escalated_output_tokens": 201,
    }

    assert validate_reproduction(baseline, candidate) == []

    candidate["decision_sha256"] = "different"
    candidate["escalated_input_tokens"] = 1006
    failures = validate_reproduction(baseline, candidate)
    assert "decision_sha256" in failures
    assert "escalated_input_tokens" in failures


def test_exact_frontier_contains_every_ranked_escalation_point():
    rows = [
        {"gold": [["r", "a"]], "pred": [], "confidence": 0.9},
        {"gold": [["r", "b"]], "pred": [["r", "b"]], "confidence": 0.2},
        {"gold": [], "pred": [], "confidence": 0.5},
    ]
    scores = [0.1, 0.2, 0.3]
    tie_keys = ["a", "b", "c"]

    frontier = exact_frontier(rows, scores, tie_keys)

    assert [point["escalated_calls"] for point in frontier] == [0, 1, 2, 3]
    assert frontier[0]["recall"] == 0.5
    assert frontier[1]["recall"] == 1.0


def test_routing_auroc_rewards_lower_scores_on_error_chunks():
    rows = [
        {"gold": [["r", "a"]], "pred": []},
        {"gold": [["r", "b"]], "pred": [["r", "b"]]},
        {"gold": [], "pred": [["r", "c"]]},
        {"gold": [], "pred": []},
    ]

    assert routing_auroc(rows, [0.1, 0.8, 0.2, 0.9]) == 1.0


def test_signal_verdict_requires_nearby_noninferiority_and_one_real_gain():
    confidence = {
        "0.15": {"recall": 0.80, "f1": 0.80, "exact": 0.80},
        "0.20": {"recall": 0.82, "f1": 0.82, "exact": 0.82},
        "0.25": {"recall": 0.84, "f1": 0.84, "exact": 0.84},
    }
    agreement = {
        "0.15": {"recall": 0.80, "f1": 0.80, "exact": 0.80},
        "0.20": {"recall": 0.83, "f1": 0.83, "exact": 0.82},
        "0.25": {"recall": 0.839, "f1": 0.839, "exact": 0.839},
    }

    assert signal_verdict([], confidence, {"agreement": agreement}) == "ECONOMICS_READY"
    assert (
        signal_verdict(["decision_sha256"], confidence, {"agreement": agreement})
        == "REPRODUCIBILITY_FAIL"
    )

    agreement["0.25"]["recall"] = 0.83
    assert signal_verdict([], confidence, {"agreement": agreement}) == "SIGNAL_NO_GO"


def test_bucket_report_declares_small_buckets_descriptive_only():
    rows = [
        {"gold": [["supplier", "a"]], "pred": [], "confidence": 0.1},
        {"gold": [], "pred": [], "confidence": 0.9},
    ]
    pairs = [
        {"filer": "A", "item_key": "Item 1", "text": "short"},
        {"filer": "B", "item_key": "Item 2", "text": "a much longer chunk"},
    ]

    report = bucket_report(rows, pairs, [0.1, 0.9], ["a", "b"], budget=0.5)

    assert report["positive"]["true"]["n"] == 1
    assert report["positive"]["true"]["gating"] is False
    assert report["relation"]["supplier"]["recall"] == 1.0


def test_build_run_manifest_parses_structured_contract_from_log():
    log = """
cascade eval: overriding --entity-markers=True with entity_markers=True from adapter meta
cascade eval: 2193 test pairs, 8993 targets
  prompt tokens: mean=588 max=845 truncated=0/2193 (mark_filer=True)
"""
    config = {
        "code_commit": "abc123",
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "targets": "vocab",
        "entity_markers": True,
        "four_bit": False,
        "max_triples": 28,
        "max_prompt_tokens": 1024,
        "device": "cuda",
    }
    manifest = build_run_manifest(
        log,
        config=config,
        hashes={"adapter": "a", "test": "b", "vocab": "c", "outcomes": "d"},
        environment={
            "python": "3.11.10",
            "torch": "2.6.0+cu124",
            "transformers": "5.14.1",
            "cuda": "12.4",
            "driver": "565.57.01",
            "gpu": "RTX 3090",
        },
        deterministic={"seed": 42, "deterministic_algorithms": True},
        exit_code=0,
    )

    assert manifest["n_pairs"] == 2193
    assert manifest["n_targets"] == 8993
    assert manifest["prompt"] == {
        "mean_tokens": 588,
        "max_tokens": 845,
        "truncated": 0,
        "total": 2193,
        "mark_filer": True,
    }
    assert validate_run_manifest(manifest) == []


def test_validate_run_manifest_fails_closed_on_missing_or_wrong_contract():
    manifest = {
        "exit_code": 0,
        "n_pairs": 2193,
        "n_targets": 8993,
        "prompt": {"truncated": 0, "total": 2193, "mark_filer": True},
        "config": {
            "code_commit": "abc",
            "model_revision": "rev",
            "targets": "chunk",
            "entity_markers": True,
            "four_bit": False,
            "max_triples": 8,
            "max_prompt_tokens": 1024,
            "device": "cuda",
        },
        "hashes": {},
        "environment": {},
        "deterministic": {},
    }

    failures = validate_run_manifest(manifest)

    assert "config.targets" in failures
    assert "config.max_triples" in failures
    assert "hashes.adapter" in failures
    assert "environment.torch" in failures


def test_write_run_manifest_cli_writes_validated_json(tmp_path: Path):
    log = tmp_path / "run.log"
    log.write_text(
        "cascade eval: 1 test pairs, 8993 targets\n"
        "prompt tokens: mean=12 max=12 truncated=0/1 (mark_filer=True)\n"
    )
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text('{"gold":[],"pred":[],"confidence":1.0}\n')
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    test = tmp_path / "test.jsonl"
    test.write_text("{}\n")
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"targets":[]}')
    output = tmp_path / "manifest.json"

    result = main(
        [
            "write-run-manifest",
            "--log",
            str(log),
            "--outcomes",
            str(outcomes),
            "--adapter",
            str(adapter),
            "--test",
            str(test),
            "--vocab",
            str(vocab),
            "--out",
            str(output),
            "--code-commit",
            "abc123",
            "--model-revision",
            "c1899de289a04d12100db370d81485cdf75e47ca",
            "--expected-pairs",
            "1",
        ]
    )

    assert result == 0
    manifest = json.loads(output.read_text())
    assert manifest["config"]["max_triples"] == 28
    assert manifest["hashes"]["outcomes"] == hashlib.sha256(outcomes.read_bytes()).hexdigest()


def test_freeze_baseline_cli_records_hash_bound_legacy_contract(tmp_path: Path, monkeypatch):
    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": str(text).split()}

    monkeypatch.setattr(agreement_validation, "_load_tokenizer", lambda _: FakeTokenizer())
    pairs = tmp_path / "test.jsonl"
    pairs.write_text('{"text":"one two","triples":[["r","a"]]}\n{"text":"three","triples":[]}\n')
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        '{"gold":[["r","a"]],"pred":[],"confidence":0.8}\n{"gold":[],"pred":[],"confidence":0.9}\n'
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "extractor_meta.json").write_text('{"entity_markers":true,"targets_mode":"vocab"}')
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"targets":["a"]}')
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}")
    output = tmp_path / "baseline.json"

    assert (
        main(
            [
                "freeze-baseline",
                "--pairs",
                str(pairs),
                "--outcomes",
                str(outcomes),
                "--adapter",
                str(adapter),
                "--vocab",
                str(vocab),
                "--tokenizer",
                str(tokenizer_dir),
                "--out",
                str(output),
                "--code-commit",
                "abc123",
                "--model-revision",
                "rev123",
            ]
        )
        == 0
    )

    baseline = json.loads(output.read_text())
    assert baseline["policy_id"] == "confidence-tau-0.85-markers-vocab-max28"
    assert baseline["legacy_contract"]["max_triples"] == 28
    assert baseline["legacy_contract"]["truncated"] == 0
    assert baseline["hashes"]["pairs"] == hashlib.sha256(pairs.read_bytes()).hexdigest()


def test_comparison_report_contains_exact_frontiers_auroc_and_buckets():
    rows = [
        {
            "gold": [["supplier", "a"]],
            "pred": [],
            "confidence": 0.8,
            "agreement": 0.1,
            "min_agreement": 0.05,
        },
        {
            "gold": [],
            "pred": [],
            "confidence": 0.9,
            "agreement": 0.9,
            "min_agreement": 0.8,
        },
    ]
    pairs = [
        {"text": "one two", "filer": "A", "item_key": "Item 1"},
        {"text": "three", "filer": "B", "item_key": "Item 2"},
    ]
    raw_lines = [b'{"id":1}\n', b'{"id":2}\n']
    contract = {
        "entity_markers": True,
        "targets": "vocab",
        "max_triples": 28,
        "max_prompt_tokens": 1024,
        "truncated": 0,
    }
    baseline = freeze_confidence_baseline(
        rows,
        raw_lines,
        threshold=0.85,
        input_token_counts=[2, 1],
        output_token_counts=[1, 1],
        contract=contract,
    )
    manifest = {
        "exit_code": 0,
        "n_pairs": 2,
        "n_targets": 8993,
        "prompt": {
            "mean_tokens": 2,
            "max_tokens": 2,
            "truncated": 0,
            "total": 2,
            "mark_filer": True,
        },
        "config": {
            "code_commit": "abc",
            "model_revision": "rev",
            "targets": "vocab",
            "entity_markers": True,
            "four_bit": False,
            "max_triples": 28,
            "max_prompt_tokens": 1024,
            "device": "cuda",
        },
        "hashes": {"adapter": "a", "test": "b", "vocab": "c", "outcomes": "d"},
        "environment": {
            "python": "3.11",
            "torch": "2.6",
            "transformers": "5.14.1",
            "cuda": "12.4",
            "driver": "565",
            "gpu": "3090",
        },
        "deterministic": {"seed": 42, "deterministic_algorithms": True},
    }

    report = comparison_report(
        baseline,
        rows,
        pairs,
        raw_lines,
        input_token_counts=[2, 1],
        output_token_counts=[1, 1],
        manifest=manifest,
    )

    assert report["reproduction_failures"] == []
    assert report["signals"]["agreement"]["auroc"] == 1.0
    assert len(report["signals"]["agreement"]["exact_frontier"]) == 3
    assert "positive" in report["signals"]["agreement"]["buckets_at_20pct"]


def test_compare_cli_writes_json_and_markdown(tmp_path: Path, monkeypatch):
    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": str(text).split()}

    monkeypatch.setattr(agreement_validation, "_load_tokenizer", lambda _: FakeTokenizer())
    pairs = tmp_path / "test.jsonl"
    pair_lines = [
        b'{"text":"one two","filer":"A","item_key":"Item 1"}\n',
        b'{"text":"three","filer":"B","item_key":"Item 2"}\n',
    ]
    pairs.write_bytes(b"".join(pair_lines))
    rows = [
        {"gold": [["r", "a"]], "pred": [], "confidence": 0.8},
        {"gold": [], "pred": [], "confidence": 0.9},
    ]
    contract = {
        "entity_markers": True,
        "targets": "vocab",
        "max_triples": 28,
        "max_prompt_tokens": 1024,
        "truncated": 0,
    }
    baseline = freeze_confidence_baseline(
        rows,
        pair_lines,
        threshold=0.85,
        input_token_counts=[2, 1],
        output_token_counts=[1, 1],
        contract=contract,
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text(
        '{"gold":[["r","a"]],"pred":[],"confidence":0.8,"agreement":0.1,"min_agreement":0.05}\n'
        '{"gold":[],"pred":[],"confidence":0.9,"agreement":0.9,"min_agreement":0.8}\n'
    )
    manifest = {
        "exit_code": 0,
        "n_pairs": 2,
        "n_targets": 8993,
        "prompt": {
            "mean_tokens": 2,
            "max_tokens": 2,
            "truncated": 0,
            "total": 2,
            "mark_filer": True,
        },
        "config": {
            "code_commit": "abc",
            "model_revision": "rev",
            "targets": "vocab",
            "entity_markers": True,
            "four_bit": False,
            "max_triples": 28,
            "max_prompt_tokens": 1024,
            "device": "cuda",
        },
        "hashes": {
            "adapter": "a",
            "test": hashlib.sha256(pairs.read_bytes()).hexdigest(),
            "vocab": "c",
            "outcomes": hashlib.sha256(outcomes.read_bytes()).hexdigest(),
        },
        "environment": {
            "python": "3.11",
            "torch": "2.6",
            "transformers": "5.14.1",
            "cuda": "12.4",
            "driver": "565",
            "gpu": "3090",
        },
        "deterministic": {"seed": 42, "deterministic_algorithms": True},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    json_out = tmp_path / "comparison.json"
    md_out = tmp_path / "comparison.md"

    assert (
        main(
            [
                "compare",
                "--baseline",
                str(baseline_path),
                "--pairs",
                str(pairs),
                "--outcomes",
                str(outcomes),
                "--tokenizer",
                str(tokenizer_dir),
                "--run-manifest",
                str(manifest_path),
                "--out-json",
                str(json_out),
                "--out-md",
                str(md_out),
            ]
        )
        == 0
    )

    assert json.loads(json_out.read_text())["reproduction_failures"] == []
    assert "ECONOMICS_READY" in md_out.read_text() or "SIGNAL_NO_GO" in md_out.read_text()
