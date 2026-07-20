import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_runner_contract_is_explicit_and_fail_closed():
    text = _script("run_agreement_max28.sh")
    required = (
        "set -Eeuo pipefail",
        "timeout 90m",
        "/workspace/models/Qwen3-0.6B-c1899de",
        "--targets vocab",
        "--entity-markers",
        "--four-bit false",
        "--max-triples 28",
        "--max-prompt-tokens 1024",
        "--device cuda",
        "--signals all",
        "kgat.eval.agreement_validation write-run-manifest",
    )
    for value in required:
        assert value in text
    assert "--loose-match" not in text
    assert '[[ "$MODE" == "smoke" || "$MODE" == "full" ]]' in text
    assert "PIPESTATUS[0]" in text


def test_pull_contract_is_narrow_and_hash_verified():
    text = _script("pull_agreement_artifacts.sh")
    for option in (
        "--host",
        "--port",
        "--pod-id",
        "--remote-output",
        "--destination",
        "--expected-hashes",
    ):
        assert option in text
    for artifact in ("outcomes.jsonl", "run.log", "run_manifest.json"):
        assert artifact in text
    assert "frontier" in text
    assert "sha256sum" in text or "shasum -a 256" in text
    assert "manifest.sha256" in text
    assert "--protect-args" not in text
    assert "rm -" not in text


def test_scripts_parse_as_bash():
    for name in ("run_agreement_max28.sh", "pull_agreement_artifacts.sh"):
        subprocess.run(["bash", "-n", ROOT / "scripts" / name], check=True)
