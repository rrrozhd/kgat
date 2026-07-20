#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-}"
if ! [[ "$MODE" == "smoke" || "$MODE" == "full" ]]; then
  echo "usage: $0 smoke|full" >&2
  exit 64
fi

: "${VALIDATION_COMMIT:?set VALIDATION_COMMIT to the exact archived source commit}"

cd /workspace/kgat
MODEL_PATH=/workspace/models/Qwen3-0.6B-c1899de
MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
ADAPTER=outputs/adapters/qwen3-0.6b-extractor-markers
DATA_DIR=data/backfill/real-strat-v2
VOCAB="$DATA_DIR/vocab.json"

if [[ "$MODE" == "smoke" ]]; then
  OUT_DIR=outputs/backfill/agreement-max28-smoke
  EXPECTED_PAIRS=10
  EXTRA_ARGS=(--max-examples 10)
else
  OUT_DIR=outputs/backfill/cascade-markers-agree-max28
  EXPECTED_PAIRS=2193
  EXTRA_ARGS=()
fi

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

set +e
timeout 90m python -m kgat.eval.extractor_cascade \
  --model-id /workspace/models/Qwen3-0.6B-c1899de \
  --adapter outputs/adapters/qwen3-0.6b-extractor-markers \
  --data-dir data/backfill/real-strat-v2 \
  --split test "${EXTRA_ARGS[@]}" \
  --targets vocab --entity-markers --four-bit false \
  --max-triples 28 --max-prompt-tokens 1024 --device cuda \
  --signals all \
  --out-dir "$OUT_DIR" 2>&1 | tee "$LOG"
RUN_STATUS=${PIPESTATUS[0]}
set -e
if [[ "$RUN_STATUS" -ne 0 ]]; then
  exit "$RUN_STATUS"
fi

python -m kgat.eval.agreement_validation write-run-manifest \
  --log "$LOG" \
  --outcomes "$OUT_DIR/outcomes.jsonl" \
  --adapter "$ADAPTER" \
  --test "$DATA_DIR/test.jsonl" \
  --vocab "$VOCAB" \
  --out "$OUT_DIR/run_manifest.json" \
  --code-commit "$VALIDATION_COMMIT" \
  --model-revision "$MODEL_REVISION" \
  --expected-pairs "$EXPECTED_PAIRS"

(
  cd "$OUT_DIR"
  find . -type f ! -name manifest.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > manifest.sha256
)
