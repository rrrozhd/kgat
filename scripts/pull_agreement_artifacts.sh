#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "usage: $0 --host HOST --port PORT --pod-id ID --remote-output PATH --destination PATH --expected-hashes PATH" >&2
  exit 64
}

HOST=""
PORT=""
POD_ID=""
REMOTE_OUTPUT=""
DESTINATION=""
EXPECTED_HASHES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --pod-id) POD_ID="${2:-}"; shift 2 ;;
    --remote-output) REMOTE_OUTPUT="${2:-}"; shift 2 ;;
    --destination) DESTINATION="${2:-}"; shift 2 ;;
    --expected-hashes) EXPECTED_HASHES="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -n "$HOST" && -n "$PORT" && -n "$POD_ID" ]] || usage
[[ "$REMOTE_OUTPUT" == /workspace/kgat/outputs/backfill/* ]] || usage
case "$DESTINATION" in
  ""|/|.|..|~|/Users|/workspace|/workspace/kgat) usage ;;
esac
[[ -f "$EXPECTED_HASHES" ]] || usage

SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod_kgat}"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -p "$PORT")
RSYNC_SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $SSH_KEY -p $PORT"

REMOTE_HASHES="$(${SSH[@]} "root@$HOST" "cd '$REMOTE_OUTPUT' && cat manifest.sha256")"
EXPECTED_CONTENT="$(cat "$EXPECTED_HASHES")"
[[ "$REMOTE_HASHES" == "$EXPECTED_CONTENT" ]] || {
  echo "remote hashes do not match --expected-hashes for pod $POD_ID" >&2
  exit 1
}

mkdir -p "$DESTINATION"
rsync -az -e "$RSYNC_SSH" \
  "root@$HOST:$REMOTE_OUTPUT/" "$DESTINATION/"

(
  cd "$DESTINATION"
  shasum -a 256 -c "$EXPECTED_HASHES"
  shasum -a 256 manifest.sha256 > manifest.sha256.final
)
cp "$EXPECTED_HASHES" "$DESTINATION/manifest.sha256"
shasum -a 256 "$DESTINATION/manifest.sha256" > "$DESTINATION/manifest.sha256.final"

test -f "$DESTINATION/outcomes.jsonl"
test -f "$DESTINATION/run.log"
test -f "$DESTINATION/run_manifest.json"
find "$DESTINATION" -type f \( -name '*frontier*.csv' -o -name 'tau_*.summary.json' \) \
  -print -quit | grep -q .
