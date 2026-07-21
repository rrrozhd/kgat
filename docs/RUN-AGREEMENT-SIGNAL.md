# Run: grammar-agreement escalation signal (v0.0.22)

> **Canonical run completed 2026-07-20.** Reproduction passed and agreement
> improved routing at fixed budget, but matched-quality call reduction was only
> 0.49% (with +0.17% total escalation tokens), below the 10% production gate.
> Shadow mode remains disabled. See
> `docs/results/agreement-routing-2026-07-20/VERDICT.md`.

The original signal code is committed on `claude/reverent-diffie-f98cea`
(`e4ce679`); the canonical validation source is
`64c33eef0f06c6941c4f785fc75d7f1810deed93`. One GPU decode
pass answers: does the agreement signal close the confidence→P(ESCALATE)
routing gap WITHOUT touching the extractor?

## Reference numbers to beat (same 2193 test chunks, same accounting)

| system | F1 @20% esc | extraction cost |
|---|---|---|
| markers extractor + confidence (τ-cascade) | 0.808 | none — the floor to beat |
| markers extractor + gate/conf blend | 0.836 | none (gate is standalone) |
| warmup policy + P(ESCALATE)/conf | 0.867 | **−0.08 F1 extraction** |

Win condition: agreement / conf_x_agree ordering lands above 0.808 (any gap
closed is free — extraction untouched by construction). Anywhere near 0.836+
makes the interference-free path the default. Also read the AUROC table the
CLI prints — it's the threshold-free routing-accuracy comparison.

## Pod setup (fresh pod; old creds are stale — probe on 2026-07-20 got publickey denied)

1. Start a 3090-class pod, update `outputs/runpod_pod_id.txt` / `runpod_ssh_user.txt`.
2. Torch pin (see memory: transformers 5.x needs it):
   `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
   then `pip uninstall -y torchvision torchaudio` and verify
   `python -c "import transformers, torch; print(torch.__version__)"`.
3. Ship the branch (no push needed):
   `rsync -az -e "ssh -i ~/.ssh/runpod_kgat -p <PORT>" --exclude .venv --exclude outputs --exclude .git /Users/dondoe/coding/kgat/.claude/worktrees/reverent-diffie-f98cea/ root@<HOST>:/workspace/kgat/`
   plus the two dirs the run needs:
   `rsync -az ... outputs/adapters/qwen3-0.6b-extractor-markers root@<HOST>:/workspace/kgat/outputs/adapters/`
   `rsync -az ... data/backfill/real-strat-v2 root@<HOST>:/workspace/kgat/data/backfill/`

## The run (~2200 decodes, ≈ round-4 scale: well under $1)

```bash
cd /workspace/kgat
python -m kgat.eval.extractor_cascade \
  --model-id /workspace/models/Qwen3-0.6B-c1899de \
  --adapter outputs/adapters/qwen3-0.6b-extractor-markers \
  --data-dir data/backfill/real-strat-v2 --split test \
  --targets vocab --entity-markers --four-bit false \
  --max-triples 28 --max-prompt-tokens 1024 --device cuda \
  --signals all \
  --out-dir outputs/backfill/cascade-markers-agree-max28
```

Notes:
- Do not add `--loose-match`: the frozen July 17 comparator stores strict,
  original-case triples. The canonical smoke proved normalization changes the
  persisted projection and is not historically comparable.
- Prints per-signal comparison (AUROC + cheapest tau meeting the 0.8 recall
  floor); confidence keeps the legacy layout at the out-dir root, other
  signals land in subdirs.
- Sanity check: the `confidence` row must reproduce the cascade-markers
  baseline (same adapter, same split) — if it doesn't, the decode isn't
  comparable and the agreement columns don't matter.

## Head-to-head vs the gate (after the decode; gate model = today's hardness retrain)

```bash
python -m kgat.eval.gate_frontier \
  --gate <gate model dir from outputs/telemetry/gate-2026-07-20/> \
  --outcomes outputs/backfill/cascade-markers-agree-max28/outcomes.jsonl \
  --pairs data/backfill/real-strat-v2/test.jsonl \
  --out outputs/gate/agree_curve.json
```

The table now includes `agreement` and `conf_x_agree` orderings next to gate /
confidence / oracle / random, matched by construction.

## Before terminating the pod

Use `scripts/pull_agreement_artifacts.sh` with explicit host, port, pod ID,
remote output, destination, and expected-hash manifest. It verifies every pulled
artifact and never deletes remote data. The canonical raw outcomes and per-row
migration audit keys are preserved under
`docs/results/agreement-routing-2026-07-20/`.
