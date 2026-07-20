# Canonical Agreement GPU Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible 2,193-chunk grammar-agreement artifact under the exact historical markers-extractor contract (`targets=vocab`, entity markers, `max_triples=28`) and fail closed unless confidence reproduces before agreement is interpreted.

**Architecture:** First add small, tested validation tooling that freezes the legacy comparator, emits structured run manifests, and computes exact matched-budget/full-frontier results. Then archive one exact validation commit and a pinned Qwen snapshot to an ephemeral RTX 3090 pod, run a ten-row deterministic smoke followed by the full decode, pull and hash raw artifacts before analysis, commit a content-addressed result bundle, and terminate the pod through a finally-style cleanup path on every exit.

**Tech Stack:** Python 3.11, pytest, PyTorch 2.6.0+cu124, Transformers 5.14.1, PEFT, pinned Qwen3-0.6B snapshot `c1899de289a04d12100db370d81485cdf75e47ca`, RunPod GraphQL/SSH, SHA-256.

---

## File and artifact responsibilities

- Create `src/kgat/eval/agreement_validation.py`: pure manifest, comparator, stable-ordering, exact-frontier, token-estimate, and verdict functions plus CLI.
- Create `tests/test_agreement_validation.py`: unit tests for every production gate and legacy tie-key behavior.
- Create `scripts/run_agreement_max28.sh`: remote fail-fast evaluator wrapper with explicit flags, `pipefail`, wall-time guard, and structured manifest generation.
- Create `scripts/pull_agreement_artifacts.sh`: narrow, tested pull/hash helper for this run; replaces the nonexistent tracked pull helper.
- Modify `docs/RUN-AGREEMENT-SIGNAL.md`: literal canonical commands after they have executed successfully.
- Read `outputs/adapters/qwen3-0.6b-extractor-markers/`: canonical closed-vocabulary markers adapter.
- Read `data/backfill/real-strat-v2/`: canonical test split and vocabulary.
- Read `outputs/backfill/cascade-markers/outcomes.jsonl`: legacy confidence comparator.
- Create and force-add `docs/results/agreement-routing-2026-07-20/`: durable Git result bundle containing raw outcomes, manifests, hashes, exact frontiers, comparison, run log, and report. Adapter/data are referenced by verified hashes, not duplicated.
- Use `outputs/telemetry/agreement-canonical-max28-2026-07-20/` only as a staging directory.

### Task 1: Add deterministic validation and manifest tooling

**Files:**
- Create: `src/kgat/eval/agreement_validation.py`
- Create: `tests/test_agreement_validation.py`

- [ ] **Step 1: Write failing tests for legacy stable keys and exact budget ordering**

Test `legacy_chunk_key(raw_line)` as SHA-256 of the exact line bytes, `floor(b*n)` selection, ascending signal ordering, and tie resolution by the key rather than row index.

- [ ] **Step 2: Run the tests and verify they fail**

Run: `.venv/bin/pytest tests/test_agreement_validation.py -v`

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement stable keys and exact ordering minimally**

Expose pure functions with no GPU dependency. The legacy key is permitted only when the manifest says `dataset_contract=legacy_real_strat_v2`; production/shadow manifests reject it.

- [ ] **Step 4: Write failing tests for the immutable confidence comparator**

Cover `tau=0.85`, exact escalated-call count, decision-bitset SHA-256, F1/recall/exact/escalation metrics, prompt input-token total, serialized-gold output-token total, and the 0.5% token-total reproduction tolerance.

- [ ] **Step 5: Implement comparator creation and validation**

Token totals use the pinned Qwen tokenizer snapshot: input tokens are the exact UTF-8 chunk text tokenization; output tokens are compact JSON serialization of teacher gold. Record that this is a deterministic migration cost estimator, not historical API billing telemetry.

- [ ] **Step 6: Write failing tests for exact full frontiers and verdicts**

Test every distinct observed signal threshold, no subsampling, routing AUROC,
15%/20%/25% gates, tolerances, and the three verdicts `REPRODUCIBILITY_FAIL`,
`SIGNAL_NO_GO`, and `ECONOMICS_READY`. Test per-bucket output for positive/negative
chunks, chunk-length quartile, filer-volume decile, item key, and each gold relation;
every bucket reports `n` and buckets below 30 are labeled descriptive-only.

- [ ] **Step 7: Implement exact frontiers and verdicts**

Do not reuse the evaluator's approximately 41-point chart frontier for gating.
Write full JSON results plus AUROC and bucket tables; charts remain non-gating.

- [ ] **Step 8: Write failing tests for structured run-manifest validation**

Require exact code commit, model/tokenizer revision, adapter/test/vocab/outcomes hashes, all CLI flags, environment versions, deterministic settings, prompt mean/max/truncated count, split/target counts, and `max_triples=28`.

- [ ] **Step 9: Implement manifest creation/validation and CLI**

The CLI has `freeze-baseline`, `write-run-manifest`, and `compare` subcommands. `write-run-manifest` parses the evaluator log for prompt statistics and refuses missing fields.

- [ ] **Step 10: Run focused and existing evaluator tests**

Run: `.venv/bin/pytest tests/test_agreement_validation.py tests/test_extractor_cascade.py tests/test_gate_frontier.py -v`

Expected: all pass.

- [ ] **Step 11: Commit the tooling**

Stage only the new module/tests and commit `feat: add canonical agreement validation tooling`. Record the resulting exact `VALIDATION_COMMIT`; no “or later” commit is allowed in the GPU run.

### Task 2: Add fail-safe run and pull scripts

**Files:**
- Create: `scripts/run_agreement_max28.sh`
- Create: `scripts/pull_agreement_artifacts.sh`
- Test: `tests/test_agreement_run_scripts.py`

- [ ] **Step 1: Write failing script-contract tests**

Assert the remote wrapper uses `set -Eeuo pipefail`, a 90-minute timeout, the pinned local model snapshot path, and literal flags: `--targets vocab`, `--entity-markers`, `--four-bit false`, `--max-triples 28`, `--max-prompt-tokens 1024`, `--device cuda`, and `--signals all`. Do not pass `--loose-match`: the canonical CUDA 12.4 smoke proved the frozen July 17 baseline stores strict, original-case triples, while that flag normalizes the persisted outcomes.

- [ ] **Step 2: Implement the remote wrapper**

The wrapper accepts only `smoke` or `full`, tees logs without masking the Python exit code, and calls `write-run-manifest` after success.

- [ ] **Step 3: Write failing pull-helper tests**

Require explicit host, port, pod ID, remote output, local destination, and expected hashes. Reject empty/broad destinations. Pull outcomes/log/manifest/frontiers, compare local/remote hashes, and create a final manifest hash.

- [ ] **Step 4: Implement the narrow pull helper**

Do not depend on the untracked `scripts/pull_artifacts.sh`. Never delete remote data.

- [ ] **Step 5: Run tests and commit**

Run: `.venv/bin/pytest tests/test_agreement_run_scripts.py -v`

Expected: pass. Commit `ops: add fail-safe agreement validation runner` and update `VALIDATION_COMMIT` to this exact commit.

### Task 3: Freeze the migration comparator before provisioning

**Files:**
- Create: `docs/results/agreement-routing-2026-07-20/confidence_baseline.json`
- Create: `docs/results/agreement-routing-2026-07-20/preflight.sha256`

- [ ] **Step 1: Download/resolve the pinned model snapshot locally without GPU execution**

Resolve Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`
and verify that Hugging Face resolves exactly that 40-character commit before
recording snapshot file hashes. The same directory contents will be transferred to
the pod.

- [ ] **Step 2: Assert the legacy fixed contract**

Require: 2,193 test rows, 8,993 vocabulary targets, adapter `entity_markers=true` and `targets_mode=vocab`, 2,193 legacy outcomes, and maximum historical prediction length 28. Record `max_triples=28` as a legacy documented assertion sourced to `docs/results/entity-markers-2026-07-17/README.md`, with `legacy_unstructured_contract=true`; do not misrepresent it as a historical machine manifest.

- [ ] **Step 3: Freeze the `tau=0.85` comparator**

Run the new `freeze-baseline` CLI with the pinned tokenizer and exact raw JSONL
lines. Store policy ID, threshold, expected escalation rate, exact call count,
quality metrics, decision-bitset hash, ordered `{gold,pred,confidence}` hash,
input/output token totals, code commit, environment, and all input hashes. Also
store `entity_markers=true`, `targets=vocab`, `max_triples=28`,
`max_prompt_tokens=1024`, and the historical truncation count of zero in a
hash-bound `legacy_contract` block. Mark the latter fields as documented legacy
assertions and cite their evidence paths.

- [ ] **Step 4: Verify and commit the comparator**

Run its self-validation command. Force-add only the result directory files and commit `test: freeze canonical confidence comparator`.

### Task 4: Provision the pod with unconditional teardown protection

**Files:**
- Stage: `outputs/telemetry/agreement-canonical-max28-2026-07-20/runpod.json`

- [ ] **Step 1: Start a local orchestration shell with cleanup trap**

Use `set -Eeuo pipefail`. Immediately after pod creation, record the exact pod ID
and install an `EXIT INT TERM` trap that gives diagnostic pull at most 45 seconds,
then terminates only that pod ID regardless of pull success. There is no
keep-alive-on-failure mode in this run, and cleanup must finish inside the local
120-minute hard deadline.

- [ ] **Step 2: Enforce spend and wall-time caps**

Create one on-demand RTX 3090 at no more than $0.30/hr, with a $1.00 experiment budget and 90-minute decode timeout. A local 120-minute hard deadline invokes cleanup even if SSH hangs.

- [ ] **Step 3: Record pre-run account state and pod metadata**

Capture balance, existing pod IDs, pre-run hourly spend, created pod ID, price,
GPU, image, host, and SSH port. Never stop or terminate an ID not equal to the
recorded experiment pod.

### Task 5: Install the immutable runtime and transfer exact inputs

**Files:**
- Transfer archive of exact `VALIDATION_COMMIT`
- Transfer pinned Qwen snapshot revision
- Transfer canonical adapter and dataset

- [ ] **Step 1: Install pinned runtime**

Install Torch `2.6.0+cu124`, Transformers `5.14.1`, pinned project dependencies, and remove torchvision/torchaudio. Record Python/package/CUDA/cuDNN/driver/GPU versions plus deterministic-algorithm and random-seed settings.

- [ ] **Step 2: Transfer a `git archive VALIDATION_COMMIT` source bundle**

Do not transfer the dirty worktree. Verify the remote source-tree archive hash before extraction.

- [ ] **Step 3: Transfer the pinned local model snapshot, adapter, and dataset**

Verify every preflight hash remotely before installing Edgewright editable with `--no-deps`.

### Task 6: Run and verify the deterministic smoke

**Files:**
- Create remotely: `outputs/backfill/agreement-max28-smoke/`
- Pull immediately: smoke outcomes, log, and manifest

- [ ] **Step 1: Execute `scripts/run_agreement_max28.sh smoke`**

The literal evaluator command inside the wrapper is:

```bash
python -m kgat.eval.extractor_cascade \
  --model-id /workspace/models/Qwen3-0.6B-c1899de \
  --adapter outputs/adapters/qwen3-0.6b-extractor-markers \
  --data-dir data/backfill/real-strat-v2 \
  --split test --max-examples 10 \
  --targets vocab --entity-markers --four-bit false \
  --max-triples 28 --max-prompt-tokens 1024 --device cuda \
  --signals all \
  --out-dir outputs/backfill/agreement-max28-smoke
```

- [ ] **Step 2: Pull and hash smoke artifacts before comparison**

Use the new pull helper. Verify local/remote equality.

- [ ] **Step 3: Require exact smoke reproduction**

The first ten ordered `{gold,pred,confidence}` rows and decision bits must match the frozen legacy comparator exactly. On mismatch, the trap pulls diagnostics and terminates the pod; the full run does not start.

### Task 7: Run the full canonical decode

**Files:**
- Create remotely: `outputs/backfill/cascade-markers-agree-max28/`

- [ ] **Step 1: Execute `scripts/run_agreement_max28.sh full`**

Use the exact smoke command without `--max-examples` and with output directory `cascade-markers-agree-max28`. The wrapper enforces `pipefail` and the 90-minute timeout.

- [ ] **Step 2: Monitor progress and account state**

Report progress periodically. Any decode/configuration/spend/time failure routes through the cleanup trap.

- [ ] **Step 3: Require structured completion evidence remotely**

Before pull, validate 2,193 outcomes, 8,993 targets, entity markers, explicit max 28, exact flags, zero exit status, prompt mean/max/truncated count, environment, and all hashes in `run_manifest.json`.

### Task 8: Pull raw artifacts, compare, publish, and terminate

**Files:**
- Populate staging: `outputs/telemetry/agreement-canonical-max28-2026-07-20/`
- Populate durable bundle: `docs/results/agreement-routing-2026-07-20/`

- [ ] **Step 1: Pull and hash raw artifacts before analysis**

Pull outcomes, run log, structured manifest, chart frontiers, and summaries. Verify local/remote outcomes and log hashes. Hash the pull manifest itself.

- [ ] **Step 2: Apply the complete reproducibility gate**

Run `agreement_validation compare` against the frozen `tau=0.85` baseline. Require F1 and recall within 0.002, escalation within 0.002, exact call count, exact ordered decision hash, token totals within 0.5%, prompt-marker match, exact truncation count, exact max-triples contract, and all identity hashes. Failure emits only `REPRODUCIBILITY_FAIL`.

- [ ] **Step 3: Compute exact full and matched-budget frontiers only after reproduction passes**

Evaluate every distinct threshold and exact 15%/20%/25% budgets using
`legacy_chunk_key=SHA256(raw JSONL line bytes)`. Compute routing AUROC and the
predeclared positive/negative, chunk-length, filer-volume, item-key, and relation
bucket tables. Apply the approved robustness tolerances and emit `SIGNAL_NO_GO` or
`ECONOMICS_READY`.

- [ ] **Step 4: Prepare and hash the uncommitted durable result bundle**

Copy raw outcomes, run log, manifests, comparison JSON/Markdown, exact frontiers,
AUROC and bucket reports, hashes, and draft README into
`docs/results/agreement-routing-2026-07-20/`. Verify every local hash and write the
bundle manifest, but do not commit yet because final billing/teardown evidence is
still pending.

- [ ] **Step 5: Correct the runbook with tested literal commands**

Document pinned commit/model revision, all explicit flags, exact preflight/install/transfer/run/compare/pull/hash/teardown commands, and the tracked narrow pull helper. Remove references to the chunk adapter, chunk grammar, CLI defaults, and nonexistent tracked pull script.

- [ ] **Step 6: Terminate and verify billing**

After local bundle verification, explicitly invoke cleanup, terminate only the
recorded pod ID, and query RunPod until the experiment pod is absent and hourly
spend returns to the recorded pre-run baseline. Do not alter unrelated pods or
serverless resources. Record final balance, pre/post hourly spend, and total
experiment draw in the report.

- [ ] **Step 7: Commit and verify the definitive result**

Update the README and manifest with final teardown/billing evidence. Force-add the
result bundle and corrected runbook, run `git diff --check`, commit, then re-hash
every bundle file from the committed tree. This Git commit is the content-addressed
release store for the small evidence bundle.
