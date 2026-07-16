# edgewright — Governed, Budget-Adaptive KG Construction & Traversal

Small constrained models + cost-aware policies, on **both sides** of a knowledge
graph:

- **Write path (graph building / backfill):** a grammar-constrained ~0.6B
  extractor reads document chunks and emits schema-valid triples or `NONE`,
  escalating only its low-confidence chunks to a big-LLM teacher pipeline.
- **Read path (traversal KGQA):** the same machinery inverted — the small model
  acts as a traversal policy: given a question and the current frontier it picks
  the next relation to expand or emits `STOP` (the counterpart of the write
  path's "skip this chunk"), under a token-trie constraint that makes off-graph
  actions impossible. A swappable synthesizer turns retrieved paths into answers.

Both paths are wrapped in a **governance layer** (per-hop / per-edge policies,
audit certificates, fail-closed provenance) and evaluated the same way: a
**cost/quality frontier** with an explicit cost axis (escalation rate on the
write path; hops/LLM calls on the read path). The demonstrated frontier is
traced by a confidence threshold; tracing it with a cost-penalized RL reward is
the active phase-2 work. The contribution is not "a small model can do X" — it
is budget-adaptive behavior, measured as quality-per-dollar, with auditability
as a first-class property. Model size is a *swept variable*, not a fixed choice.

---

## Status

Both pipelines run end-to-end on real data. **Current focus: the write path**
(KG construction); the read path is validated method-wise and parked until a
real-graph training corpus is exported.

**Write path (backfill, 2026-07 — active):** trained on distant supervision from
a production SEC-filing pipeline's own logged extractions (every committed edge
stores its evidence), the grammar-constrained 0.6B extractor recovers **~86% of
the teacher's edges at 27.8% escalation on the held-out test mix** (an estimated
~21% at the deployment chunk distribution; 25.1% with per-route thresholds) —
a dialable cost/quality frontier for graph construction, with escalation rate as
the cost axis. Phase 2 — GRPO over per-chunk `{skip | extract | escalate}` with
a judge-audited reward — is wired and GPU-validated at smoke scale: the teacher's
critic is distilled into a 150M cross-encoder from ~490k logged verdicts
(81% held-out agreement; the decision threshold is a tuned post-hoc knob, no LLM
inside the RL loop), objective evidence gates and per-edge governance
(fail-closed grounding, per-filing audit certificates) mirror the read path, and
the routing policy trains with grammar-masked clipped policy gradients. Known
open item before a full RL run: the SFT-initialized policy never samples
`ESCALATE` (cold start) — the fix is a warm-start SFT pass that labels the
extractor's own low-confidence chunks as escalation targets.

**Read path:** the full chain — mine → SFT → trie-constrained decode → GRPO —
is GPU-validated. On the synthetic FinKG testbed (templated questions; 90-question
dev split), SFT lifts the 0.6B controller from 0.16 to 0.82 Hit with
depth-adaptive search; GRPO (Dr. GRPO defaults + regret cost + exact potential
shaping) reaches 0.89. The λ sweep produced no cost separation on this KG — the
trained policy already operates at near-oracle depth, so FinKG's cost/quality
frontier degenerates to a point; demonstrating a real frontier awaits
distractor-dense data (WebQSP/CWQ).

| Milestone | What | State |
|-----------|------|-------|
| M0 | Skeleton, schemas, ABCs, `DummyController`, pytest green | implemented |
| M1 | Data + eval foundation (metrics, cost, frontier, harness) | implemented |
| M2 | Baseline reproduction harness (RoG / GCR / GNN-RAG wrappers) | stubbed (needs verified official repos + published numbers) |
| M3 | Trajectory mining (BFS oracle → engine replay → SFT JSONL) | implemented + tested |
| M4 | Decoder controller + trie-constrained decoding + LoRA/QLoRA SFT | GPU-validated (FinKG + WebQSP mining) |
| M5 | Trajectory-level GRPO + λ frontier | GPU-validated (λ + lr sweeps) |
| M6 | Size sweep + cross-encoder floor | cross-encoder now trained as the write-path judge; controller floor pending |
| M7 | Arch B / Arch C arms | stub (`gnn_proposer`; dynamic trie shares `constrained_decoding`) |
| M8 | Governance layer + audit + overhead measurement | read + write policies/certificates implemented & wired; overhead study pending |
| M9 | Ablations, transfer KG, write-up | future |
| — | **Write path**: extractor SFT + confidence cascade + frontier | 4 measured rounds on real data |
| — | **Write path phase 2**: routing RL + distilled judge + edge governance | loop GPU-validated (smoke); escalation warm-start → full run next |

---

## Install

Requires **Python 3.11+**. The foundation installs with *no model dependencies*.

```bash
# with uv (recommended)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# or with pip
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Extras: `.[ml]` (torch/transformers/peft/trl — the model arm), `.[gnn]`
(torch-geometric), `.[data]` (HuggingFace `datasets` for real dataset downloads),
`.[wandb]`, `.[neo4j]`.

## Quickstart

```bash
pytest -q                       # foundation tests — green with zero model deps

# End-to-end dummy run on the bundled sample dataset (offline, no GPU):
python -m kgat.eval.harness controller=dummy synth=dummy dataset=sample

# Trivial cost/quality frontier on the sample:
bash scripts/eval_frontier.sh
```

### Training workflow (M3 → M5)

```bash
# 1. Mine oracle trajectories (CPU-only)
python -m kgat.train.mine_trajectories dataset=webqsp dataset.split=train

# 2. SFT the controller (QLoRA on CUDA; LoRA on MPS/CPU)
python -m kgat.train.sft train=sft dataset=webqsp dataset.split=train model=qwen3-0.6b

# 3. Evaluate the trained controller (trie-constrained decoding)
python -m kgat.eval.harness dataset=webqsp model=qwen3-0.6b controller=decoder \
    controller.adapter_path=outputs/adapters/qwen3-0.6b-sft

# 4. GRPO + the lambda frontier sweep
python -m kgat.train.grpo -m dataset=webqsp dataset.split=train model=qwen3-0.6b \
    train.grpo.lam=0.0,0.05,0.1,0.2,0.4
```

The full chain smoke-tests anywhere (CPU/MPS, ~1 min, tiny random model):

```bash
bash scripts/run_sft.sh sample tiny-test dev
```

### Write-path workflow (backfill extractor)

```bash
# Offline end-to-end (synthetic pairs, tiny model — no GPU, no data export):
bash scripts/run_backfill_pilot.sh tiny-test 20

# Real pipeline: export distant-supervision pairs (SQL contract in the module
# docstring), SFT the extractor, sweep the escalation threshold into a frontier:
python -m kgat.data.backfill_export --export exports/pairs.jsonl --out data/backfill/real
python -m kgat.train.sft_extractor train=sft_extractor model=qwen3-0.6b \
    train.sft_extractor.data_dir=data/backfill/real
python -m kgat.eval.extractor_cascade --model-id Qwen/Qwen3-0.6B \
    --adapter outputs/adapters/qwen3-0.6b-extractor \
    --data-dir data/backfill/real --out-dir outputs/backfill/cascade

# Distill the teacher's per-edge critic into the 150M reward judge:
python -m kgat.data.judge_export --export exports/judge.jsonl --out data/judge
python -m kgat.train.judge train=judge model=crossencoder-modernbert

# Phase 2: GRPO over per-chunk {skip | extract | escalate} routing
# (warm-starts from the SFT extractor adapter; judge = gates + distilled critic):
python -m kgat.train.grpo_routing train=grpo_routing model=qwen3-0.6b \
    train.grpo_routing.judge=outputs/judges/crossencoder-modernbert
```

On Colab, open `notebooks/colab_kgat.ipynb` — T4 covers mining/SFT/eval; use an
A100 for the full GRPO sweep. **Note:** smoke-test (`train.grpo.max_questions=32`)
before committing to a long sweep.

Multirun sweeps (the reason we use Hydra):

```bash
python -m kgat.eval.harness -m model=qwen2.5-0.5b,qwen2.5-1.5b,qwen2.5-3b dataset=webqsp controller=dummy
python -m kgat.eval.harness -m train.grpo.lam=0.0,0.05,0.1,0.2,0.4 dataset=webqsp controller=dummy
```

## Datasets

We reuse the **preprocessed per-question subgraph** format released with the
baselines rather than rebuilding Freebase. Expected on-disk schema (one JSON object
per line, `*.jsonl`):

```json
{
  "id": "WebQTrn-0",
  "question": "what is the name of justin bieber brother",
  "q_entity": ["m.06w2sn5"],
  "a_entity": ["m.0gxnnwc"],
  "graph": [["m.06w2sn5", "people.person.sibling_s", "m.0gxnnwc"], ...]
}
```

> **Assumption to verify at M2:** this matches the `rmanluo/RoG-webqsp` /
> `rmanluo/RoG-cwq` HuggingFace release. `data/loaders.py` documents this and the
> real download path; the bundled `data/sample/*.jsonl` follows the same schema so
> tests and the offline smoke run need no network. Confirm the field names against
> the actual release before running M2.

## Repository layout

The Python package keeps the historical import name `kgat` (`import kgat`);
only the project/distribution name changed to `edgewright`.

```
configs/          Hydra config groups (model / dataset / train / experiment / controller / synth)
src/kgat/
  data/           schemas (THE contract), loaders, finkg generator,
                  backfill_export + judge_export (write-path distant supervision)
  graph/          KGStore ABC + in-memory impl (+ Neo4j adapter stub);
                  KGWriteStore + per-edge provenance (write side)
  controller/     TraversalController ABC + decoder/dummy policies;
                  constrained_decoding: relation trie (read) + triple grammar (write)
  synthesis/      AnswerSynthesizer ABC + DummySynthesizer (+ path_reader stub)
  governance/     read: HopPolicy chain + AuditCertificate;
                  write: EdgePolicy chain + WriteCertificate + governed_commit
  traversal/      engine (main loop) + budget ledger
  train/          reward, mining, sft, grpo (read path);
                  sft_extractor, backfill_routing, edge_judge, judge,
                  grpo_routing (write path)
  eval/           metrics, cost, frontier, harness; extractor_cascade (write path)
  baselines/      RoG / GCR / GNN-RAG wrappers — stubs
  utils/          HF loading, JSONL + optional W&B logging, seeding
scripts/          download_data / run_sft / run_grpo / run_backfill_pilot / eval_frontier / sweep
tests/            pytest suite (149 tests; every module above with pure-python coverage)
```

## Design notes (deviations from the brief, and why)

- **`Path.root`** — the brief's `Path` holds only `triples` and `current_node`
  raises when empty, yet the frontier is seeded from `question.topic_entities`. An
  unexpanded path has no triples, so it cannot report a position. We add an optional
  `root: Entity | None` to anchor the starting topic entity. `current_node` returns
  the tail of the last triple, else `root`, and still **raises for a truly empty
  path** (no triples *and* no root) — honoring the literal contract.
- **`controller` and `synth` config groups** — not in the brief's config tree, but
  `controller=dummy` appears in its example command. We add both so every swappable
  piece is a config choice (`hydra.utils.instantiate` via `_target_`).
- **`bind_store` hook** on `TraversalController` — a default no-op the engine calls
  once per question so a controller *may* consult the store (the `DummyController`'s
  highest-degree heuristic needs it). Neural controllers ignore it; `select()`'s
  signature stays `(state, candidates)`.
- **`dataset=sample`** — a tiny bundled dataset (same schema as the real releases)
  so `pytest` and `scripts/eval_frontier.sh` run fully offline. Real WebQSP/CWQ/MetaQA
  configs point at downloaded data via `scripts/download_data.sh`.

## License

Apache-2.0.
