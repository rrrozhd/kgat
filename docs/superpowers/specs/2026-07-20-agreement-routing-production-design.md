# Edgewright Agreement Routing: Production Validation and Shadow Design

## Status

Approved conversational design. This specification defines the work required to
pre-validate grammar-agreement routing economically and then enable it in shadow
mode. It does not authorize agreement to control production escalation.

## Objective

Reduce Edgewright's end-to-end backfill cost while preserving or improving the
current confidence cascade's extraction quality. A candidate agreement policy may
enter shadow mode only after an exact canonical evaluation demonstrates:

1. reproducible confidence-baseline behavior;
2. equal-or-better recall and F1;
3. at least 10% projected total-cost reduction over the 1.16M-chunk backfill; and
4. robust performance across nearby escalation budgets.

The production pipeline remains confidence-controlled throughout shadow mode.

## Scope

This design includes:

- canonical GPU re-evaluation of all agreement signals;
- a versioned full-backfill economics model;
- selection and freezing of a candidate routing policy;
- a production-safe shadow router with confidence fallback;
- Edgewright-side decision provenance and audit records;
- monitoring and promotion-eligibility reporting; and
- corrected reproduction documentation.

This design excludes:

- activation of agreement as the authoritative production router;
- changes to the extractor model or teacher;
- retraining a standalone gate or routing policy;
- retrieval provenance implementation inside Edgewright; and
- unrelated package renaming. `kgat` may remain the Python module name while the
  product and documentation use Edgewright.

## Ownership and provenance boundary

Alphina owns retrieval provenance and chain of custody. Its provenance envelope is
the authoritative record of where a chunk came from. Alphina is responsible for:

- source and document identity;
- source revision or snapshot identity;
- retrieval query and access-path lineage;
- retrieval-run identity and timestamps;
- raw-content retention and transformations before chunking;
- chunk identity and content hashes; and
- source-authenticity and chain-of-custody guarantees.

Edgewright must require and propagate the Alphina envelope. Edgewright owns the
audit trail for what it does with the chunk, including:

- extraction model, base-model revision, adapter hash, and tokenizer revision;
- grammar mode, vocabulary hash, entity-marker contract, and `max_triples`;
- emitted triples and decode diagnostics;
- confidence, agreement, and minimum-agreement values;
- authoritative and shadow policy identifiers and thresholds;
- both routing decisions and their timestamps;
- fallback reasons and validation status; and
- downstream write or rejection outcome.

Edgewright must not claim to establish source authenticity or retrieval lineage.
It links its decision record to Alphina's envelope and echoes the stable identifiers
and content hash needed to verify that linkage.

Missing, malformed, or hash-mismatched provenance blocks production writes and
creates an auditable rejection. Shadow scoring may proceed only when the record is
explicitly marked `provenance_incomplete`; it may not be counted as promotion
evidence.

## Canonical pre-validation

### Fixed evaluation contract

The canonical evaluation uses:

- branch/release containing the grammar-agreement scorer;
- `Qwen/Qwen3-0.6B` with an immutable base-model revision;
- `qwen3-0.6b-extractor-markers` adapter with recorded SHA-256 hashes;
- `real-strat-v2` test split with exactly 2,193 chunks and recorded hashes;
- closed-vocabulary grammar with the 8,993-target vocabulary;
- entity markers from the adapter contract;
- `max_triples=28`;
- the historical normalization and loose-match accounting;
- Torch, Transformers, PEFT, CUDA, and driver versions recorded in the manifest;
- deterministic decoding; and
- all signals: confidence, agreement, minimum agreement, confidence × agreement,
  and confidence × minimum agreement.

No CLI default may silently determine a production-relevant setting. The run
manifest records every item above.

### Reproducibility gate

Before agreement results are interpreted, confidence must reproduce the historical
canonical baseline within an absolute F1 tolerance of 0.002. Split length, split
hash, adapter hash, vocabulary hash, prompt-marker behavior, truncation count, and
maximum-triples contract must also match.

If this gate fails, the run is diagnostic only. It cannot select a production
candidate or support a savings claim.

### Candidate comparison

Each candidate is evaluated at matched escalation budgets near 15%, 20%, and 25%,
and across its full threshold frontier. Reports include micro precision, recall,
F1, exact match, escalation rate, routing AUROC, and per-bucket behavior.

The selected operating point must satisfy all of the following against confidence:

- recall is equal or higher;
- F1 is equal or higher;
- exact match regresses by no more than 0.005 absolute;
- total projected cost falls by at least 10%; and
- the result is not an isolated threshold accident: the candidate remains
  competitive across the neighboring budget points.

If no signal passes every gate, confidence remains the sole production policy and
no shadow candidate is enabled.

## Full-backfill economics model

The economics command consumes an immutable outcomes artifact, a versioned pricing
profile, and workload assumptions for the 1.16M-chunk backfill. It produces both
machine-readable JSON and a concise human-readable report.

The report separates:

- fixed and variable GPU decode cost;
- number and token volume of teacher escalations;
- teacher API input, cached-input, and output cost where applicable;
- total cost for the confidence baseline and each candidate;
- absolute and percentage savings;
- quality at matched cost;
- cost at matched quality; and
- sensitivity to teacher price, average token volume, GPU throughput, and workload
  mix.

Pricing inputs must be dated, sourced, and versioned. Measured throughput takes
precedence over estimates. The command must not mix costs from different model,
adapter, grammar, or dataset manifests.

## Candidate policy artifact

A passing candidate is frozen as a versioned, immutable policy artifact containing:

- policy ID and schema version;
- selected signal and comparison direction;
- numeric threshold and expected escalation rate;
- evaluation dataset and outcomes hashes;
- model, adapter, tokenizer, grammar, vocabulary, and environment identities;
- baseline and candidate quality metrics;
- economics-profile identity and projected savings;
- creation timestamp and code commit; and
- explicit status: `shadow_only`.

Production code loads this artifact rather than reconstructing policy settings from
CLI defaults or documentation.

## Shadow-mode architecture

For each eligible chunk, Edgewright performs one extractor decode and calculates
all routing signals. Two policy paths then run on the same record:

1. **Authoritative path:** the existing confidence policy determines whether the
   chunk escalates.
2. **Shadow path:** the frozen candidate computes the decision it would have made.

Only the authoritative result is returned to the live pipeline. Shadow evaluation
must not make a teacher call, suppress an authoritative call, write graph data, or
change user-visible behavior.

The audit record includes the Alphina provenance link, all signal values, both
decisions, both policy identities, projected teacher-call delta, latency, and any
fallback or exclusion reason.

## Failure behavior

- Missing or non-finite agreement values fall back to confidence and record a
  machine-readable reason.
- Model, adapter, vocabulary, grammar, or policy-artifact mismatch disables shadow
  evaluation for that chunk and emits telemetry.
- Missing or mismatched Alphina provenance blocks production writes. Shadow-only
  diagnostics are marked `provenance_incomplete` and excluded from promotion data.
- Shadow logging failure never blocks extraction and never changes authoritative
  escalation.
- Unexpected escalation drift, latency regression, schema incompatibility, or
  economics-profile mismatch marks the candidate ineligible for promotion.
- No failure path silently substitutes an unversioned threshold.

## Shadow acceptance and monitoring

Shadow observation runs for at least 10,000 representative chunks or seven days,
whichever occurs later. Monitoring covers:

- authoritative and candidate escalation rates;
- decision-disagreement matrix;
- projected teacher-call and dollar delta;
- extraction and routing latency;
- missing, invalid, fallback, and provenance-incomplete rates;
- drift by relation, filer, positive/negative chunk, and chunk-length bucket; and
- policy, adapter, grammar, and provenance schema versions.

Promotion eligibility requires:

- at least 10% projected total-cost reduction remains after observed workload mix;
- no material latency or operational-error regression;
- no unexplained bucket-level degradation or escalation drift;
- complete, replayable audit records for the promotion evidence set; and
- a separately reviewed production-activation change.

Passing shadow gates does not activate the candidate automatically.

## Testing strategy

Unit tests cover:

- signal calculation and finite-value validation;
- threshold boundary behavior;
- policy serialization, hashing, and schema validation;
- confidence fallback and reason codes;
- economics calculations and sensitivity cases;
- matched-cost and matched-quality candidate selection;
- Alphina provenance-envelope validation and propagation; and
- telemetry serialization and redaction constraints.

Integration tests cover:

- canonical manifest validation;
- rejection of mismatched dataset, adapter, vocabulary, or `max_triples` settings;
- end-to-end shadow evaluation with authoritative confidence output unchanged;
- logging failure without live-routing impact;
- missing provenance blocking a production write; and
- replay of a chunk from its Alphina envelope through both policy decisions.

The canonical GPU result and economics report are retained as release artifacts,
with hashes verified after transfer and before pod termination.

## Deliverables

1. Corrected canonical runbook and immutable evaluation manifest.
2. Canonical max-triples-28 outcomes and per-signal frontier artifacts.
3. Versioned 1.16M-chunk economics profile and comparison report.
4. Candidate-selection report or an explicit no-go result.
5. Versioned `shadow_only` policy artifact when all offline gates pass.
6. Shadow router, audit schema, monitoring metrics, and tests.
7. Edgewright/Alphina provenance integration contract.
8. Shadow-readiness report documenting every gate and artifact hash.

## Explicit production decision

This phase ends with either:

- **NO-GO:** confidence remains authoritative and no candidate runs in shadow; or
- **SHADOW-READY:** confidence remains authoritative and one frozen candidate runs
  in non-effecting shadow mode.

Making agreement authoritative requires a separate production-activation design,
review, and approval after the shadow evidence is complete.
