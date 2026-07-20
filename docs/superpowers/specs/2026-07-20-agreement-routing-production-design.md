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
canonical baseline. The comparator is an immutable `confidence_baseline` policy
artifact, not a label in a report. It records the policy ID, confidence threshold,
expected escalation rate, code commit, environment manifest, dataset and outcomes
hashes, ordered decision-bitset hash, aggregate escalated input/output-token totals,
and quality metrics. Until a current production artifact exists, the migration
comparator is the documented markers policy at confidence `tau=0.85`; the canonical
rerun freezes its exact artifact before any candidate comparison.

The rerun must satisfy all of these checks:

- F1 and recall each reproduce within 0.002 absolute;
- escalation rate reproduces within 0.002 absolute;
- escalated call count matches exactly;
- ordered confidence decisions match exactly by SHA-256 bitset hash;
- aggregate escalated input and output token estimates reproduce within 0.5%; and
- split length/hash, adapter hash, vocabulary hash, prompt-marker behavior,
  truncation count, and maximum-triples contract match exactly.

If this gate fails, the run is diagnostic only. It cannot select a production
candidate or support a savings claim.

### Candidate comparison

Each candidate is evaluated at exact matched escalation budgets of 15%, 20%, and
25%, and across its full threshold frontier. For a budget `b` and `n` chunks,
exactly `floor(b*n)` chunks are escalated. Rows sort by ascending routing signal,
then stable Alphina chunk ID; this rule resolves ties. No interpolation is used for
gate decisions. Charts may interpolate for display only and must label interpolated
values as non-gating. Reports include micro precision, recall, F1, exact match,
escalation rate, routing AUROC, and per-bucket behavior.

At each of 15%, 20%, and 25%, the candidate must be non-inferior to confidence:

- candidate recall minus confidence recall is at least -0.002;
- candidate F1 minus confidence F1 is at least -0.002; and
- candidate exact match minus confidence exact match is at least -0.005.

It must also improve recall or F1 by at least 0.005 at one or more of the three
budgets. This prevents a candidate that merely ties confidence everywhere from
passing the robustness gate.

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
profile, and a frozen workload manifest for exactly 1.16M chunks. The workload
manifest contains stratum counts and input-token totals keyed by filer-volume
decile, positive/negative status when known, relation bucket when known, and
chunk-length bucket. Evaluation rows receive deterministic post-stratification
weights from those counts; empty evaluation strata are a hard validation failure.
It produces both machine-readable JSON and a concise human-readable report.

For policy `p`, the normative projected cost is:

`C(p) = C_gpu_fixed + C_gpu_per_chunk * 1,160,000 + Σ_s w_s * [E_p(I_s)*P_in + E_p(C_s)*P_cached + E_p(O_s)*P_out]`

where `s` ranges over evaluation chunks, `w_s` is its workload-manifest weight,
`I_s`, `C_s`, and `O_s` are uncached input, cached input, and teacher output tokens
for an escalation, `E_p` is the policy's 0/1 escalation decision, and `P_*` are
dated per-token prices. GPU fixed cost is measured model-load/setup wall time times
the dated GPU hourly price. GPU per-chunk cost is steady-state measured decode time
times that price. Costs are accumulated in decimal USD at full precision and only
rounded to cents for display.

The candidate threshold is selected by minimum projected cost subject to matching
or exceeding the frozen confidence baseline's recall and F1 and staying within the
exact-match tolerance. The candidate passes the savings gate only when the lower
bound of its savings percentage is at least 10%, where the bound is the 5th
percentile from 10,000 deterministic, stratum-preserving bootstrap resamples using
seed 42. Point-estimate savings are reported but are not sufficient to pass.

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

Pricing inputs must be dated, sourced, and versioned. Teacher token estimates come
from actual historical teacher-call telemetry for the same chunk when available;
otherwise they use the conservative 95th percentile of the matching stratum and
are marked imputed. Measured throughput takes precedence over estimates. The
command must not mix costs from different model, adapter, grammar, dataset,
workload, or pricing manifests.

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

### Alphina provenance envelope contract

Edgewright accepts a versioned envelope with these required fields:

- `schema_version`;
- `retrieval_run_id`;
- `source_id` and `document_id`;
- `source_revision` or `snapshot_id`;
- `chunk_id` and zero-based `chunk_index`;
- `retrieved_at` and `chunked_at` UTC timestamps;
- `content_sha256`, computed over the exact UTF-8 chunk bytes presented to
  Edgewright with no newline or Unicode normalization by Edgewright;
- `transform_chain_id`, referring to Alphina's pre-chunk transformation record;
  and
- `envelope_signature` plus `signing_key_id` when the negotiated schema requires
  signed envelopes.

Edgewright supports the current schema and one immediately preceding compatible
minor version. Unknown major versions fail closed. It recomputes `content_sha256`,
checks required fields and identifier formats, and verifies a required signature
against configured Alphina public keys. Alphina remains responsible for the truth
of retrieval lineage and for signing-key custody; Edgewright is responsible for
verification before processing.

Validation emits one or more stable reason codes: `missing_envelope`,
`unsupported_schema`, `missing_field`, `invalid_identifier`, `invalid_timestamp`,
`content_hash_mismatch`, `signature_required`, `unknown_signing_key`, or
`invalid_signature`. The write-path coordinator enforces the production-write block
before creating its graph-write outbox entry.

### Mandatory audit versus optional shadow telemetry

Maximum per-chunk auditability applies to the production write path. Before a graph
write becomes eligible, Edgewright transactionally appends an immutable decision
audit record and a graph-write outbox entry in the same durable transaction. The
graph writer consumes only committed outbox entries that reference a valid audit
record. Therefore a mandatory-audit failure fails the graph write closed.

Audit persistence retries with bounded exponential backoff, then places the record
in a durable dead-letter queue with its provenance identifiers and reason code.
Dead-lettered records are not graph-write eligible. Reprocessing is idempotent on
`(chunk_id, content_sha256, policy_id, extraction_run_id)`.

High-volume shadow metrics and traces are optional telemetry. Their failure does
not block extraction or an already-audited authoritative write, but increments a
durable telemetry-loss counter. A missing per-chunk shadow decision record excludes
that chunk from shadow promotion evidence.

## Failure behavior

- Missing or non-finite agreement values fall back to confidence and record a
  machine-readable reason.
- Model, adapter, vocabulary, grammar, or policy-artifact mismatch disables shadow
  evaluation for that chunk and emits telemetry.
- Missing or mismatched Alphina provenance blocks production writes. Shadow-only
  diagnostics are marked `provenance_incomplete` and excluded from promotion data.
- Optional shadow telemetry failure never blocks extraction and never changes
  authoritative escalation. Mandatory decision-audit failure blocks graph writes.
- Unexpected escalation drift, latency regression, schema incompatibility, or
  economics-profile mismatch marks the candidate ineligible for promotion.
- No failure path silently substitutes an unversioned threshold.

## Shadow acceptance and monitoring

Shadow observation runs for at least 10,000 eligible chunks and seven complete UTC
days; both minima must be met. The sample is the deterministic 1-in-N hash sample
of production `chunk_id`s needed to reach the volume target, plus all policy
disagreements. Eligible evidence requires valid provenance, complete mandatory and
shadow audit records, a recognized policy artifact, and no replay/test marker.

Coverage is adequate only when the sample covers strata representing at least 95%
of production volume and every gating stratum contains at least 100 chunks. Strata
below 100 are merged into a declared `other` bucket before the observation window;
post-hoc merging is forbidden. The production-versus-shadow population stability
index for chunk-length and filer-volume strata must be at most 0.10.

Monitoring covers:

- authoritative and candidate escalation rates;
- decision-disagreement matrix;
- projected teacher-call and dollar delta;
- extraction and routing latency;
- missing, invalid, fallback, and provenance-incomplete rates;
- drift by relation, filer, positive/negative chunk, and chunk-length bucket; and
- policy, adapter, grammar, and provenance schema versions.

Promotion eligibility requires:

- at least 10% projected total-cost reduction remains after observed workload mix;
- candidate escalation-rate drift from its artifact expectation is at most 2.0
  percentage points overall and 3.0 points in every gating stratum;
- Edgewright p95 processing-latency regression is at most 5% and p99 regression is
  at most 10% relative to the simultaneous confidence path;
- operational-error-rate increase is at most 0.1 percentage points, agreement
  fallback rate is at most 0.1%, mandatory audit completeness is 100%, and optional
  shadow record completeness is at least 99.9%;
- provenance-incomplete chunks are excluded and their rate is at most 0.1%;
- no gating stratum violates the pre-declared escalation-drift or coverage limits;
- complete, replayable audit records for the promotion evidence set; and
- a separately reviewed production-activation change.

Live shadow traffic has no complete counterfactual teacher labels and therefore
cannot claim extraction-quality improvement. Quality eligibility remains anchored
to the canonical labeled evaluation. During the shadow window that evaluation is
replayed once against the frozen candidate artifact; its decision hash and metrics
must reproduce exactly. Shadow traffic validates workload, cost, latency, failure,
and audit assumptions only.

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
