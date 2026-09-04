# Forge SOUP post-training runner v1

Status: process contract for the already-published `forge_bridge::soup_posttrain` domain.

Binary: `forge-soup-posttrain`

Campaign schema: `1`

Underlying SOUP domain source merge: `1385c71a541419f15a558a5e94bc8a4a60567a4a` (Forge PR #25).

Qualified SOUP source used by that domain: `05b646523727925990530667e7012ede50bd30b2`.

## Ownership boundary

Forge owns candidate generation/mutation, verify-before-measure ordering, Pareto selection, campaign seeds and the search report. The external evaluator owns verification evidence and objective measurements. SOUP or a Hub-qualified SOUP adapter remains authoritative for training, evaluation, model loading and metric semantics.

The runner never fabricates quality, latency, VRAM or other metrics. It consumes only evaluator evidence accepted by `SoupPostTrainDomain`.

## Invocation

```text
forge-soup-posttrain \
  --campaign /path/to/campaign.json \
  --evaluator /absolute/path/to/evaluator \
  --evaluator-arg fixed-arg \
  --output /path/to/report.json
```

Use `--isolation-available` only when an external isolation boundary satisfying the campaign's declared policy is actually present. The flag is an assertion by the caller; the runner does not create or attest a sandbox.

`--max-response-bytes` changes the bounded evaluator response limit. The underlying evaluator contract accepts 1..=16 MiB.

The evaluator path must be absolute and both campaign/evaluator paths must be regular non-symlink files. Evaluator execution uses structured argv and JSON stdin/stdout; no shell is involved.

## Campaign envelope

The top-level JSON object is fail-closed on unknown fields:

```json
{
  "schema_version": 1,
  "external_domain": {
    "schema_version": 1,
    "domain_id": "soup/posttrain-v1",
    "upstream": {
      "repository": "MakazhanAlpamys/Soup",
      "commit_id": "<exact-qualified-git-object>",
      "contract_sha256": "<64-lowercase-hex>"
    },
    "allowed_candidate_dimensions": ["recipe.example"],
    "data_boundary": {
      "generation_sources": ["train-id"],
      "verification_sources": ["validation-id"],
      "final_holdout_sources": ["holdout-id"]
    },
    "verification": {
      "adapter_id": "<stable-adapter-id>",
      "adapter_sha256": "<64-lowercase-hex>"
    },
    "objectives": [
      {"name": "task_quality", "direction": "maximize"},
      {"name": "wall_ms", "direction": "minimize"}
    ],
    "environment": {
      "fingerprint_required": true,
      "isolation_required": false
    }
  },
  "dimensions": {
    "recipe.example": ["a", "b"]
  },
  "baseline": {
    "recipe.example": "a"
  },
  "engine": {
    "generations": 4,
    "population": 8,
    "survivors": 2,
    "base_seed": 7
  }
}
```

The concrete dimension names and values are contract/caller supplied. Forge does not hardcode unqualified SOUP recipe fields as semantic truth.

The runner validates:

- campaign schema version;
- the existing `ExternalDomainManifestV1` contract and holdout separation;
- exact agreement between allowed candidate dimensions and search-space dimensions;
- baseline values belong to the declared search space;
- `generations` in `1..=10000`;
- `population` in `1..=4096`;
- `survivors` in `1..=population`;
- required external isolation before campaign execution.

The v1 runner always sets Forge `worker_addresses` to `None`. Distributed campaign transport/trust remains a separate FG1/FG6 concern.

## Evaluator protocol

For each candidate/trial the evaluator receives one `SoupEvaluatorRequest` JSON object on stdin with `schema_version=1` and phase `verify` or `measure`.

Verification must return candidate/trial-bound `SoupVerificationEvidence`. Measurement is called only after verification passes and must return `SoupMeasurementEvidence` whose metric-name set exactly matches the external-domain objective set. Required environment fingerprints must be non-empty. Identity mismatches, missing metrics, extra metrics, non-finite values or unsupported schema versions fail closed.

## Report

The output is JSON schema v1 and records:

- exact upstream repository/commit/contract hash;
- verification adapter identity/hash;
- engine seed and search sizes;
- best candidate and final Pareto front;
- baseline and holdout scores exposed by the Forge engine;
- campaign history and failure diagnostics;
- each objective's original direction/value plus the minimization-normalized value actually used by Forge.

A `maximize` objective is sign-normalized only internally for Forge's minimization-oriented Pareto comparison; the report restores the original sign and also exposes `forge_minimized_value`.

## Non-claims

This process contract does not establish a hostile-code sandbox, authenticated distributed workers, SOUP metric semantics, model quality, hardware performance, or automatic promotion of a winning recipe. Destination/runtime requalification remains required.
