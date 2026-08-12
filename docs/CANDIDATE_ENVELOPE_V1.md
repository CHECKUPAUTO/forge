# Candidate Envelope v1

Forge emits an architecture-neutral candidate envelope for the Memorithm
Research-Lab execution pipeline without depending on CCOS or RSI.

The receiving contract is:

```text
Forge Candidate
    |
    v
CandidateEnvelopeV1
    |
    v
wire JSON
    |
    v
CCOS Research-Lab / sealed evaluator
```

## Identity

The candidate content identity is:

```text
SHA256(
    "memorithm.candidate.identity.v1\0"
    || origin_tag(Forge = 1)
    || len64le(domain) || domain
    || len64le(source_sha256) || source_sha256
)
```

`source_sha256` is SHA-256 of `Candidate::repr()` bytes.

The candidate id deliberately excludes trial seed and lineage. The same source in
two experiments is the same candidate content. The envelope fingerprint binds the
experiment-specific fields.

## Envelope fingerprint

The binary canonical encoding is:

```text
"memorithm.candidate-envelope.v1\0"
schema_version:u16le
string(candidate_id)
optional_string(producer_candidate_id)
optional_string(parent_candidate_id)
origin_tag:u8
string(domain)
string(source_sha256)
optional_string(proposal_sha256)
trial_seed:u64le
```

where `string(x) = len(x):u64le || UTF-8(x)` and optional strings use a one-byte
presence tag (`0` absent, `1` present) before the string.

The envelope fingerprint is SHA-256 of this binary representation.

## JSON transport

JSON is transport, not fingerprint material. Keys are emitted in deterministic
lexicographic order. `trial_seed` is a decimal string rather than a JSON number so
receivers that parse numbers through IEEE-754 cannot lose `u64` precision.

SHA-256 values are exactly 64 lowercase hexadecimal characters.

## Cross-repository golden vector

Input:

```text
repr                 = "pub fn kernel() {}"
producer_candidate_id = "42"
domain               = "simd_gemm"
parent               = null
proposal_sha256       = 11 repeated 32 bytes
trial_seed            = 18446744073709551615
```

Expected:

```text
source_sha256 = 3b6e6e212c45273719067e12eac78aceaf44fbb2ffcafef4ab4519a64c5083e1
candidate_id  = 4457784cc3119a48ab2f90fbac86d5e5c1ab0c99b46b567edd8dbd1bb3a3446f
fingerprint   = 9a531d78fbf991077c087bdac953db53b1ede544349a71c5e6bdbe25f00e8693
```

The same vector is pinned in `CCOS-Research-Lab/crates/ccos-rsi/tests`.

## Security boundary

This envelope proves content identity and detects transport mutation when the
receiver verifies its canonical fingerprint. It is not a signature and does not
authenticate the producer. Worker authentication and signed evaluation receipts
are later protocol layers.
