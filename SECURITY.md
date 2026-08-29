# Forge Security Model

Forge compiles and executes candidate code. That execution model is intentionally powerful and must be treated as a security boundary concern, not as ordinary input parsing.

## Candidate code is untrusted

`forge-core` applies execution timeouts and, for selected domains, POSIX resource limits. Some domains also reject obvious capabilities such as filesystem, networking, process spawning, global state, `unsafe`, and FFI in generated source.

These controls are **defense in depth only**. They are not a complete sandbox against hostile code.

For candidates that are not fully trusted, run Forge workers inside an external operating-system isolation boundary appropriate to your threat model, for example a dedicated VM/container/user, cgroups and syscall restrictions, with a minimal filesystem and no network access unless explicitly required.

Do not run arbitrary third-party candidate code directly on a production host merely because it passes Forge source filters.

## Master / Worker trust

The current Master/Worker transport is a length-prefixed bincode protocol over TCP. Frames are bounded. The master rejects candidate-ID mismatches, source-hash mismatches, unsupported result-envelope versions, non-finite/empty valid objectives, unknown benchmark-protocol identifiers, and results without a declared execution context.

The worker result envelope records:

- the result protocol version;
- an independent FNV-1a hash of the exact source received;
- the domain actually used by the worker;
- the verify-then-measure benchmark protocol identifier;
- worker identity, toolchain, OS, architecture, hardware description, and an environment fingerprint.

This is **provenance binding, not cryptographic attestation**. FNV-1a is used only as a deterministic content identity/check, not as a signature or security primitive.

The protocol still does **not** provide TLS, worker authentication, or cryptographic proof that the claimed execution occurred. Therefore:

- deploy workers only on a trusted network or behind an authenticated encrypted tunnel/VPN;
- treat configured workers as trusted evaluators;
- do not expose a Forge worker directly to an untrusted network;
- do not interpret a remote worker score as cryptographic proof that the claimed execution occurred.

## Resource isolation

Timeouts and `rlimit` reduce accidental damage from runaway candidates. They do not cover every resource or kernel capability and do not replace OS-level isolation.

CUDA and other accelerator-backed domains also depend on the isolation and reset guarantees of the underlying driver/runtime. Use dedicated workers for experiments that may destabilize an accelerator process.

## Measurement integrity

Forge separates correctness verification from performance measurement. Remote workers now return an explicit execution-context fingerprint with every result so operators can retain the hardware/toolchain provenance of measurements.

Persistent cache reuse is conservative: `EvaluationCache` does not return cached scores unless an explicit context identity is supplied through `FORGE_CACHE_ENV` or `with_environment_fingerprint`. This prevents the default configuration from silently reusing a benchmark score when the master cannot prove that the relevant hardware/toolchain/benchmark context is the same. Cache records remain scoped by domain, trial seed, candidate identity, Forge version, OS/architecture and the explicit environment identity.

For a homogeneous worker pool, set `FORGE_CACHE_ENV` to a stable identifier that uniquely represents the complete benchmark context (toolchain, compiler flags, CPU/GPU model/configuration, benchmark harness and relevant environment). For heterogeneous workers, use distinct context identities or leave reuse disabled.

## Reporting vulnerabilities

Do not open a public issue for a vulnerability involving private infrastructure, credentials, or a practical sandbox escape. Report it privately to the repository maintainers.
