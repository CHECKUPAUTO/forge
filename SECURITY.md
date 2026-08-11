# Forge Security Model

Forge compiles and executes candidate code. That execution model is intentionally powerful and must be treated as a security boundary concern, not as ordinary input parsing.

## Candidate code is untrusted

`forge-core` applies execution timeouts and, for selected domains, POSIX resource limits. Some domains also reject obvious capabilities such as filesystem, networking, process spawning, global state, `unsafe`, and FFI in generated source.

These controls are **defense in depth only**. They are not a complete sandbox against hostile code.

For candidates that are not fully trusted, run Forge workers inside an external operating-system isolation boundary appropriate to your threat model, for example a dedicated VM/container/user, cgroups and syscall restrictions, with a minimal filesystem and no network access unless explicitly required.

Do not run arbitrary third-party candidate code directly on a production host merely because it passes Forge source filters.

## Master / Worker trust

The current Master/Worker protocol is a length-prefixed bincode protocol over TCP. It validates framing, message size, candidate identity, and finite objective values, but it does **not** provide TLS, authentication, or cryptographic execution attestation.

Therefore:

- deploy workers only on a trusted network or behind an authenticated encrypted tunnel/VPN;
- treat configured workers as trusted evaluators;
- do not expose a Forge worker directly to an untrusted network;
- do not interpret a remote worker score as cryptographic proof that the claimed execution occurred.

## Resource isolation

Timeouts and `rlimit` reduce accidental damage from runaway candidates. They do not cover every resource or kernel capability and do not replace OS-level isolation.

CUDA and other accelerator-backed domains also depend on the isolation and reset guarantees of the underlying driver/runtime. Use dedicated workers for experiments that may destabilize an accelerator process.

## Measurement integrity

Forge separates correctness verification from performance measurement. Evaluation cache entries are scoped by domain, trial seed, Forge version, OS/architecture and an optional `FORGE_CACHE_ENV` fingerprint. Set `FORGE_CACHE_ENV` to a stable identifier for the relevant toolchain/hardware configuration when persistent benchmark caches are shared across runs.

When changing compiler flags, toolchain versions, CPU/GPU configuration, benchmark harnesses or relevant environment settings, use a different `FORGE_CACHE_ENV` or a fresh cache.

## Reporting vulnerabilities

Do not open a public issue for a vulnerability involving private infrastructure, credentials, or a practical sandbox escape. Report it privately to the repository maintainers.
