# Forge Agent Bootstrap Contract

Before autonomous coding, candidate-domain work, security/trust changes, distributed-worker changes, cross-repository integration, candidate promotion, PR creation, or merge decisions, read:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/FORGE_ECOSYSTEM_ROADMAP.yaml
```

For any ML kernel, fusion, schedule, representation, attention, distributed evaluation, benchmark, or promotion work, also read:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/ML_MATURITY_5_OF_5.yaml
```

For any generated native kernel, CUDA-agent-style search loop, NNIS backend integration, numerical-contract comparison, kernel verification/measurement envelope, or kernel candidate promotion work, also read:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/KERNEL_AGENT_PROGRAM.yaml
```

The kernel-agent program preserves the execution-driven order `PROPOSE/MUTATE -> COMPILE -> VERIFY -> MEASURE -> SELECT`. Baseline and candidate precision semantics must be explicit and comparable; TF32, mixed precision, relaxed tolerances, or other numerical-policy changes are distinct campaigns rather than hidden speedups. Destination repositories retain independent correctness and final promotion authority.

The ML maturity overlay makes 5/5 an evidence-backed exit criterion. Candidate generation, LLM preference, compilation success, or microbenchmark speed never replace independent correctness/quality verification, trustworthy measurement identity, sandbox/trust requirements, or destination-repository qualification.

If the roadmap or an applicable overlay cannot be fetched or read, fail closed for major search-policy, security, cross-repository integration, candidate-promotion, or merge decisions. Read-only diagnosis is allowed.

## Core identity

Forge is an execution-driven algorithm-search engine:

`PROPOSE/MUTATE -> COMPILE -> VERIFY -> MEASURE -> SELECT`

The LLM is optional proposal machinery. Executed evidence is authoritative. A candidate that fails or lacks independent verification cannot survive regardless of performance.

Do not turn Forge into SciRust, ElasticXxx, FLAT-ATTENTION, NNIS, SLHAv2, SciRust Hub, SciRust-Verify, or SciCapsule. Those repositories own their domain semantics; Forge owns search.

Generated-code resource limits are not automatically a hostile-code sandbox. Remote worker transport is not automatically trusted merely because it is reachable.

Required CI must be green on the exact PR head before merge. A 5/5 maturity claim additionally requires the applicable end-to-end, isolation, distributed-worker, environment-identity and evidence gates in the ML overlay.

Reread the roadmap and applicable overlays at every session start, before new security/distributed/domain phases, before cross-repository work, after strategy/trust/ML-priority changes, and before promotion or merge decisions.

Do not merge the roadmap or ML/research overlays themselves into `main` unless the user explicitly requests it.
