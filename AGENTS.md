# Forge Agent Bootstrap Contract

Before autonomous coding, candidate-domain work, security/trust changes, distributed-worker changes, cross-repository integration, candidate promotion, PR creation, or merge decisions, read:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/FORGE_ECOSYSTEM_ROADMAP.yaml
```

If the roadmap cannot be fetched or read, fail closed for major search-policy, security, cross-repository integration, candidate-promotion, or merge decisions. Read-only diagnosis is allowed.

## Core identity

Forge is an execution-driven algorithm-search engine:

`PROPOSE/MUTATE -> COMPILE -> VERIFY -> MEASURE -> SELECT`

The LLM is optional proposal machinery. Executed evidence is authoritative. A candidate that fails or lacks independent verification cannot survive regardless of performance.

Do not turn Forge into SciRust, ElasticXxx, FLAT-ATTENTION, NNIS, SLHAv2, SciRust Hub, SciRust-Verify, or SciCapsule. Those repositories own their domain semantics; Forge owns search.

Generated-code resource limits are not automatically a hostile-code sandbox. Remote worker transport is not automatically trusted merely because it is reachable.

Required CI must be green on the exact PR head before merge.

Reread the roadmap at every session start, before new security/distributed/domain phases, before cross-repository work, after strategy/trust changes, and before promotion or merge decisions.

Do not merge the roadmap itself into `main` unless the user explicitly requests it.
