# Forge repository agent instructions

Before repository changes, fetch and read the persistent off-main roadmap:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/FORGE_ECOSYSTEM_ROADMAP.yaml
```

For ML kernel, fusion, schedule, representation, attention, distributed evaluation, benchmark, or promotion work, also read:

```bash
git show origin/agent/ecosystem-roadmap:.agent/ML_MATURITY_5_OF_5.yaml
```

Treat root `AGENTS.md` as mandatory bootstrap policy. If the roadmap or applicable ML overlay is unavailable, fail closed for major search-policy, security, distributed-worker, cross-repository integration, candidate-promotion, or merge decisions.

Preserve Forge's execution-driven identity: candidates are proposed, compiled, independently verified, measured, then selected. LLM output is never authoritative evidence. A `5/5` claim requires the overlay's independent correctness/quality, isolation, environment-identity, distributed-worker and destination-promotion gates.
