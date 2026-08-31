# Forge repository agent instructions

Before repository changes, fetch and read the persistent off-main roadmap:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/FORGE_ECOSYSTEM_ROADMAP.yaml
```

Treat root `AGENTS.md` as mandatory bootstrap policy. If the roadmap is unavailable, fail closed for major search-policy, security, distributed-worker, cross-repository integration, candidate-promotion, or merge decisions.

Preserve Forge's execution-driven identity: candidates are proposed, compiled, independently verified, measured, then selected. LLM output is never authoritative evidence.
