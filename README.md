# Forge

Moteur de recherche évolutionnaire d'algorithmes piloté par exécution.

Forge utilise l'évolution génétique combinée à du LLM-guided mutation pour
découvrir et optimiser des algorithmes, en conditions réelles d'exécution.

## Crates

- **forge-core** — Moteur d'évolution, domaines, isolation sandbox
- **forge** — CLI et orchestration
- **forge-bridge** — Pont HTTP vers SoulSystem
- **forge-cli** — Interface ligne de commande
- **forge-domains** — Domaines spécialisés (low-rank tensor, SIMD gemm, CUDA)
- **forge-worker** — Worker distribué

## Domaines

| Domaine | Description |
|---------|-------------|
| `simd_gemm` | Optimisation GEMM pour auto-vectorisation LLVM |
| `cuda_kernel` | Optimisation kernels CUDA natifs |
| `low_rank` | Factorisation tensorielle low-rank |

## Build

```bash
cargo build --release
cargo test --workspace
```
