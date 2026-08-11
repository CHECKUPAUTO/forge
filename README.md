<img width="1024" height="572" alt="Forge" src="https://github.com/user-attachments/assets/3c1385de-87d3-4d5b-853c-82dee104635c" />

# Forge

Moteur de recherche évolutionnaire d'algorithmes **piloté par exécution**.

Forge génère ou mute des candidats, les compile, vérifie leur correction dans un harnais indépendant, mesure leurs performances réelles puis sélectionne les survivants. Le LLM est un moteur de proposition optionnel : la vérité du score vient de l'artefact exécuté.

## Architecture

- **forge-core** — moteur évolutionnaire, sélection Pareto, holdout, cache contextualisé, checkpoints, protocole distribué et domaines d'exécution `low_rank`, `simd_gemm`, `cuda_gemm`.
- **forge** — démonstration exécutable du moteur sur le bin-packing.
- **forge-worker** — worker TCP distribué pour l'évaluation distante de candidats.
- **forge-cli** — analytics du registre Sled et inspection des checkpoints.
- **forge-bridge** — façade Rust typée pour intégrer Forge à d'autres briques ; aucun service HTTP n'est actuellement fourni par ce crate.
- **forge-domains** — domaines légers/de démonstration séparés du noyau, dont un modèle Tensor Train paramétrique.

Le dépôt est un workspace Cargo unique ; `Cargo.lock` à la racine est la source de vérité pour le workspace.

## Domaines d'exécution du core

| Domaine | Candidat | Mesure principale |
|---|---|---|
| `simd_gemm` | fonction Rust GEMM | latence Criterion avec `target-cpu=native` |
| `cuda_gemm` | kernel CUDA natif | latence CUDA Events + taille instructionnelle PTX |
| `low_rank_compression` | `compress` / `reconstruct` Rust | erreur L2, latence, paramètres stockés |

Chaque domaine implémente le trait `Domain`. La porte `verify` est séparée de `measure` afin qu'un candidat rapide mais incorrect ne puisse pas obtenir un bon score de performance.

## Build et validation

Depuis la racine :

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --lib
```

Le chemin LLM/UCB1 est compilé séparément :

```bash
cargo check -p forge-core --all-targets --features bandit
```

La CI GitHub applique ces gates sur chaque pull request.

## LLM local

La feature `llm` active le mutateur Ollama. Les runners `run_campaign`, `run_simd` et `run_cuda` utilisent notamment :

```bash
OLLAMA_URL=http://localhost:11434/api/generate \
OLLAMA_MODEL=qwen2.5-coder:1.5b \
cargo run -p forge-core --features llm --bin run_simd
```

La feature `bandit` active le sélecteur UCB1 de stratégies de mutation et implique `llm` :

```bash
FORGE_MAB=1 cargo run -p forge-core --features bandit --bin run_simd
```

## Évaluation distribuée

Le Master et `forge-worker` utilisent un protocole bincode encadré par une longueur explicite. Les messages sont bornés et le Master vérifie notamment l'identité du candidat retourné et la validité numérique des objectifs.

Ce protocole **n'est pas un protocole de sécurité** : il ne fournit actuellement ni TLS, ni authentification, ni attestation cryptographique. Les workers doivent être considérés comme des évaluateurs de confiance et déployés sur un réseau de confiance ou derrière un tunnel authentifié.

## Sécurité du code généré

Forge applique des timeouts, des limites de ressources sur certains chemins et des filtres de capacités dans certains domaines. Ces mécanismes constituent une défense en profondeur, **pas un sandbox complet pour du code hostile**.

Pour exécuter des candidats non fiables, placez `forge-worker` dans une frontière d'isolation OS dédiée (VM/conteneur/utilisateur/cgroups/seccomp selon le modèle de menace), avec filesystem et réseau minimaux. Voir [SECURITY.md](SECURITY.md).

## Cache et reproductibilité

Les entrées de cache d'évaluation sont contextualisées par domaine, graine de trial et environnement. `FORGE_CACHE_ENV` permet d'ajouter une empreinte stable de toolchain/matériel lorsque des caches persistants sont partagés.

`base_seed` rend les chemins RNG du moteur déterministes, mais une campagne LLM n'est pas bit-exactement reproductible tant que le backend d'inférence lui-même n'est pas configuré de manière déterministe et identifié avec précision.

## Licence

Double licence : [PolyForm Noncommercial 1.0.0](LICENSE.md) pour les usages non commerciaux et personnels ; une licence commerciale séparée est requise pour tout usage commercial. Voir [LICENSING.md](LICENSING.md).
