//! Implémentations de domaines concrets pour forge-core.
//! Chaque domaine définit un terrain d'optimisation avec ses propres
//! contraintes de vérification et métriques de performance.

pub mod cuda_kernel;
pub mod low_rank;
pub mod simd_kernel;
