//! Domains d'évaluation pour forge-core.
//!
//! Chaque module implémente le trait [`forge_core::Domain`] pour un terrain
//! d'optimisation spécifique.
//!
//! ## Domaines disponibles
//! - [`tensor_train`] : Compression Tensor Train (Low-Rank) — optimise les rangs
//!   pour maximiser le ratio de compression tout en minimisant l'erreur de
//!   reconstruction.

pub mod tensor_train;
