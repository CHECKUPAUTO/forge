#![deny(unsafe_code)]
//! # forge-core
//!
//! Moteur de recherche evolutionnaire d'algorithmes **pilote par execution**.
//! Le LLM (ou une mutation) propose, le domaine compile/mesure/verifie sur le
//! terrain reel, le moteur selectionne et fait evoluer. La verite vient de
//! l'artefact execute, pas d'un raisonnement : c'est tout l'interet.
//!
//! ## Forme
//! - [`Domain`] : la frontiere d'extension. Une campagne = une implementation.
//!   Les 4 cibles (compression, quantification, kernels SIMD/GPU, routage MoE)
//!   sont 4 `Domain` independants ; le moteur n'en connait aucun.
//! - [`Engine`] : la boucle generique (seed -> evaluation -> archive -> mutation),
//!   avec rotation des entrees (anti-overfit) et validation holdout finale.
//! - Anti-triche : la porte de correction ([`Domain::verify`]) est separee de la
//!   mesure ([`Domain::measure`]) ; le candidat ne calcule jamais son score.
//!
//! ## 4 Piliers industriels
//! - [`diagnostics`] : capture des échecs (Pilier 1) + parsing Criterion (Pilier 2)
//! - [`isolation::run_with_secure_limits`] : double verrou timeout + rlimit Posix
//! - [`registry::AlgorithmRegistry`] : registre transactionnel Sled avec lignage (Pilier 4)
//! - [`domains::simd_kernel`] : micro-kernels SIMD auto-vectorisés (Pilier 3)
//! - [`mutation::llm_mutator`] : mutation macroscopique par LLM avec injection de feedback
//!
//! ## Modules auxiliaires
//! - [`cache::EvaluationCache`] : interception des candidats redondants
//! - [`micro_mutator::MicroMutator`] : mutations fines de constantes
//! - [`protocol`] : communication Master/Worker pour évaluation distribuée
//! - [`criterion_parser`] : parsing avancé des métriques Criterion
//!
//! La feature `llm` ajoute un client Ollama ([`llm::ollama_generate`]) pour le
//! generateur de candidats des domaines "code".

pub mod cache;
mod candidate;
pub mod criterion_parser;
pub mod diagnostics;
mod domain;
pub mod domains;
mod error;
mod evolve;
pub mod isolation;
pub mod micro_mutator;
pub mod mutation;
pub mod protocol;
pub mod registry;
mod trial;

pub mod report_util;

#[cfg(feature = "llm")]
pub mod llm;

pub use candidate::{fnv1a, Candidate, CandidateId};
pub use diagnostics::{FailureDiagnostics, FailureStage};
pub use domain::{Domain, Score};
pub use error::{ForgeError, Result};
pub use evolve::{
    evaluate_parallel_distributed, evaluate_with_feedback, sort_by_pareto_domination, Config,
    DeserializeFromSource, Engine, EngineState, Individual, Report,
};
pub use isolation::run_with_timeout;
pub use trial::Trial;
