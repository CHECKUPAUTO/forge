//! Sous-système de mutation pour forge-core.
//!
//! La mutation est le moteur de variation qui génère de nouveaux candidats
//! à partir des élites de la génération courante. Deux stratégies coexistent :
//! - [`LlmMutator`] : mutation macroscopique via inférence LLM locale (Ollama)
//! - [`micro_mutator`](crate::micro_mutator) : mutations fines déterministes de constantes

#[cfg(feature = "llm")]
pub mod llm_mutator;
