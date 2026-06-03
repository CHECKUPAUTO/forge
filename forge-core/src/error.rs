//! Type d'erreur unique du moteur. Aucune dependance, aucun panic.

use std::fmt;

/// Erreurs remontees par le moteur, les domaines et le generateur.
#[derive(Debug)]
pub enum ForgeError {
    /// Un candidat est structurellement invalide (avant meme l'evaluation).
    InvalidCandidate(String),
    /// Echec pendant la phase de mesure / verification.
    Evaluation(String),
    /// Echec pendant la generation (mutation, seed, appel LLM).
    Generation(String),
    /// Echec pendant une mutation LLM (prompt, reponse, extraction de code).
    Mutation(String),
}

impl fmt::Display for ForgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForgeError::InvalidCandidate(m) => write!(f, "candidat invalide: {m}"),
            ForgeError::Evaluation(m) => write!(f, "evaluation: {m}"),
            ForgeError::Generation(m) => write!(f, "generation: {m}"),
            ForgeError::Mutation(m) => write!(f, "mutation: {m}"),
        }
    }
}

impl std::error::Error for ForgeError {}

/// Alias de resultat du crate.
pub type Result<T> = std::result::Result<T, ForgeError>;
