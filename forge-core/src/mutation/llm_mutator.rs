//! Pipeline d'inférence locale pour la mutation d'algorithmes et la correction
//! d'erreurs par boucle de rétroaction (self-correction loop).
//!
//! ## Flux
//! 1. Reçoit le code parent et un éventuel [`FailureDiagnostics`]
//! 2. Construit un prompt système enrichi (code + traces d'erreur si échec)
//! 3. Appelle l'API Ollama locale (compatible OpenAI) via `ureq`
//! 4. Extrait le code Rust du bloc markdown ```rust ... ```
//! 5. Retourne le code mutant propre, prêt à être compilé
//!
//! ## Dépendance
//! Activé par la feature `llm` (utilise `ureq` pour les appels HTTP bloquants).

use std::time::Duration;

use crate::diagnostics::FailureDiagnostics;
use crate::error::{ForgeError, Result};

/// Client de mutation LLM pour l'inférence locale (Ollama).
#[derive(Clone)]
pub struct LlmMutator {
    endpoint: String,
    model_name: String,
    timeout_secs: u64,
    objective: String,
}

impl LlmMutator {
    /// Crée un nouveau mutateur pointant vers une API Ollama.
    ///
    /// # Arguments
    /// * `endpoint` — URL de l'API (ex: `http://localhost:11434/api/generate`)
    /// * `model_name` — nom du modèle (ex: `codellama:7b`, `deepseek-coder:6.7b`)
    pub fn new(endpoint: &str, model_name: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            model_name: model_name.to_string(),
            timeout_secs: 60,
            objective: String::new(),
        }
    }

    /// Définit le timeout des appels HTTP.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    pub fn with_objective(mut self, objective: &str) -> Self {
        self.objective = objective.to_string();
        self
    }

    /// Formule l'invite système et injecte le code parent ainsi que les
    /// diagnostics d'échec si présents. Retourne le code Rust mutant extrait
    /// du bloc markdown.
    pub fn mutate_with_feedback(
        &self,
        parent_code: &str,
        diagnostics: Option<&FailureDiagnostics>,
    ) -> Result<String> {
        let mut prompt = format!(
            "Voici un algorithme Rust performant a optimiser :\n\n\
             ```rust\n{parent_code}\n```\n\n\
             Propose une version modifiee et optimisee du code, en conservant toutes les fonctions publiques. \
             Respecte imperativement les signatures publiques existantes. \
             Renvoie UNIQUEMENT le code Rust valide enveloppe dans un bloc \
             markdown ```rust ... ```.",
        );

        // Injection de la boucle de rétroaction (Self-Correction Loop)
        if let Some(diag) = diagnostics {
            let stage_desc = match diag.stage {
                crate::diagnostics::FailureStage::Compilation => {
                    "Compilation (cargo build/check)"
                }
                crate::diagnostics::FailureStage::Execution => {
                    "Execution (Benchmark runtime)"
                }
                crate::diagnostics::FailureStage::Verification => {
                    "Verification Mathematique (cargo run)"
                }
            };

            prompt.push_str(&format!(
                "\n\n⚠️ ATTENTION - Ta tentative precedente a ECHOUE a l'etape [{}].\n\
                 Traces d'erreurs retournees par le compilateur ou le superviseur d'isolation :\n\
                 ```\n{}\n```\n\
                 Analyse precisement cette erreur (probleme de borne, de type, \
                 de lifetime ou de convergence) et corrige-la de maniere rigoureuse.",
                stage_desc, diag.stderr
            ));
        }

        if !self.objective.is_empty() {
            prompt = format!("OBJECTIF (CRITIQUE) : {}\n\n{}", self.objective, prompt);
        }
        let body = ureq::json!({
            "model": self.model_name,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.5,
                "top_p": 0.9
            }
        });

        let response: serde_json::Value = ureq::post(&self.endpoint)
            .timeout(Duration::from_secs(self.timeout_secs))
            .send_json(body)
            .map_err(|e| {
                ForgeError::Mutation(format!(
                    "Impossible de joindre le LLM local (Ollama) : {e}"
                ))
            })?
            .into_json()
            .map_err(|e| {
                ForgeError::Mutation(format!(
                    "Reponse du LLM non-parsable (JSON invalide) : {e}"
                ))
            })?;

        let raw_output = response["response"]
            .as_str()
            .ok_or_else(|| {
                ForgeError::Mutation(
                    "Champ 'response' manquant dans la reponse LLM — le modele n'a pas produit de sortie exploitable".into(),
                )
            })?;

        extract_code_block(raw_output).ok_or_else(|| {
            ForgeError::Evaluation(
                "Le LLM n'a pas renvoye de bloc de code ```rust ... ```".into(),
            )
        })
    }
}

/// Extrait proprement le contenu d'un bloc markdown ```rust ... ```.
///
/// # Exemple
/// ```ignore
/// let input = "Voici du code :\n```rust\nfn main() {}\n```\nFin.";
/// assert_eq!(
///     extract_code_block(input),
///     Some("fn main() {}".to_string())
/// );
/// ```
pub fn extract_code_block(output: &str) -> Option<String> {
    let start_tag = "```rust";
    let end_tag = "```";

    let start_idx = output.find(start_tag)?;
    let code_start = start_idx + start_tag.len();

    let sub_str = &output[code_start..];
    let end_idx = sub_str.find(end_tag)?;

    Some(sub_str[..end_idx].trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_block_valid() {
        let input = "Some text\n```rust\nfn hello() {\n    println!(\"hi\");\n}\n```\nMore text";
        let extracted = extract_code_block(input).expect("should extract");
        assert!(extracted.contains("fn hello"));
        assert!(!extracted.contains("```"));
    }

    #[test]
    fn test_extract_code_block_no_block() {
        assert!(extract_code_block("no code block here").is_none());
    }

    #[test]
    fn test_extract_code_block_only_rust() {
        let input = "```rust\nlet x = 42;\n```";
        let extracted = extract_code_block(input).expect("should extract");
        assert_eq!(extracted, "let x = 42;");
    }

    #[test]
    fn test_extract_code_block_multiple_blocks() {
        // Should extract the first one
        let input = "```rust\nfn a() {}\n```\n```rust\nfn b() {}\n```";
        let extracted = extract_code_block(input).expect("should extract");
        assert!(extracted.contains("fn a()"));
    }
}
