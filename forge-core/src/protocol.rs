//! Structures de données sérialisables transitant entre le Master et les Workers
//! d'évaluation. Utilise `bincode` pour une sérialisation binaire ultra-rapide
//! sans parsing textuel.
//!
//! ## Protocole TCP
//! 1. Le Master ouvre une connexion TCP synchrone vers le Worker.
//! 2. Il envoie un [`EvaluationPayload`] sérialisé en bincode.
//! 3. Il lit la réponse [`EvaluationResult`] désérialisée en bincode.
//!
//! Aucun framing complexe : une connexion = une requête-réponse.

use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::candidate::CandidateId;
use crate::error::{ForgeError, Result};

// ---------------------------------------------------------------------------
// Structures de données du protocole
// ---------------------------------------------------------------------------

/// Paquet envoyé par le Master à un Worker pour demander l'évaluation
/// d'un candidat.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EvaluationPayload {
    /// Identifiant unique du candidat (hash FNV-1a).
    pub candidate_id: CandidateId,
    /// Code source Rust du candidat à compiler et exécuter.
    pub source_code: String,
    /// Graine du trial pour reproductibilité.
    pub seed: u64,
    /// Génération courante.
    pub generation: u64,
}

/// Réponse renvoyée par le Worker après évaluation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EvaluationResult {
    /// Identifiant du candidat évalué.
    pub candidate_id: CandidateId,
    /// Le candidat a-t-il passé la porte de vérification ?
    pub is_valid: bool,
    /// Objectifs mesurés (vide si invalide).
    pub objectives: Vec<f64>,
    /// Message d'erreur en cas d'échec de compilation ou de crash.
    pub error_message: Option<String>,
}

// ---------------------------------------------------------------------------
// Fonction de routage maître (dispatch synchrone)
// ---------------------------------------------------------------------------

/// Envoie un [`EvaluationPayload`] à un Worker distant et récupère le
/// [`EvaluationResult`]. Conçu pour être appelé depuis un thread Rayon
/// (dispatch synchrone avec timeout agressif).
///
/// # Arguments
/// * `addr` — adresse du Worker au format `"host:port"` (ex: `"192.168.1.10:9000"`).
/// * `payload` — le paquet d'évaluation à transmettre.
/// * `timeout` — timeout de connexion ET de lecture/écriture.
///
/// # Protocole binaire
/// 1. Sérialise `payload` avec `bincode::serialize_into` sur le flux TCP.
/// 2. Flushe pour garantir l'envoi complet.
/// 3. Désérialise la réponse avec `bincode::deserialize_from`.
pub fn dispatch_evaluation_to_worker(
    addr: &str,
    payload: &EvaluationPayload,
    timeout: Duration,
) -> Result<EvaluationResult> {
    let socket_addr: SocketAddr = addr
        .parse()
        .map_err(|e| ForgeError::Evaluation(format!("Adresse worker invalide '{addr}': {e}")))?;

    let mut stream = TcpStream::connect_timeout(&socket_addr, timeout).map_err(|e| {
        ForgeError::Evaluation(format!("Connexion worker perdue ({addr}): {e}"))
    })?;

    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| ForgeError::Evaluation(format!("Configuration timeout lecture: {e}")))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| ForgeError::Evaluation(format!("Configuration timeout écriture: {e}")))?;

    // Sérialisation binaire du payload dans le flux TCP
    bincode::serialize_into(&mut stream, payload)
        .map_err(|e| ForgeError::Evaluation(format!("Échec sérialisation payload: {e}")))?;

    // Flush impératif pour garantir que le Worker reçoit le message complet
    stream
        .flush()
        .map_err(|e| ForgeError::Evaluation(format!("Échec flush socket: {e}")))?;

    // Désérialisation binaire de la réponse
    let result: EvaluationResult =
        bincode::deserialize_from(&mut stream).map_err(|e| {
            ForgeError::Evaluation(format!("Payload corrompu du worker: {e}"))
        })?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Vérifie la sérialisation/désérialisation aller-retour en mémoire.
    #[test]
    fn test_payload_bincode_roundtrip() {
        let payload = EvaluationPayload {
            candidate_id: 0xABCD_1234,
            source_code: "fn main() {}".into(),
            seed: 42,
            generation: 7,
        };

        let bytes = bincode::serialize(&payload).expect("sérialisation");
        let recovered: EvaluationPayload =
            bincode::deserialize(&bytes).expect("désérialisation");

        assert_eq!(recovered.candidate_id, payload.candidate_id);
        assert_eq!(recovered.source_code, payload.source_code);
        assert_eq!(recovered.seed, payload.seed);
        assert_eq!(recovered.generation, payload.generation);
    }

    #[test]
    fn test_result_bincode_roundtrip() {
        let res = EvaluationResult {
            candidate_id: 12345,
            is_valid: true,
            objectives: vec![1.5, 2.7, 3.9],
            error_message: None,
        };

        let bytes = bincode::serialize(&res).expect("sérialisation");
        let recovered: EvaluationResult =
            bincode::deserialize(&bytes).expect("désérialisation");

        assert_eq!(recovered.candidate_id, res.candidate_id);
        assert!(recovered.is_valid);
        assert_eq!(recovered.objectives, vec![1.5, 2.7, 3.9]);
        assert!(recovered.error_message.is_none());
    }

    #[test]
    fn test_result_with_error_bincode_roundtrip() {
        let res = EvaluationResult {
            candidate_id: 999,
            is_valid: false,
            objectives: vec![],
            error_message: Some("Compilation échouée: syntax error".into()),
        };

        let bytes = bincode::serialize(&res).expect("sérialisation");
        let recovered: EvaluationResult =
            bincode::deserialize(&bytes).expect("désérialisation");

        assert!(!recovered.is_valid);
        assert_eq!(
            recovered.error_message.unwrap(),
            "Compilation échouée: syntax error"
        );
    }

    #[test]
    fn test_dispatch_invalid_addr() {
        let payload = EvaluationPayload {
            candidate_id: 1,
            source_code: "fn main() {}".into(),
            seed: 0,
            generation: 0,
        };
        let result = dispatch_evaluation_to_worker(
            "invalid-addr",
            &payload,
            Duration::from_secs(1),
        );
        assert!(result.is_err());
    }
}
