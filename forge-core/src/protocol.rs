//! Structures de données sérialisables transitant entre le Master et les Workers
//! d'évaluation. Le transport utilise `bincode` avec un framing explicite :
//! un entier u32 big-endian contenant la taille, suivi du payload sérialisé.
//!
//! ## Protocole TCP
//! 1. Le Master ouvre une connexion TCP synchrone vers le Worker.
//! 2. Il envoie `len || EvaluationPayload`.
//! 3. Le Worker renvoie `len || EvaluationResult`.
//!
//! Le framing évite de dépendre d'un EOF pour délimiter un message et permet
//! de réutiliser la même connexion en requête-réponse sans deadlock.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::candidate::CandidateId;
use crate::error::{ForgeError, Result};

/// Taille maximale d'un message Forge sur le réseau (code candidat inclus).
pub const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Structures de données du protocole
// ---------------------------------------------------------------------------

/// Paquet envoyé par le Master à un Worker pour demander l'évaluation
/// d'un candidat.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EvaluationPayload {
    /// Identifiant unique du candidat (hash FNV-1a).
    pub candidate_id: CandidateId,
    /// Code source du candidat à compiler et exécuter.
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

fn write_frame<T: Serialize>(stream: &mut TcpStream, value: &T) -> Result<()> {
    let bytes = bincode::serialize(value)
        .map_err(|e| ForgeError::Evaluation(format!("Échec sérialisation bincode: {e}")))?;
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err(ForgeError::Evaluation(format!(
            "Message réseau trop volumineux: {} octets > limite {}",
            bytes.len(),
            MAX_MESSAGE_BYTES
        )));
    }
    let len = u32::try_from(bytes.len()).map_err(|_| {
        ForgeError::Evaluation("Message réseau trop volumineux pour le framing u32".into())
    })?;
    stream
        .write_all(&len.to_be_bytes())
        .map_err(|e| ForgeError::Evaluation(format!("Échec écriture taille du frame: {e}")))?;
    stream
        .write_all(&bytes)
        .map_err(|e| ForgeError::Evaluation(format!("Échec écriture payload: {e}")))?;
    stream
        .flush()
        .map_err(|e| ForgeError::Evaluation(format!("Échec flush socket: {e}")))?;
    Ok(())
}

fn read_frame<T: for<'de> Deserialize<'de>>(stream: &mut TcpStream) -> Result<T> {
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .map_err(|e| ForgeError::Evaluation(format!("Échec lecture taille du frame: {e}")))?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len == 0 || len > MAX_MESSAGE_BYTES {
        return Err(ForgeError::Evaluation(format!(
            "Taille de frame invalide: {len} octets (limite {MAX_MESSAGE_BYTES})"
        )));
    }
    let mut bytes = vec![0u8; len];
    stream
        .read_exact(&mut bytes)
        .map_err(|e| ForgeError::Evaluation(format!("Échec lecture payload: {e}")))?;
    bincode::deserialize(&bytes)
        .map_err(|e| ForgeError::Evaluation(format!("Payload corrompu du worker: {e}")))
}

// ---------------------------------------------------------------------------
// Fonction de routage maître (dispatch synchrone)
// ---------------------------------------------------------------------------

/// Envoie un [`EvaluationPayload`] à un Worker distant et récupère le
/// [`EvaluationResult`]. Conçu pour être appelé depuis un thread Rayon.
pub fn dispatch_evaluation_to_worker(
    addr: &str,
    payload: &EvaluationPayload,
    timeout: Duration,
) -> Result<EvaluationResult> {
    let socket_addr: SocketAddr = addr
        .parse()
        .map_err(|e| ForgeError::Evaluation(format!("Adresse worker invalide '{addr}': {e}")))?;

    let mut stream = TcpStream::connect_timeout(&socket_addr, timeout)
        .map_err(|e| ForgeError::Evaluation(format!("Connexion worker perdue ({addr}): {e}")))?;

    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| ForgeError::Evaluation(format!("Configuration timeout lecture: {e}")))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| ForgeError::Evaluation(format!("Configuration timeout écriture: {e}")))?;

    write_frame(&mut stream, payload)?;
    let result: EvaluationResult = read_frame(&mut stream)?;

    if result.candidate_id != payload.candidate_id {
        return Err(ForgeError::Evaluation(format!(
            "Réponse worker incohérente: candidate_id={} attendu={}",
            result.candidate_id, payload.candidate_id
        )));
    }

    if result.is_valid
        && (result.objectives.is_empty() || !result.objectives.iter().all(|v| v.is_finite()))
    {
        return Err(ForgeError::Evaluation(
            "Réponse worker invalide: score déclaré valide mais objectifs absents/non finis".into(),
        ));
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn test_payload_bincode_roundtrip() {
        let payload = EvaluationPayload {
            candidate_id: 0xABCD_1234,
            source_code: "fn main() {}".into(),
            seed: 42,
            generation: 7,
        };

        let bytes = bincode::serialize(&payload).expect("sérialisation");
        let recovered: EvaluationPayload = bincode::deserialize(&bytes).expect("désérialisation");

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
        let recovered: EvaluationResult = bincode::deserialize(&bytes).expect("désérialisation");

        assert_eq!(recovered.candidate_id, res.candidate_id);
        assert!(recovered.is_valid);
        assert_eq!(recovered.objectives, vec![1.5, 2.7, 3.9]);
        assert!(recovered.error_message.is_none());
    }

    #[test]
    fn test_dispatch_invalid_addr() {
        let payload = EvaluationPayload {
            candidate_id: 1,
            source_code: "fn main() {}".into(),
            seed: 0,
            generation: 0,
        };
        let result =
            dispatch_evaluation_to_worker("invalid-addr", &payload, Duration::from_secs(1));
        assert!(result.is_err());
    }

    #[test]
    fn framed_dispatch_roundtrip_without_eof() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let payload: EvaluationPayload = read_frame(&mut stream).expect("read request");
            let result = EvaluationResult {
                candidate_id: payload.candidate_id,
                is_valid: true,
                objectives: vec![42.0],
                error_message: None,
            };
            write_frame(&mut stream, &result).expect("write response");
        });

        let payload = EvaluationPayload {
            candidate_id: 77,
            source_code: "x".repeat(128 * 1024),
            seed: 123,
            generation: 9,
        };
        let result =
            dispatch_evaluation_to_worker(&addr.to_string(), &payload, Duration::from_secs(2))
                .expect("dispatch");
        assert_eq!(result.candidate_id, 77);
        assert_eq!(result.objectives, vec![42.0]);
        worker.join().expect("worker thread");
    }
}
