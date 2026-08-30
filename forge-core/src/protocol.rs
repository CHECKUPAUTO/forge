//! Structures de données sérialisables transitant entre le Master et les Workers
//! d'évaluation. Le transport utilise `bincode` avec un framing explicite :
//! un entier u32 big-endian contenant la taille, suivi du payload sérialisé.
//!
//! Les adresses `tls://host:port` utilisent TLS avec validation de la chaîne et
//! du nom du certificat worker via `FORGE_TLS_CA_CERT`. Les adresses historiques
//! `host:port` restent TCP non authentifié et doivent être réservées à un réseau
//! de confiance ou à la boucle locale.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::candidate::{fnv1a, CandidateId};
use crate::error::{ForgeError, Result};
use crate::tls::{connect_tls, parse_tls_endpoint};

pub const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
pub const PROTOCOL_VERSION: u32 = 2;
pub const BENCHMARK_PROTOCOL: &str = "forge.verify-then-measure.v1";

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WorkerExecutionContext {
    pub worker_id: String,
    pub toolchain: String,
    pub os: String,
    pub arch: String,
    pub hardware: String,
    pub environment_fingerprint: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EvaluationPayload {
    pub candidate_id: CandidateId,
    pub source_code: String,
    pub seed: u64,
    pub generation: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EvaluationResult {
    pub protocol_version: u32,
    pub candidate_id: CandidateId,
    pub source_hash: u64,
    pub domain: String,
    pub benchmark_protocol: String,
    pub execution_context: WorkerExecutionContext,
    pub is_valid: bool,
    pub objectives: Vec<f64>,
    pub error_message: Option<String>,
}

fn write_frame<S: Write, T: Serialize>(stream: &mut S, value: &T) -> Result<()> {
    let bytes = bincode::serialize(value)
        .map_err(|e| ForgeError::Evaluation(format!("Échec sérialisation bincode: {e}")))?;
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err(ForgeError::Evaluation(format!(
            "Message réseau trop volumineux: {} octets > limite {}",
            bytes.len(), MAX_MESSAGE_BYTES
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

fn read_frame<S: Read, T: for<'de> Deserialize<'de>>(stream: &mut S) -> Result<T> {
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

fn validate_result(
    result: &EvaluationResult,
    payload: &EvaluationPayload,
    expected_domain: &str,
) -> Result<()> {
    if result.protocol_version != PROTOCOL_VERSION {
        return Err(ForgeError::Evaluation(format!(
            "Réponse worker de version incompatible: {} attendu={PROTOCOL_VERSION}",
            result.protocol_version
        )));
    }
    if result.candidate_id != payload.candidate_id {
        return Err(ForgeError::Evaluation(format!(
            "Réponse worker incohérente: candidate_id={} attendu={}",
            result.candidate_id, payload.candidate_id
        )));
    }
    let expected_source_hash = fnv1a(&payload.source_code);
    if result.source_hash != expected_source_hash {
        return Err(ForgeError::Evaluation(format!(
            "Réponse worker incohérente: source_hash={:016x} attendu={expected_source_hash:016x}",
            result.source_hash
        )));
    }
    if result.domain != expected_domain {
        return Err(ForgeError::Evaluation(format!(
            "Réponse worker incohérente: domain='{}' attendu='{expected_domain}'",
            result.domain
        )));
    }
    if result.benchmark_protocol != BENCHMARK_PROTOCOL {
        return Err(ForgeError::Evaluation(format!(
            "Réponse worker incohérente: benchmark_protocol='{}' attendu='{BENCHMARK_PROTOCOL}'",
            result.benchmark_protocol
        )));
    }
    let context = &result.execution_context;
    if context.worker_id.trim().is_empty()
        || context.toolchain.trim().is_empty()
        || context.os.trim().is_empty()
        || context.arch.trim().is_empty()
        || context.hardware.trim().is_empty()
        || context.environment_fingerprint.trim().is_empty()
    {
        return Err(ForgeError::Evaluation(
            "Réponse worker sans contexte reproductible complet".into(),
        ));
    }
    if result.is_valid
        && (result.objectives.is_empty() || !result.objectives.iter().all(|v| v.is_finite()))
    {
        return Err(ForgeError::Evaluation(
            "Réponse worker invalide: score déclaré valide mais objectifs absents/non finis".into(),
        ));
    }
    Ok(())
}

pub fn dispatch_evaluation_to_worker(
    addr: &str,
    payload: &EvaluationPayload,
    expected_domain: &str,
    timeout: Duration,
) -> Result<EvaluationResult> {
    if expected_domain.trim().is_empty() {
        return Err(ForgeError::Evaluation(
            "Domaine attendu vide pour l'évaluation distribuée".into(),
        ));
    }

    let result: EvaluationResult = if let Some(endpoint) = parse_tls_endpoint(addr)? {
        let mut stream = connect_tls(&endpoint, timeout)?;
        write_frame(&mut stream, payload)?;
        read_frame(&mut stream)?
    } else {
        let socket_addr: SocketAddr = addr.parse().map_err(|e| {
            ForgeError::Evaluation(format!("Adresse worker invalide '{addr}': {e}"))
        })?;
        let mut stream = TcpStream::connect_timeout(&socket_addr, timeout).map_err(|e| {
            ForgeError::Evaluation(format!("Connexion worker perdue ({addr}): {e}"))
        })?;
        stream.set_read_timeout(Some(timeout)).map_err(|e| {
            ForgeError::Evaluation(format!("Configuration timeout lecture: {e}"))
        })?;
        stream.set_write_timeout(Some(timeout)).map_err(|e| {
            ForgeError::Evaluation(format!("Configuration timeout écriture: {e}"))
        })?;
        write_frame(&mut stream, payload)?;
        read_frame(&mut stream)?
    };

    validate_result(&result, payload, expected_domain)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    fn context() -> WorkerExecutionContext {
        WorkerExecutionContext {
            worker_id: "test-worker".into(),
            toolchain: "rustc test".into(),
            os: "test-os".into(),
            arch: "test-arch".into(),
            hardware: "test-cpu".into(),
            environment_fingerprint: "env-123".into(),
        }
    }

    fn result_for(payload: &EvaluationPayload, domain: &str) -> EvaluationResult {
        EvaluationResult {
            protocol_version: PROTOCOL_VERSION,
            candidate_id: payload.candidate_id,
            source_hash: fnv1a(&payload.source_code),
            domain: domain.into(),
            benchmark_protocol: BENCHMARK_PROTOCOL.into(),
            execution_context: context(),
            is_valid: true,
            objectives: vec![42.0],
            error_message: None,
        }
    }

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
        let payload = EvaluationPayload {
            candidate_id: 12345,
            source_code: "source".into(),
            seed: 1,
            generation: 2,
        };
        let res = result_for(&payload, "test");
        let bytes = bincode::serialize(&res).expect("sérialisation");
        let recovered: EvaluationResult = bincode::deserialize(&bytes).expect("désérialisation");
        assert_eq!(recovered.execution_context, context());
        assert_eq!(recovered.objectives, vec![42.0]);
    }

    #[test]
    fn test_dispatch_invalid_addr() {
        let payload = EvaluationPayload {
            candidate_id: 1,
            source_code: "fn main() {}".into(),
            seed: 0,
            generation: 0,
        };
        assert!(dispatch_evaluation_to_worker(
            "invalid-addr",
            &payload,
            "test",
            Duration::from_secs(1)
        )
        .is_err());
    }

    #[test]
    fn framed_dispatch_roundtrip_without_eof() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let payload: EvaluationPayload = read_frame(&mut stream).expect("read request");
            write_frame(&mut stream, &result_for(&payload, "test")).expect("write response");
        });
        let payload = EvaluationPayload {
            candidate_id: 77,
            source_code: "x".repeat(128 * 1024),
            seed: 123,
            generation: 9,
        };
        let result = dispatch_evaluation_to_worker(
            &addr.to_string(),
            &payload,
            "test",
            Duration::from_secs(2),
        )
        .expect("dispatch");
        assert_eq!(result.objectives, vec![42.0]);
        worker.join().expect("worker thread");
    }

    #[test]
    fn dispatch_rejects_wrong_source_hash() {
        let payload = EvaluationPayload {
            candidate_id: 5,
            source_code: "actual-source".into(),
            seed: 1,
            generation: 1,
        };
        let mut result = result_for(&payload, "test");
        result.source_hash = 0;
        assert!(validate_result(&result, &payload, "test").is_err());
    }

    #[test]
    fn dispatch_rejects_wrong_domain() {
        let payload = EvaluationPayload {
            candidate_id: 6,
            source_code: "actual-source".into(),
            seed: 1,
            generation: 1,
        };
        let result = result_for(&payload, "wrong-domain");
        assert!(validate_result(&result, &payload, "expected-domain").is_err());
    }

    #[test]
    fn dispatch_rejects_incomplete_execution_context() {
        let payload = EvaluationPayload {
            candidate_id: 7,
            source_code: "actual-source".into(),
            seed: 1,
            generation: 1,
        };
        let mut result = result_for(&payload, "test");
        result.execution_context.worker_id.clear();
        assert!(validate_result(&result, &payload, "test").is_err());
    }
}
