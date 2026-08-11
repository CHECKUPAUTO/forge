//! Worker d'évaluation Forge — démon réseau asynchrone Tokio.
//!
//! Le worker reçoit des messages bincode encadrés par une longueur u32
//! big-endian, exécute la vérification puis la mesure du candidat et renvoie
//! un `EvaluationResult` selon le même framing. Le framing partagé avec le
//! Master évite toute dépendance à EOF et accepte des candidats > 64 KiB.

use std::net::SocketAddr;
use std::sync::Arc;

use forge_core::domains::low_rank::{TensorCode, TensorTrainDomain};
use forge_core::domains::simd_kernel::{SimdKernelCode, SimdKernelDomain};
use forge_core::protocol::{EvaluationPayload, EvaluationResult, MAX_MESSAGE_BYTES};
use forge_core::{Domain, Trial};
use serde::{de::DeserializeOwned, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

// ---------------------------------------------------------------------------
// Enum de dispatch pour les domaines supportés
// ---------------------------------------------------------------------------

enum WorkerDomain {
    LowRank(TensorTrainDomain),
    SimdKernel(SimdKernelDomain),
}

impl WorkerDomain {
    fn evaluate(
        &self,
        source_code: &str,
        candidate_id: u64,
        trial: &Trial,
    ) -> (bool, Vec<f64>, Option<String>) {
        match self {
            WorkerDomain::LowRank(domain) => {
                let candidate = TensorCode {
                    raw_source: source_code.to_string(),
                    id: candidate_id,
                };
                evaluate_candidate(domain, &candidate, trial)
            }
            WorkerDomain::SimdKernel(domain) => {
                let candidate = SimdKernelCode {
                    source: source_code.to_string(),
                    id: candidate_id,
                };
                evaluate_candidate(domain, &candidate, trial)
            }
        }
    }
}

fn evaluate_candidate<D: Domain>(
    domain: &D,
    candidate: &D::Cand,
    trial: &Trial,
) -> (bool, Vec<f64>, Option<String>) {
    match domain.verify(candidate, trial) {
        Ok(true) => match domain.measure(candidate, trial) {
            Ok(objectives)
                if !objectives.is_empty() && objectives.iter().all(|v| v.is_finite()) =>
            {
                (true, objectives, None)
            }
            Ok(_) => (
                false,
                vec![],
                Some("Mesure invalide: objectifs absents ou non finis".into()),
            ),
            Err(e) => (false, vec![], Some(format!("Échec mesure: {e}"))),
        },
        Ok(false) => (
            false,
            vec![],
            Some(
                "Porte de vérification rejetée — échec compilation ou assertion mathématique"
                    .into(),
            ),
        ),
        Err(e) => (
            false,
            vec![],
            Some(format!("Erreur critique d'évaluation: {e}")),
        ),
    }
}

async fn read_frame<T: DeserializeOwned>(
    socket: &mut TcpStream,
) -> Result<T, Box<dyn std::error::Error>> {
    let mut len_buf = [0u8; 4];
    socket.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len == 0 || len > MAX_MESSAGE_BYTES {
        return Err(
            format!("Taille de frame invalide: {len} octets (limite {MAX_MESSAGE_BYTES})").into(),
        );
    }

    let mut payload = vec![0u8; len];
    socket.read_exact(&mut payload).await?;
    Ok(bincode::deserialize(&payload)?)
}

async fn write_frame<T: Serialize>(
    socket: &mut TcpStream,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    let payload = bincode::serialize(value)?;
    if payload.len() > MAX_MESSAGE_BYTES {
        return Err(format!(
            "Réponse trop volumineuse: {} octets > limite {}",
            payload.len(),
            MAX_MESSAGE_BYTES
        )
        .into());
    }
    let len = u32::try_from(payload.len())?;
    socket.write_all(&len.to_be_bytes()).await?;
    socket.write_all(&payload).await?;
    socket.flush().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Point d'entrée
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let addr: SocketAddr = std::env::var("FORGE_WORKER_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:9000".to_string())
        .parse()
        .map_err(|e| format!("Adresse worker invalide (FORGE_WORKER_ADDR): {e}"))?;

    let domain_kind =
        std::env::var("FORGE_WORKER_DOMAIN").unwrap_or_else(|_| "low_rank".to_string());
    let domain = init_domain(&domain_kind)
        .map_err(|e| format!("Initialisation domaine '{domain_kind}' échouée: {e}"))?;

    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("Impossible de binder sur {addr}: {e}"))?;

    tracing::info!(
        "[WORKER] démon d'évaluation actif sur {} | domaine: {}",
        addr,
        domain_kind
    );

    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            let mut sigint =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                    .expect("signal SIGINT");
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("signal SIGTERM");
            tokio::select! {
                _ = sigint.recv() => tracing::info!("[WORKER] SIGINT reçu"),
                _ = sigterm.recv() => tracing::info!("[WORKER] SIGTERM reçu"),
            }
        }

        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("[WORKER] Ctrl+C reçu");
        }

        let _ = shutdown_tx.send(());
    });

    loop {
        tokio::select! {
            conn = listener.accept() => {
                match conn {
                    Ok((mut socket, peer)) => {
                        let domain = Arc::clone(&domain);
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(domain, &mut socket).await {
                                tracing::warn!("[WORKER] erreur traitement {peer}: {e}");
                            }
                        });
                    }
                    Err(e) => tracing::error!("[WORKER] erreur acceptation: {e}"),
                }
            }
            _ = shutdown_rx.recv() => {
                tracing::info!("[WORKER] arrêt demandé");
                break;
            }
        }
    }

    Ok(())
}

fn init_domain(kind: &str) -> Result<Arc<WorkerDomain>, Box<dyn std::error::Error>> {
    let scratch =
        std::env::var("FORGE_WORKER_SCRATCH").unwrap_or_else(|_| "./worker_scratch".to_string());
    std::fs::create_dir_all(&scratch)?;

    match kind {
        "low_rank" => Ok(Arc::new(WorkerDomain::LowRank(TensorTrainDomain::new(
            &scratch,
        )))),
        "simd_kernel" => Ok(Arc::new(WorkerDomain::SimdKernel(SimdKernelDomain::new(
            &scratch,
        )))),
        other => Err(format!(
            "Domaine inconnu: '{other}'. Domaines disponibles: low_rank, simd_kernel"
        )
        .into()),
    }
}

async fn handle_connection(
    domain: Arc<WorkerDomain>,
    socket: &mut TcpStream,
) -> Result<(), Box<dyn std::error::Error>> {
    let payload: EvaluationPayload = read_frame(socket).await?;

    tracing::info!(
        "[WORKER] évaluation candidat {} | génération {}",
        payload.candidate_id,
        payload.generation
    );

    let trial = Trial {
        generation: payload.generation,
        seed: payload.seed,
    };
    let source_code = payload.source_code;
    let candidate_id = payload.candidate_id;

    let result = tokio::task::spawn_blocking(move || {
        let (is_valid, objectives, error_message) =
            domain.evaluate(&source_code, candidate_id, &trial);
        EvaluationResult {
            candidate_id,
            is_valid,
            objectives,
            error_message,
        }
    })
    .await
    .map_err(|e| format!("Panique dans le thread d'évaluation (candidat {candidate_id}): {e}"))?;

    write_frame(socket, &result).await?;

    tracing::info!(
        "[WORKER] candidat {} — valid={} | obj={:?}",
        result.candidate_id,
        result.is_valid,
        result.objectives
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn frame_roundtrip_accepts_large_payload() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let payload: EvaluationPayload = read_frame(&mut socket).await.expect("read");
            assert!(payload.source_code.len() > 65_536);
            write_frame(
                &mut socket,
                &EvaluationResult {
                    candidate_id: payload.candidate_id,
                    is_valid: true,
                    objectives: vec![1.0],
                    error_message: None,
                },
            )
            .await
            .expect("write");
        });

        let client = tokio::task::spawn_blocking(move || {
            forge_core::protocol::dispatch_evaluation_to_worker(
                &addr.to_string(),
                &EvaluationPayload {
                    candidate_id: 12,
                    source_code: "x".repeat(128 * 1024),
                    seed: 1,
                    generation: 2,
                },
                std::time::Duration::from_secs(2),
            )
        })
        .await
        .expect("join")
        .expect("dispatch");

        assert_eq!(client.candidate_id, 12);
        server.await.expect("server");
    }
}
