//! Worker d'évaluation Forge — démon réseau asynchrone Tokio.
//!
//! Le worker reçoit des messages bincode encadrés par une longueur u32
//! big-endian, exécute la vérification puis la mesure du candidat et renvoie
//! une enveloppe de résultat versionnée contenant la provenance descriptive.
//!
//! Quand `FORGE_WORKER_TLS_CERT` et `FORGE_WORKER_TLS_KEY` sont définis, les
//! connexions sont protégées par TLS standard via rustls. Sinon le worker reste
//! en TCP non authentifié, destiné uniquement à la boucle locale ou à un réseau
//! de confiance explicitement protégé.

mod tls;

use std::net::SocketAddr;
use std::process::Command;
use std::sync::Arc;

use forge_core::domains::low_rank::{TensorCode, TensorTrainDomain};
use forge_core::domains::simd_kernel::{SimdKernelCode, SimdKernelDomain};
use forge_core::protocol::{
    EvaluationPayload, EvaluationResult, WorkerExecutionContext, BENCHMARK_PROTOCOL,
    MAX_MESSAGE_BYTES, PROTOCOL_VERSION, WORKER_DESCRIPTOR_VERSION,
};
use forge_core::{fnv1a, Domain, Trial};
use serde::{de::DeserializeOwned, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::tls::acceptor_from_env;

enum WorkerDomain {
    LowRank(TensorTrainDomain),
    SimdKernel(SimdKernelDomain),
}

impl WorkerDomain {
    fn name(&self) -> &str {
        match self {
            WorkerDomain::LowRank(domain) => domain.name(),
            WorkerDomain::SimdKernel(domain) => domain.name(),
        }
    }

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

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn detected_hardware() -> String {
    if let Ok(value) = std::env::var("FORGE_WORKER_HARDWARE") {
        if !value.trim().is_empty() {
            return value;
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if let Some((key, value)) = line.split_once(':') {
                    if matches!(key.trim(), "model name" | "Hardware" | "Processor")
                        && !value.trim().is_empty()
                    {
                        return value.trim().to_string();
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    if let Some(cpu) = command_output("sysctl", &["-n", "machdep.cpu.brand_string"]) {
        return cpu;
    }

    std::env::var("PROCESSOR_IDENTIFIER")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| format!("{}-hardware-unreported", std::env::consts::ARCH))
}

fn worker_execution_context(domain: &str) -> WorkerExecutionContext {
    let worker_id = std::env::var("FORGE_WORKER_ID")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .unwrap_or_else(|| "worker-unidentified".into());
    let toolchain = command_output("rustc", &["-Vv"]).unwrap_or_else(|| "rustc-unreported".into());
    let hardware = detected_hardware();
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    let explicit_env = std::env::var("FORGE_WORKER_ENV").unwrap_or_default();
    let material = format!(
        "forge-worker:{}|descriptor={WORKER_DESCRIPTOR_VERSION}|domain={domain}|protocol={PROTOCOL_VERSION}|benchmark={BENCHMARK_PROTOCOL}|os={os}|arch={arch}|hardware={hardware}|toolchain={toolchain}|env={explicit_env}",
        env!("CARGO_PKG_VERSION")
    );
    let environment_fingerprint = format!("fnv1a64:{:016x}", fnv1a(&material));

    WorkerExecutionContext {
        descriptor_version: WORKER_DESCRIPTOR_VERSION,
        worker_id,
        toolchain,
        os,
        arch,
        hardware,
        environment_fingerprint,
    }
}

async fn read_frame<S, T>(socket: &mut S) -> Result<T, Box<dyn std::error::Error>>
where
    S: AsyncRead + Unpin,
    T: DeserializeOwned,
{
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

async fn write_frame<S, T>(socket: &mut S, value: &T) -> Result<(), Box<dyn std::error::Error>>
where
    S: AsyncWrite + Unpin,
    T: Serialize,
{
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
    let context = worker_execution_context(domain.name());
    let tls_acceptor = acceptor_from_env()?;

    tracing::info!(
        worker_id = %context.worker_id,
        hardware = %context.hardware,
        environment_fingerprint = %context.environment_fingerprint,
        transport = if tls_acceptor.is_some() { "tls" } else { "tcp" },
        "[WORKER] contexte d'exécution"
    );

    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("Impossible de binder sur {addr}: {e}"))?;

    tracing::info!(
        "[WORKER] démon d'évaluation actif sur {} | domaine: {} | protocole: {} | transport: {}",
        addr,
        domain.name(),
        PROTOCOL_VERSION,
        if tls_acceptor.is_some() { "tls" } else { "tcp" }
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
                    Ok((socket, peer)) => {
                        let domain = Arc::clone(&domain);
                        let context = context.clone();
                        let tls_acceptor = tls_acceptor.clone();
                        tokio::spawn(async move {
                            let result = if let Some(acceptor) = tls_acceptor {
                                match acceptor.accept(socket).await {
                                    Ok(mut tls_stream) => handle_connection(domain, context, &mut tls_stream).await,
                                    Err(e) => Err(format!("Échec handshake TLS: {e}").into()),
                                }
                            } else {
                                let mut socket = socket;
                                handle_connection(domain, context, &mut socket).await
                            };
                            if let Err(e) = result {
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

async fn handle_connection<S>(
    domain: Arc<WorkerDomain>,
    context: WorkerExecutionContext,
    socket: &mut S,
) -> Result<(), Box<dyn std::error::Error>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let payload: EvaluationPayload = read_frame(socket).await?;

    tracing::info!(
        "[WORKER] évaluation candidat {} | génération {} | source_hash={:016x}",
        payload.candidate_id,
        payload.generation,
        fnv1a(&payload.source_code)
    );

    let trial = Trial {
        generation: payload.generation,
        seed: payload.seed,
    };
    let source_hash = fnv1a(&payload.source_code);
    let source_code = payload.source_code;
    let candidate_id = payload.candidate_id;
    let trial_seed = payload.seed;
    let generation = payload.generation;
    let response_domain = domain.name().to_string();

    let result = tokio::task::spawn_blocking(move || {
        let (is_valid, objectives, error_message) =
            domain.evaluate(&source_code, candidate_id, &trial);
        EvaluationResult {
            protocol_version: PROTOCOL_VERSION,
            candidate_id,
            source_hash,
            trial_seed,
            generation,
            domain: response_domain,
            benchmark_protocol: BENCHMARK_PROTOCOL.to_string(),
            execution_context: context,
            is_valid,
            objectives,
            error_message,
        }
    })
    .await
    .map_err(|e| format!("Panique dans le thread d'évaluation (candidat {candidate_id}): {e}"))?;

    write_frame(socket, &result).await?;

    tracing::info!(
        "[WORKER] candidat {} — valid={} | obj={:?} | env={}",
        result.candidate_id,
        result.is_valid,
        result.objectives,
        result.execution_context.environment_fingerprint
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[test]
    fn execution_context_is_non_empty_and_stable_within_process() {
        let first = worker_execution_context("test-domain");
        let second = worker_execution_context("test-domain");
        assert_eq!(first.descriptor_version, WORKER_DESCRIPTOR_VERSION);
        assert!(!first.worker_id.is_empty());
        assert!(!first.hardware.is_empty());
        assert!(!first.toolchain.is_empty());
        assert!(!first.os.is_empty());
        assert!(!first.arch.is_empty());
        assert_eq!(
            first.environment_fingerprint,
            second.environment_fingerprint
        );
    }

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
                    protocol_version: PROTOCOL_VERSION,
                    candidate_id: payload.candidate_id,
                    source_hash: fnv1a(&payload.source_code),
                    trial_seed: payload.seed,
                    generation: payload.generation,
                    domain: "test".into(),
                    benchmark_protocol: BENCHMARK_PROTOCOL.into(),
                    execution_context: WorkerExecutionContext {
                        descriptor_version: WORKER_DESCRIPTOR_VERSION,
                        worker_id: "test".into(),
                        toolchain: "rustc test".into(),
                        os: "test".into(),
                        arch: "test".into(),
                        hardware: "test-hardware".into(),
                        environment_fingerprint: "test-env".into(),
                    },
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
                "test",
                std::time::Duration::from_secs(2),
            )
        })
        .await
        .expect("join")
        .expect("dispatch");
        assert_eq!(client.candidate_id, 12);
        assert_eq!(client.trial_seed, 1);
        assert_eq!(client.generation, 2);
        server.await.expect("server");
    }
}
