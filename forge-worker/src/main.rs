//! Worker d'évaluation Forge — Démon réseau asynchrone Tokio de production.
//!
//! Ce binaire écoute sur un port TCP configurable, reçoit des
//! [`EvaluationPayload`] binaires (bincode) contenant le code source d'un
//! candidat, le compile dans un environnement isolé avec limites rlimit,
//! exécute les assertions de vérification, mesure les objectifs via
//! Criterion, et renvoie un [`EvaluationResult`] binaire au Master.
//!
//! ## Architecture
//! - **Tokio** : boucle d'acceptation asynchrone, une tâche par connexion.
//! - **spawn_blocking** : exécution synchrone des `Domain::verify` / `measure`
//!   pour ne pas bloquer l'exécuteur Tokio.
//! - **bincode** : sérialisation binaire pour des transactions TCP ultra-rapides.
//! - **Signal handling** : arrêt propre sur SIGINT / SIGTERM avec drainage
//!   des connexions en cours.
//!
//! ## Configuration par variables d'environnement
//! | Variable               | Défaut              | Description                              |
//! |------------------------|---------------------|------------------------------------------|
//! | `FORGE_WORKER_ADDR`    | `127.0.0.1:9000`   | Adresse d'écoute TCP                     |
//! | `FORGE_WORKER_DOMAIN`  | `low_rank`          | Domaine : `low_rank` ou `simd_kernel`    |
//! | `FORGE_WORKER_SCRATCH` | `./worker_scratch`  | Répertoire de travail temporaire         |
//!
//! ## Utilisation
//! ```bash
//! # Démarrage standard (domaine low_rank)
//! cargo run -p forge-worker
//!
//! # Domaine SIMD kernels sur un port personnalisé
//! FORGE_WORKER_DOMAIN=simd_kernel FORGE_WORKER_ADDR=0.0.0.0:9999 cargo run -p forge-worker
//! ```

use std::net::SocketAddr;
use std::sync::Arc;

use forge_core::domains::low_rank::{TensorCode, TensorTrainDomain};
use forge_core::domains::simd_kernel::{SimdKernelCode, SimdKernelDomain};
use forge_core::protocol::{EvaluationPayload, EvaluationResult};
use forge_core::{Domain, Trial};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

// ---------------------------------------------------------------------------
// Enum de dispatch pour les domaines supportés
// ---------------------------------------------------------------------------

/// Wrapper concret sur l'ensemble des domaines connus du worker.
/// Chaque variante sait instancier son type de candidat et appeler
/// `verify` / `measure` avec isolation complète.
enum WorkerDomain {
    LowRank(TensorTrainDomain),
    SimdKernel(SimdKernelDomain),
}

impl WorkerDomain {
    /// Évalue un candidat à partir du code source reçu et du contexte de trial.
    /// Retourne `(is_valid, objectives, error_message)`.
    ///
    /// Toutes les erreurs internes (compilation, exécution, timeout, crash)
    /// sont capturées et mappées dans le triplet de retour — aucun panic.
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

/// Exécute `verify` puis `measure` pour un couple (domaine, candidat) donné.
/// Propagation zéro-panique : toutes les erreurs sont capturées et retournées
/// sous forme de chaînes diagnostiques dans le champ `error_message`.
fn evaluate_candidate<D: Domain>(
    domain: &D,
    candidate: &D::Cand,
    trial: &Trial,
) -> (bool, Vec<f64>, Option<String>) {
    match domain.verify(candidate, trial) {
        Ok(true) => match domain.measure(candidate, trial) {
            Ok(objectives) => (true, objectives, None),
            Err(e) => (false, vec![], Some(format!("Échec mesure: {e}"))),
        },
        Ok(false) => (
            false,
            vec![],
            Some("Porte de vérification rejetée — échec compilation ou assertion mathématique".into()),
        ),
        Err(e) => (false, vec![], Some(format!("Erreur critique d'évaluation: {e}"))),
    }
}

// ---------------------------------------------------------------------------
// Point d'entrée — Démon de production
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialisation du système de log structuré
    tracing_subscriber::fmt::init();

    // ── Configuration par variables d'environnement ──
    let addr: SocketAddr = std::env::var("FORGE_WORKER_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:9000".to_string())
        .parse()
        .map_err(|e| format!("Adresse worker invalide (FORGE_WORKER_ADDR): {e}"))?;

    let domain_kind = std::env::var("FORGE_WORKER_DOMAIN")
        .unwrap_or_else(|_| "low_rank".to_string());

    let domain = init_domain(&domain_kind)
        .map_err(|e| format!("Initialisation domaine '{domain_kind}' échouée: {e}"))?;

    // ── Démarrage du listener TCP ──
    let listener = TcpListener::bind(&addr).await.map_err(|e| {
        format!("Impossible de binder sur {addr}: {e}")
    })?;

    tracing::info!(
        "[WORKER] 🔧 Démon d'évaluation actif sur {} | domaine: {}",
        addr,
        domain_kind
    );
    tracing::info!(
        "[WORKER] 📡 En attente de connexions Master..."
    );

    // ── Gestion des signaux d'arrêt système ──
    // On utilise un canal unbounded pour notifier la boucle principale.
    // Le handler de signal tourne dans une tâche séparée et envoie
    // un message unique quand un signal d'arrêt est reçu.
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::unbounded_channel::<()>();

    tokio::spawn(async move {
        #[cfg(unix)]
        {
            let mut sigint = tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::interrupt(),
            )
            .expect("signal SIGINT");
            let mut sigterm = tokio::signal::unix::signal(
                tokio::signal::unix::SignalKind::terminate(),
            )
            .expect("signal SIGTERM");

            tokio::select! {
                _ = sigint.recv() => {
                    tracing::info!("[WORKER] ⏹️  Signal SIGINT reçu");
                }
                _ = sigterm.recv() => {
                    tracing::info!("[WORKER] ⏹️  Signal SIGTERM reçu");
                }
            }
        }

        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("[WORKER] ⏹️  Signal Ctrl+C reçu");
        }

        // Notifier la boucle principale
        let _ = shutdown_tx.send(());
    });

    // ── Boucle principale avec écoute et arrêt propre ──
    loop {
        tokio::select! {
            conn = listener.accept() => {
                match conn {
                    Ok((mut socket, peer)) => {
                        tracing::debug!("[WORKER] 🔗 Connexion de {peer}");
                        let domain = domain.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(domain, &mut socket).await {
                                tracing::warn!("[WORKER] ❌ Erreur traitement {peer}: {e}");
                            }
                            tracing::debug!("[WORKER] 🔌 Session {peer} terminée");
                        });
                    }
                    Err(e) => {
                        tracing::error!("[WORKER] Erreur d'acceptation: {e}");
                        // Continue la boucle même en cas d'erreur d'acceptation
                    }
                }
            }
            _ = shutdown_rx.recv() => {
                tracing::info!("[WORKER] 🛑 Arrêt du démon demandé — drainage des connexions en cours...");
                // Le contexte tokio va se fermer proprement, libérant les ressources
                break;
            }
        }
    }

    tracing::info!("[WORKER] ✅ Démon arrêté proprement.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Initialisation du domaine
// ---------------------------------------------------------------------------

/// Instancie le domaine configuré via la variable d'environnement
/// `FORGE_WORKER_DOMAIN`. Le domaine est enveloppé dans un `Arc` pour
/// être partagé entre toutes les tâches asynchrones.
fn init_domain(kind: &str) -> Result<Arc<WorkerDomain>, Box<dyn std::error::Error>> {
    let scratch = std::env::var("FORGE_WORKER_SCRATCH")
        .unwrap_or_else(|_| "./worker_scratch".to_string());

    // Création du répertoire de travail s'il n'existe pas
    std::fs::create_dir_all(&scratch).map_err(|e| {
        format!("Impossible de créer le répertoire de travail '{scratch}': {e}")
    })?;

    match kind {
        "low_rank" => {
            let domain = TensorTrainDomain::new(&scratch);
            tracing::info!("[WORKER] Domaine chargé: Tensor Train (low_rank) — scratch: {scratch}");
            Ok(Arc::new(WorkerDomain::LowRank(domain)))
        }
        "simd_kernel" => {
            let domain = SimdKernelDomain::new(&scratch);
            tracing::info!("[WORKER] Domaine chargé: SIMD GEMM kernels — scratch: {scratch}");
            Ok(Arc::new(WorkerDomain::SimdKernel(domain)))
        }
        other => Err(format!(
            "Domaine inconnu: '{other}'. Domaines disponibles: low_rank, simd_kernel"
        )
        .into()),
    }
}

// ---------------------------------------------------------------------------
// Traitement d'une connexion
// ---------------------------------------------------------------------------

/// Traite une connexion TCP entrante :
/// 1. Lit le flux TCP et désérialise l'`EvaluationPayload` en bincode natif.
/// 2. Exécute l'évaluation lourde dans `spawn_blocking` (compilation Cargo,
///    exécution rlimit, bench Criterion) pour ne pas bloquer l'exécuteur Tokio.
/// 3. Capture tous les retours d'erreur (timeout sous-processus, échec
///    compilation, crash) et les mappe dans le champ `error_message`.
/// 4. Sérialise et renvoie l'`EvaluationResult` sur le flux TCP.
async fn handle_connection(
    domain: Arc<WorkerDomain>,
    socket: &mut TcpStream,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Lecture du payload binaire
    //    Stratégie : on lit tout ce que le socket nous donne.
    //    Le Master envoie le payload puis shutdown(SHUT_WR), ce qui
    //    déclenche EOF côté Worker et termine la lecture.
    let mut buffer = vec![0u8; 65536];
    let mut total = 0usize;

    loop {
        let n = socket.read(&mut buffer[total..]).await?;
        if n == 0 {
            break; // EOF — le Master a terminé d'envoyer
        }
        total += n;
        if total >= buffer.len() {
            // Buffer saturé — on désérialise avec ce qu'on a
            break;
        }
    }

    if total == 0 {
        return Err("Connexion fermée sans données (EOF avant lecture du payload)".into());
    }

    let payload: EvaluationPayload =
        bincode::deserialize(&buffer[..total]).map_err(|e| {
            format!("Désérialisation bincode échouée (données corrompues?): {e}")
        })?;

    tracing::info!(
        "[WORKER] 📦 Évaluation candidat {} | génération {}",
        payload.candidate_id,
        payload.generation
    );

    // 2. Reconstruction du contexte Trial (graine déterministe)
    let trial = Trial {
        generation: payload.generation,
        seed: payload.seed,
    };

    // 3. Exécution lourde dans spawn_blocking — le domaine est synchrone
    //    (compilation cargo, exécution, bench Criterion).
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
    .map_err(|e| {
        format!(
            "Panique dans le thread d'évaluation bloquant (candidat {candidate_id}): {e}"
        )
    })?;

    // 4. Sérialisation binaire et envoi de la réponse
    let response_bytes = bincode::serialize(&result).map_err(|e| {
        format!("Sérialisation bincode de la réponse échouée: {e}")
    })?;

    socket.write_all(&response_bytes).await.map_err(|e| {
        format!("Échec d'écriture de la réponse sur le socket: {e}")
    })?;

    socket.flush().await.map_err(|e| {
        format!("Échec du flush du socket: {e}")
    })?;

    tracing::info!(
        "[WORKER] ✅ Candidat {} — valid={} | obj={:?}",
        result.candidate_id,
        result.is_valid,
        result.objectives
    );

    Ok(())
}
