//! Stress-test d'intégration distribué pour forge-core.
//!
//! Ce test :
//! 1. Démarre un Worker mock (TCP listener) dans un thread d'arrière-plan
//!    qui évalue les candidats selon leur type (valide / syntax_error / loop).
//! 2. Configure le Master avec `evaluate_parallel_distributed` pointant
//!    vers ce worker local.
//! 3. Envoie une rafale de 50 candidats via Rayon pour saturer la boucle
//!    de connexion.
//! 4. Valide que 100% des réponses sont collectées sans corruption.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

use forge_core::evaluate_parallel_distributed;
use forge_core::protocol::{
    EvaluationPayload, EvaluationResult, WorkerExecutionContext, BENCHMARK_PROTOCOL,
    PROTOCOL_VERSION,
};
use forge_core::{fnv1a, Candidate, CandidateId};
use forge_core::{Individual, Trial};

const STUB_DOMAIN: &str = "stub-domain";

fn read_frame<T: serde::de::DeserializeOwned>(stream: &mut std::net::TcpStream) -> T {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).expect("read frame length");
    let len = u32::from_be_bytes(len) as usize;
    assert!(len > 0 && len <= forge_core::protocol::MAX_MESSAGE_BYTES);
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).expect("read frame payload");
    bincode::deserialize(&payload).expect("decode frame")
}

fn write_frame<T: serde::Serialize>(stream: &mut std::net::TcpStream, value: &T) {
    let payload = bincode::serialize(value).expect("encode frame");
    let len = u32::try_from(payload.len()).expect("frame length");
    stream
        .write_all(&len.to_be_bytes())
        .expect("write frame length");
    stream.write_all(&payload).expect("write frame payload");
    stream.flush().expect("flush frame");
}

// ---------------------------------------------------------------------------
// Candidat de stub pour le test
// ---------------------------------------------------------------------------

/// Candidat minimal implémentant `Candidate` pour les besoins du stress-test.
#[derive(Clone, Debug)]
struct StubCandidate {
    id: u64,
    source: String,
}

impl Candidate for StubCandidate {
    fn id(&self) -> CandidateId {
        self.id
    }
    fn repr(&self) -> String {
        self.source.clone()
    }
}

// ---------------------------------------------------------------------------
// Logique d'évaluation du Worker mock
// ---------------------------------------------------------------------------

fn worker_context() -> WorkerExecutionContext {
    WorkerExecutionContext {
        worker_id: "stub-worker".into(),
        toolchain: "rustc-test".into(),
        os: "test-os".into(),
        arch: "test-arch".into(),
        hardware: "test-hardware".into(),
        environment_fingerprint: "test-environment".into(),
    }
}

fn result_for(
    payload: &EvaluationPayload,
    is_valid: bool,
    objectives: Vec<f64>,
    error_message: Option<String>,
) -> EvaluationResult {
    EvaluationResult {
        protocol_version: PROTOCOL_VERSION,
        candidate_id: payload.candidate_id,
        source_hash: fnv1a(&payload.source_code),
        domain: STUB_DOMAIN.into(),
        benchmark_protocol: BENCHMARK_PROTOCOL.into(),
        execution_context: worker_context(),
        is_valid,
        objectives,
        error_message,
    }
}

/// Évalue un candidat reçu dans le Worker mock.
///
/// - `valid_*` → valide avec objectifs simulés
/// - `syntax_error_*` → invalide (erreur de compilation)
/// - `loop_*` → invalide (timeout / boucle infinie simulée)
fn evaluate_stub(payload: &EvaluationPayload) -> EvaluationResult {
    if payload.source_code.contains("syntax_error") {
        result_for(
            payload,
            false,
            vec![],
            Some(format!(
                "Erreur de compilation simulée: unexpected token in '{}'",
                payload.source_code
            )),
        )
    } else if payload.source_code.contains("loop_infinite") {
        result_for(
            payload,
            false,
            vec![],
            Some("Timeout dépassé : boucle infinie détectée, processus tué.".into()),
        )
    } else {
        // Candidat valide — objectifs simulés
        let base_latency = 1000.0 + (payload.candidate_id as f64 % 100.0) * 10.0;
        result_for(payload, true, vec![0.001, base_latency, 50.0], None)
    }
}

// ---------------------------------------------------------------------------
// Test principal
// ---------------------------------------------------------------------------

#[test]
fn test_distributed_evolution_under_stress() {
    // ── 1. Démarrage du Worker mock ──
    let barrier = Arc::new(Barrier::new(2));
    let shutdown = Arc::new(AtomicBool::new(false));
    let worker_errors = Arc::new(Mutex::new(Vec::<String>::new()));

    let b = barrier.clone();
    let s = shutdown.clone();
    let e = worker_errors.clone();

    let worker_handle = thread::spawn(move || {
        let listener = match TcpListener::bind("127.0.0.1:19999") {
            Ok(l) => l,
            Err(err) => {
                e.lock().unwrap().push(format!("bind: {err}"));
                return;
            }
        };

        // Non-bloquant pour permettre l'arrêt propre
        listener.set_nonblocking(true).expect("set_nonblocking");

        // Signale que le worker est prêt
        b.wait();

        loop {
            if s.load(Ordering::Relaxed) {
                break;
            }

            match listener.accept() {
                Ok((mut stream, _peer)) => {
                    // Une tâche par connexion pour la concurrence
                    thread::spawn(move || {
                        stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
                        stream.set_write_timeout(Some(Duration::from_secs(10))).ok();

                        let payload: EvaluationPayload = read_frame(&mut stream);

                        let result = evaluate_stub(&payload);

                        write_frame(&mut stream, &result);
                    });
                }
                Err(ref err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    // Pas de connexion entrante — courte pause
                    thread::sleep(Duration::from_millis(5));
                }
                Err(err) => {
                    e.lock().unwrap().push(format!("accept error: {err}"));
                    break;
                }
            }
        }
    });

    // ── 2. Attente de la disponibilité du Worker ──
    barrier.wait();

    // ── 3. Construction des 50 candidats de stress ──
    let mut population: Vec<StubCandidate> = Vec::with_capacity(50);

    // 10 candidats parfaits (valides avec score)
    for i in 0..10u64 {
        population.push(StubCandidate {
            id: i,
            source: format!("valid_fn_{i}"),
        });
    }

    // 20 candidats syntaxiquement faux
    for i in 10..30u64 {
        population.push(StubCandidate {
            id: i,
            source: format!("syntax_error_fn_{i}"),
        });
    }

    // 20 candidats avec boucle infinie
    for i in 30..50u64 {
        population.push(StubCandidate {
            id: i,
            source: format!("loop_infinite_fn_{i}"),
        });
    }

    // ── 4. Dispatch distribué via Rayon ──
    let workers = vec!["127.0.0.1:19999".to_string()];
    let trial = Trial {
        generation: 0,
        seed: 42,
    };
    let failure_sink = Mutex::new(Vec::new());

    let individuals: Vec<Individual<StubCandidate>> = evaluate_parallel_distributed(
        &population,
        &workers,
        STUB_DOMAIN,
        &trial,
        None, // pas de registre Sled dans ce test
        None,
        0,
        &failure_sink,
    );

    // ── 5. Assertions ──

    // 5a. 100% des 50 réponses collectées
    assert_eq!(
        individuals.len(),
        50,
        "Tous les candidats doivent avoir une réponse (50 attendus)"
    );

    // 5b. 10 valides
    let valid_count = individuals.iter().filter(|i| i.score.valid).count();
    assert_eq!(
        valid_count, 10,
        "10 candidats valides attendus, trouvé {valid_count}"
    );

    // 5c. 40 invalides (20 syntax + 20 loop)
    let invalid_count = individuals.iter().filter(|i| !i.score.valid).count();
    assert_eq!(
        invalid_count, 40,
        "40 candidats invalides attendus, trouvé {invalid_count}"
    );

    // 5d. Vérification des objectifs pour les valides
    for ind in &individuals {
        if ind.score.valid {
            assert!(
                !ind.score.objectives.is_empty(),
                "Les candidats valides doivent avoir des objectifs"
            );
            assert!(
                ind.score.objectives.iter().all(|x| x.is_finite()),
                "Les objectifs doivent être finis"
            );
            // Vérification du matching candidate_id
            assert!(
                ind.cand.id < 10,
                "Seuls les 10 premiers candidats doivent être valides"
            );
        }
    }

    // 5e. Aucune corruption du canal : chaque individu correspond à son candidat
    for (idx, ind) in individuals.iter().enumerate() {
        assert_eq!(
            ind.cand.id, idx as u64,
            "L'ordre des candidats doit être préservé (candidat {idx})"
        );
    }

    // 5f. Les failure_diagnostics devraient être vides (pas de panne réseau)
    let failures = failure_sink.into_inner().unwrap();
    assert!(
        failures.is_empty(),
        "Aucun diagnostic d'échec réseau attendu (toutes les connexions réussissent). \
         Reçu {} échecs: {:?}",
        failures.len(),
        failures
    );

    // 5g. Vérification des erreurs worker (devrait être vide si tout s'est bien passé)
    let worker_errs = worker_errors.lock().unwrap();
    assert!(
        worker_errs.is_empty(),
        "Le Worker mock ne doit signaler aucune erreur. Reçu: {:?}",
        *worker_errs
    );

    // ── 6. Arrêt propre du Worker ──
    shutdown.store(true, Ordering::Relaxed);
    // On laisse le worker se terminer — le join a un timeout court
    let _ = worker_handle.join();
}

// ---------------------------------------------------------------------------
// Test de robustesse réseau : worker injoignable
// ---------------------------------------------------------------------------

#[test]
fn test_distributed_worker_unreachable_is_resilient() {
    let population: Vec<StubCandidate> = (0..5u64)
        .map(|i| StubCandidate {
            id: i,
            source: format!("valid_fn_{i}"),
        })
        .collect();

    let workers = vec!["127.0.0.1:19998".to_string()]; // port sans écoute
    let trial = Trial {
        generation: 0,
        seed: 42,
    };
    let failure_sink = Mutex::new(Vec::new());

    let individuals: Vec<Individual<StubCandidate>> = evaluate_parallel_distributed(
        &population,
        &workers,
        STUB_DOMAIN,
        &trial,
        None,
        None,
        0,
        &failure_sink,
    );

    // Tous les candidats doivent revenir (marqués invalides)
    assert_eq!(individuals.len(), 5);

    // Tous invalides car le worker est injoignable
    for ind in &individuals {
        assert!(!ind.score.valid, "Worker injoignable → tous invalides");
    }

    // Des diagnostics d'échec doivent avoir été collectés
    let failures = failure_sink.into_inner().unwrap();
    assert!(
        !failures.is_empty(),
        "Des diagnostics d'échec réseau doivent être produits"
    );

    // Vérifier que les diagnostics mentionnent bien le worker
    for diag in &failures {
        assert!(
            diag.stderr.contains("127.0.0.1:19998")
                || diag.stderr.contains("Connexion")
                || diag.stderr.contains("connection"),
            "Le diagnostic doit référencer l'adresse du worker: {}",
            diag.stderr
        );
    }
}

// ---------------------------------------------------------------------------
// Test Round-Robin sur plusieurs workers (même worker simulé)
// ---------------------------------------------------------------------------

#[test]
fn test_round_robin_distribution() {
    // Démarre un worker mock pour le Round-Robin
    let barrier = Arc::new(Barrier::new(2));
    let shutdown = Arc::new(AtomicBool::new(false));

    let b = barrier.clone();
    let s = shutdown.clone();

    let _handle = thread::spawn(move || {
        let listener = TcpListener::bind("127.0.0.1:19997").unwrap();
        listener.set_nonblocking(true).unwrap();
        b.wait();

        loop {
            if s.load(Ordering::Relaxed) {
                break;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    thread::spawn(move || {
                        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                        let payload: EvaluationPayload = read_frame(&mut stream);
                        let result = evaluate_stub(&payload);
                        write_frame(&mut stream, &result);
                    });
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
    });

    barrier.wait();

    // 10 candidats, 2 workers (même adresse répétée = Round-Robin vers même worker)
    let population: Vec<StubCandidate> = (0..10u64)
        .map(|i| StubCandidate {
            id: i,
            source: format!("valid_fn_{i}"),
        })
        .collect();

    // Simuler 2 workers (même adresse pour le test — dans la pratique ce
    // seraient des machines différentes)
    let workers = vec!["127.0.0.1:19997".to_string(), "127.0.0.1:19997".to_string()];
    let trial = Trial {
        generation: 0,
        seed: 42,
    };
    let failure_sink = Mutex::new(Vec::new());

    let individuals: Vec<Individual<StubCandidate>> = evaluate_parallel_distributed(
        &population,
        &workers,
        STUB_DOMAIN,
        &trial,
        None,
        None,
        0,
        &failure_sink,
    );

    assert_eq!(individuals.len(), 10);
    // Tous valides
    assert!(individuals.iter().all(|i| i.score.valid));

    shutdown.store(true, Ordering::Relaxed);
}
