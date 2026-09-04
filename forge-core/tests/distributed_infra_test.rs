//! Stress tests for the distributed Forge evaluation protocol.
//!
//! The mock worker speaks the production bincode framing and echoes the exact
//! candidate, source and trial identity carried by the request. Listener ports
//! are allocated by the OS so parallel CI jobs do not depend on fixed ports.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use forge_core::evaluate_parallel_distributed;
use forge_core::protocol::{
    EvaluationPayload, EvaluationResult, WorkerExecutionContext, BENCHMARK_PROTOCOL,
    PROTOCOL_VERSION, WORKER_DESCRIPTOR_VERSION,
};
use forge_core::{fnv1a, Candidate, CandidateId, Individual, Trial};

const STUB_DOMAIN: &str = "stub-domain";

type WorkerHandle = (
    String,
    Arc<AtomicBool>,
    JoinHandle<()>,
    Arc<Mutex<Vec<String>>>,
);

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

fn worker_context() -> WorkerExecutionContext {
    WorkerExecutionContext {
        descriptor_version: WORKER_DESCRIPTOR_VERSION,
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
        trial_seed: payload.seed,
        generation: payload.generation,
        domain: STUB_DOMAIN.into(),
        benchmark_protocol: BENCHMARK_PROTOCOL.into(),
        execution_context: worker_context(),
        is_valid,
        objectives,
        error_message,
    }
}

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
        let base_latency = 1000.0 + (payload.candidate_id as f64 % 100.0) * 10.0;
        result_for(payload, true, vec![0.001, base_latency, 50.0], None)
    }
}

fn spawn_worker() -> WorkerHandle {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock worker");
    listener
        .set_nonblocking(true)
        .expect("set mock worker nonblocking");
    let addr = listener.local_addr().expect("mock worker address").to_string();
    let shutdown = Arc::new(AtomicBool::new(false));
    let worker_errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let thread_shutdown = Arc::clone(&shutdown);
    let thread_errors = Arc::clone(&worker_errors);

    let handle = thread::spawn(move || {
        while !thread_shutdown.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    thread::spawn(move || {
                        let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                        let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
                        let payload: EvaluationPayload = read_frame(&mut stream);
                        let result = evaluate_stub(&payload);
                        write_frame(&mut stream, &result);
                    });
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(err) => {
                    thread_errors
                        .lock()
                        .expect("worker error mutex")
                        .push(format!("accept error: {err}"));
                    break;
                }
            }
        }
    });

    (addr, shutdown, handle, worker_errors)
}

fn stop_worker(shutdown: Arc<AtomicBool>, handle: JoinHandle<()>) {
    shutdown.store(true, Ordering::Relaxed);
    handle.join().expect("join mock worker");
}

#[test]
fn worker_result_echoes_exact_trial_identity() {
    let payload = EvaluationPayload {
        candidate_id: 7,
        source_code: "valid_fn_7".into(),
        seed: 0xA5A5,
        generation: 19,
    };
    let result = evaluate_stub(&payload);
    assert_eq!(result.trial_seed, payload.seed);
    assert_eq!(result.generation, payload.generation);
    assert_eq!(result.execution_context.descriptor_version, WORKER_DESCRIPTOR_VERSION);
}

#[test]
fn test_distributed_evolution_under_stress() {
    let (addr, shutdown, worker_handle, worker_errors) = spawn_worker();

    let mut population = Vec::with_capacity(50);
    for i in 0..10u64 {
        population.push(StubCandidate {
            id: i,
            source: format!("valid_fn_{i}"),
        });
    }
    for i in 10..30u64 {
        population.push(StubCandidate {
            id: i,
            source: format!("syntax_error_fn_{i}"),
        });
    }
    for i in 30..50u64 {
        population.push(StubCandidate {
            id: i,
            source: format!("loop_infinite_fn_{i}"),
        });
    }

    let workers = vec![addr];
    let trial = Trial {
        generation: 11,
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
        trial.generation,
        &failure_sink,
    );

    assert_eq!(individuals.len(), 50);
    assert_eq!(individuals.iter().filter(|i| i.score.valid).count(), 10);
    assert_eq!(individuals.iter().filter(|i| !i.score.valid).count(), 40);
    for (idx, individual) in individuals.iter().enumerate() {
        assert_eq!(individual.cand.id, idx as u64);
        if individual.score.valid {
            assert!(individual.cand.id < 10);
            assert!(!individual.score.objectives.is_empty());
            assert!(individual.score.objectives.iter().all(|x| x.is_finite()));
        }
    }
    assert!(failure_sink.into_inner().expect("failure sink").is_empty());

    stop_worker(shutdown, worker_handle);
    assert!(worker_errors
        .lock()
        .expect("worker errors")
        .is_empty());
}

#[test]
fn test_distributed_worker_unreachable_is_resilient() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve unused port");
    let addr = listener.local_addr().expect("unused address").to_string();
    drop(listener);

    let population: Vec<StubCandidate> = (0..5u64)
        .map(|i| StubCandidate {
            id: i,
            source: format!("valid_fn_{i}"),
        })
        .collect();
    let workers = vec![addr.clone()];
    let trial = Trial {
        generation: 3,
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
        trial.generation,
        &failure_sink,
    );

    assert_eq!(individuals.len(), 5);
    assert!(individuals.iter().all(|i| !i.score.valid));
    let failures = failure_sink.into_inner().expect("failure sink");
    assert!(!failures.is_empty());
    assert!(failures.iter().all(|diag| {
        diag.stderr.contains(&addr)
            || diag.stderr.contains("Connexion")
            || diag.stderr.contains("connection")
    }));
}

#[test]
fn test_round_robin_distribution() {
    let (addr, shutdown, worker_handle, worker_errors) = spawn_worker();
    let population: Vec<StubCandidate> = (0..10u64)
        .map(|i| StubCandidate {
            id: i,
            source: format!("valid_fn_{i}"),
        })
        .collect();
    let workers = vec![addr.clone(), addr];
    let trial = Trial {
        generation: 5,
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
        trial.generation,
        &failure_sink,
    );

    assert_eq!(individuals.len(), 10);
    assert!(individuals.iter().all(|i| i.score.valid));
    assert!(failure_sink.into_inner().expect("failure sink").is_empty());

    stop_worker(shutdown, worker_handle);
    assert!(worker_errors
        .lock()
        .expect("worker errors")
        .is_empty());
}
