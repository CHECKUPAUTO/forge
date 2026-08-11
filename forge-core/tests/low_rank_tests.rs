use forge_core::domains::low_rank::{TensorCode, TensorTrainDomain};
use forge_core::{Domain, Trial};
use rand::SeedableRng;

#[test]
fn test_low_rank_baseline_compiles_and_runs() {
    let workspace = "/tmp/forge_lowrank_test";
    let _ = std::fs::remove_dir_all(workspace);

    let domain = TensorTrainDomain::new(workspace);
    let cand = domain.seed(&mut rand::rngs::StdRng::seed_from_u64(42));
    let trial = Trial {
        generation: 0,
        seed: 100,
    };

    // Le candidat baseline doit passer la vérification
    let valid = domain
        .verify(&cand, &trial)
        .expect("verify should not error");
    assert!(
        valid,
        "Baseline candidate should compile and run successfully"
    );

    // Nettoyage
    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn test_low_rank_invalid_code_fails_verify() {
    let workspace = "/tmp/forge_lowrank_invalid";
    let _ = std::fs::remove_dir_all(workspace);

    let domain = TensorTrainDomain::new(workspace);
    let bad_cand = TensorCode {
        raw_source: "invalid rust code !!!!".to_string(),
        id: forge_core::fnv1a("bad_code"),
    };
    let trial = Trial {
        generation: 0,
        seed: 42,
    };

    let valid = domain
        .verify(&bad_cand, &trial)
        .expect("verify should not error");
    assert!(!valid, "Invalid code should fail verification");

    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn test_low_rank_measure_returns_three_objectives() {
    let workspace = "/tmp/forge_lowrank_measure";
    let _ = std::fs::remove_dir_all(workspace);

    let domain = TensorTrainDomain::new(workspace);
    let cand = domain.seed(&mut rand::rngs::StdRng::seed_from_u64(42));
    let trial = Trial {
        generation: 0,
        seed: 200,
    };

    let objectives = domain
        .measure(&cand, &trial)
        .expect("measure should succeed");
    assert_eq!(objectives.len(), 3);
    assert!(objectives.iter().all(|v| v.is_finite()));

    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn test_low_rank_objective_names() {
    let domain = TensorTrainDomain::new("/tmp/irrelevant");
    let names = domain.objective_names();
    assert_eq!(names.len(), 3);
    assert_eq!(names[0], "reconstruction_error_L2");
    assert_eq!(names[1], "latency_ns");
    assert_eq!(names[2], "parameters_count");
}
