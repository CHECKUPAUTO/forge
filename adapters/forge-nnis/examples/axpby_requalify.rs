use forge_kernel_agent::{
    evaluate_candidate, KernelCandidate, KernelEvaluation, KernelLaunchPolicy,
    KernelSourceLanguage, KernelTask, NumericalContract,
};
use forge_nnis::NnisAxpbyBackend;
use nnis_bench::BenchConfig;
use serde_json::{json, Value};
use std::error::Error;

const BASELINE_BLOCK_SIZE: u32 = 256;
const DISCOVERY_WINNER_BLOCK_SIZE: u32 = 128;

const CORRECT_CANDIDATE: &str = r#"
extern "C" __global__ void forge_axpby_f32(
    const float* left,
    const float* right,
    float* output,
    float alpha,
    float beta,
    int elements
) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) {
        output[index] = fmaf(alpha, left[index], beta * right[index]);
    }
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairOrder {
    BaselineThenCandidate,
    CandidateThenBaseline,
}

impl PairOrder {
    fn for_round(round: usize) -> Self {
        if round % 2 == 0 {
            Self::BaselineThenCandidate
        } else {
            Self::CandidateThenBaseline
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::BaselineThenCandidate => "baseline_then_candidate",
            Self::CandidateThenBaseline => "candidate_then_baseline",
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let run_context = std::env::var("NNIS_BENCH_RUN_CONTEXT_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or("NNIS_BENCH_RUN_CONTEXT_ID must be set for fail-closed benchmark evidence")?;

    let elements = env_u64("FORGE_NNIS_AXPBY_ELEMENTS", 1 << 20)?;
    let warmups = env_usize("FORGE_NNIS_AXPBY_WARMUPS", 20)?;
    let iterations = env_usize("FORGE_NNIS_AXPBY_ITERATIONS", 100)?;
    let rounds = env_usize("FORGE_NNIS_AXPBY_REQUALIFY_ROUNDS", 8)?;
    if rounds < 2 {
        return Err("FORGE_NNIS_AXPBY_REQUALIFY_ROUNDS must be at least 2".into());
    }

    let task = KernelTask::new(
        format!("axpby-f32-n{elements}"),
        "axpby",
        NumericalContract::f32_strict(1.0e-6, 1.0e-6),
    )
    .with_dimension("elements", elements);
    let backend =
        NnisAxpbyBackend::first()?.with_bench_config(BenchConfig::new(warmups, iterations))?;

    let baseline = KernelCandidate::new(KernelSourceLanguage::CudaCpp, CORRECT_CANDIDATE)
        .with_launch_policy(KernelLaunchPolicy::block_x(BASELINE_BLOCK_SIZE));
    let candidate = KernelCandidate::new(KernelSourceLanguage::CudaCpp, CORRECT_CANDIDATE)
        .with_launch_policy(KernelLaunchPolicy::block_x(DISCOVERY_WINNER_BLOCK_SIZE));

    let mut expected_environment_id: Option<String> = None;
    let mut baseline_round_medians = Vec::with_capacity(rounds);
    let mut candidate_round_medians = Vec::with_capacity(rounds);
    let mut paired_relative_improvements = Vec::with_capacity(rounds);
    let mut candidate_wins = 0usize;
    let mut baseline_wins = 0usize;
    let mut ties = 0usize;
    let mut observations = Vec::<Value>::with_capacity(rounds);

    for round in 0..rounds {
        let order = PairOrder::for_round(round);
        let (baseline_evaluation, candidate_evaluation) = match order {
            PairOrder::BaselineThenCandidate => (
                evaluate_selectable(&backend, &task, &baseline)?,
                evaluate_selectable(&backend, &task, &candidate)?,
            ),
            PairOrder::CandidateThenBaseline => {
                let candidate_evaluation = evaluate_selectable(&backend, &task, &candidate)?;
                let baseline_evaluation = evaluate_selectable(&backend, &task, &baseline)?;
                (baseline_evaluation, candidate_evaluation)
            }
        };

        require_same_environment(&mut expected_environment_id, &baseline_evaluation)?;
        require_same_environment(&mut expected_environment_id, &candidate_evaluation)?;

        let baseline_median = median_ms(&baseline_evaluation)?;
        let candidate_median = median_ms(&candidate_evaluation)?;
        let relative_improvement = (baseline_median - candidate_median) / baseline_median;

        match candidate_median.total_cmp(&baseline_median) {
            std::cmp::Ordering::Less => candidate_wins += 1,
            std::cmp::Ordering::Greater => baseline_wins += 1,
            std::cmp::Ordering::Equal => ties += 1,
        }

        baseline_round_medians.push(baseline_median);
        candidate_round_medians.push(candidate_median);
        paired_relative_improvements.push(relative_improvement);
        observations.push(json!({
            "round": round,
            "order": order.as_str(),
            "baseline_median_ms": baseline_median,
            "candidate_median_ms": candidate_median,
            "relative_improvement": relative_improvement,
            "baseline_evaluation": baseline_evaluation,
            "candidate_evaluation": candidate_evaluation,
        }));
    }

    let median_baseline_ms = median(&baseline_round_medians)?;
    let median_candidate_ms = median(&candidate_round_medians)?;
    let median_paired_relative_improvement = median(&paired_relative_improvements)?;
    let aggregate_ratio = median_baseline_ms / median_candidate_ms;

    let result = json!({
        "schema_version": 1,
        "campaign_kind": "forge_nnis_axpby_paired_requalification_v1",
        "run_context_id": run_context,
        "baseline_block_size": BASELINE_BLOCK_SIZE,
        "candidate_block_size": DISCOVERY_WINNER_BLOCK_SIZE,
        "rounds": rounds,
        "warmup_iterations_per_observation": warmups,
        "measured_iterations_per_observation": iterations,
        "environment_id": expected_environment_id,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "median_baseline_ms": median_baseline_ms,
        "median_candidate_ms": median_candidate_ms,
        "median_paired_relative_improvement": median_paired_relative_improvement,
        "aggregate_microbenchmark_speedup_ratio": aggregate_ratio,
        "claim_boundary": "paired isolated AXPBY kernel requalification only; no end-to-end NNIS speedup claim and no automatic promotion decision",
        "observations": observations,
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn evaluate_selectable(
    backend: &NnisAxpbyBackend,
    task: &KernelTask,
    candidate: &KernelCandidate,
) -> Result<KernelEvaluation, Box<dyn Error>> {
    let evaluation = evaluate_candidate(backend, task, candidate)?;
    if !evaluation.verification.passed {
        return Err(format!("candidate {} failed verification", candidate.id).into());
    }
    if evaluation.measurement.is_none() {
        return Err(format!("candidate {} produced no measurement", candidate.id).into());
    }
    Ok(evaluation)
}

fn require_same_environment(
    expected: &mut Option<String>,
    evaluation: &KernelEvaluation,
) -> Result<(), Box<dyn Error>> {
    let environment_id = evaluation
        .measurement
        .as_ref()
        .ok_or("selectable evaluation unexpectedly has no measurement")?
        .environment_id
        .clone();
    match expected {
        Some(expected_id) if expected_id != &environment_id => {
            Err("benchmark environment changed during paired requalification".into())
        }
        Some(_) => Ok(()),
        None => {
            *expected = Some(environment_id);
            Ok(())
        }
    }
}

fn median_ms(evaluation: &KernelEvaluation) -> Result<f64, Box<dyn Error>> {
    let measurement = evaluation
        .measurement
        .as_ref()
        .ok_or("selectable evaluation unexpectedly has no measurement")?;
    let value = measurement
        .metrics
        .get("median_ms")
        .copied()
        .ok_or("measurement is missing median_ms")?;
    if !value.is_finite() || value <= 0.0 {
        return Err("measurement median_ms must be finite and positive".into());
    }
    Ok(value)
}

fn median(values: &[f64]) -> Result<f64, Box<dyn Error>> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("median requires non-empty finite values".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let midpoint = sorted.len() / 2;
    Ok(if sorted.len() % 2 == 0 {
        (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
    } else {
        sorted[midpoint]
    })
}

fn env_u64(name: &str, default: u64) -> Result<u64, Box<dyn Error>> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(default))
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(default))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_order_alternates_deterministically() {
        assert_eq!(PairOrder::for_round(0), PairOrder::BaselineThenCandidate);
        assert_eq!(PairOrder::for_round(1), PairOrder::CandidateThenBaseline);
        assert_eq!(PairOrder::for_round(2), PairOrder::BaselineThenCandidate);
        assert_eq!(PairOrder::for_round(3), PairOrder::CandidateThenBaseline);
    }

    #[test]
    fn median_handles_even_and_odd_observation_counts() {
        assert_eq!(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
    }

    #[test]
    fn requalification_candidates_differ_only_by_launch_policy() {
        let baseline = KernelCandidate::new(KernelSourceLanguage::CudaCpp, CORRECT_CANDIDATE)
            .with_launch_policy(KernelLaunchPolicy::block_x(BASELINE_BLOCK_SIZE));
        let candidate = KernelCandidate::new(KernelSourceLanguage::CudaCpp, CORRECT_CANDIDATE)
            .with_launch_policy(KernelLaunchPolicy::block_x(DISCOVERY_WINNER_BLOCK_SIZE));

        assert_eq!(baseline.source, candidate.source);
        assert_eq!(baseline.source_language, candidate.source_language);
        assert_ne!(baseline.launch_policy, candidate.launch_policy);
        assert_ne!(baseline.id, candidate.id);
    }
}
