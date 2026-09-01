use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkMetadata, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::F32RmsNorm;
use nnis_rt::{gpu_context, DeviceBuffer, Stream};
use serde_json::{json, Value};
use std::error::Error;

const BASELINE_BLOCK_SIZE: u32 = 256;
const DISCOVERY_WINNER_BLOCK_SIZE: u32 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairOrder {
    BaselineThenCandidate,
    CandidateThenBaseline,
}

impl PairOrder {
    fn for_round(round: usize) -> Self {
        if round.is_multiple_of(2) {
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
    let rows = env_usize("FORGE_NNIS_RMSNORM_ROWS", 32)?;
    let cols = env_usize("FORGE_NNIS_RMSNORM_COLS", 4096)?;
    let warmups = env_usize("FORGE_NNIS_RMSNORM_WARMUPS", 20)?;
    let iterations = env_usize("FORGE_NNIS_RMSNORM_ITERATIONS", 100)?;
    let rounds = env_usize("FORGE_NNIS_RMSNORM_REQUALIFY_ROUNDS", 8)?;
    let epsilon = env_f32("FORGE_NNIS_RMSNORM_EPSILON", 1.0e-6)?;
    let gamma = env_f32("FORGE_NNIS_RMSNORM_GAMMA", 1.0)?;
    let atol = env_f64("FORGE_NNIS_RMSNORM_ATOL", 5.0e-5)?;
    let rtol = env_f64("FORGE_NNIS_RMSNORM_RTOL", 5.0e-5)?;

    if rows == 0 || cols == 0 {
        return Err("RMSNorm rows and cols must be non-zero".into());
    }
    if rounds < 2 {
        return Err("FORGE_NNIS_RMSNORM_REQUALIFY_ROUNDS must be at least 2".into());
    }
    if !epsilon.is_finite() || epsilon <= 0.0 || !gamma.is_finite() {
        return Err("RMSNorm epsilon/gamma must be finite and epsilon must be positive".into());
    }
    if !atol.is_finite() || atol < 0.0 || !rtol.is_finite() || rtol < 0.0 {
        return Err("RMSNorm tolerances must be finite and non-negative".into());
    }

    let elements = rows
        .checked_mul(cols)
        .ok_or("RMSNorm shape overflows usize")?;
    let bytes_per_iteration = (elements as u64)
        .checked_mul(2 * std::mem::size_of::<f32>() as u64)
        .ok_or("RMSNorm byte count overflows u64")?;
    let bench_config = BenchConfig::new(warmups, iterations);
    bench_config.validate()?;

    let context = gpu_context().ok_or("no CUDA device is available")?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let baseline = F32RmsNorm::load_with_block_size(&context, &compiler, BASELINE_BLOCK_SIZE)?;
    let candidate =
        F32RmsNorm::load_with_block_size(&context, &compiler, DISCOVERY_WINNER_BLOCK_SIZE)?;
    if !baseline.fused_available(cols) {
        return Err("canonical block-256 RMSNorm fused path is unavailable for this shape".into());
    }
    if !candidate.fused_available(cols) {
        return Err("canonical block-512 RMSNorm fused path is unavailable for this shape".into());
    }

    let input_host = deterministic_input(elements);
    let input = DeviceBuffer::from_host(&context, &stream, &input_host)?;
    let output = DeviceBuffer::<f32>::new(&context, elements)?;

    let baseline_verification = verify_family(
        &baseline,
        &stream,
        &input,
        &output,
        &input_host,
        rows,
        cols,
        epsilon,
        gamma,
        atol,
        rtol,
    )?;
    let candidate_verification = verify_family(
        &candidate,
        &stream,
        &input,
        &output,
        &input_host,
        rows,
        cols,
        epsilon,
        gamma,
        atol,
        rtol,
    )?;
    if !baseline_verification.passed || !candidate_verification.passed {
        return Err(
            "RMSNorm paired requalification requires both launch variants to verify".into(),
        );
    }

    let mut expected_metadata: Option<BenchmarkMetadata> = None;
    let mut baseline_round_medians = Vec::with_capacity(rounds);
    let mut candidate_round_medians = Vec::with_capacity(rounds);
    let mut paired_relative_improvements = Vec::with_capacity(rounds);
    let mut candidate_wins = 0usize;
    let mut baseline_wins = 0usize;
    let mut ties = 0usize;
    let mut observations = Vec::<Value>::with_capacity(rounds);

    for round in 0..rounds {
        let order = PairOrder::for_round(round);
        let (baseline_report, candidate_report) = match order {
            PairOrder::BaselineThenCandidate => (
                measure_family(
                    &baseline,
                    BASELINE_BLOCK_SIZE,
                    &context,
                    &stream,
                    &input,
                    &output,
                    rows,
                    cols,
                    epsilon,
                    gamma,
                    bytes_per_iteration,
                    bench_config,
                )?,
                measure_family(
                    &candidate,
                    DISCOVERY_WINNER_BLOCK_SIZE,
                    &context,
                    &stream,
                    &input,
                    &output,
                    rows,
                    cols,
                    epsilon,
                    gamma,
                    bytes_per_iteration,
                    bench_config,
                )?,
            ),
            PairOrder::CandidateThenBaseline => {
                let candidate_report = measure_family(
                    &candidate,
                    DISCOVERY_WINNER_BLOCK_SIZE,
                    &context,
                    &stream,
                    &input,
                    &output,
                    rows,
                    cols,
                    epsilon,
                    gamma,
                    bytes_per_iteration,
                    bench_config,
                )?;
                let baseline_report = measure_family(
                    &baseline,
                    BASELINE_BLOCK_SIZE,
                    &context,
                    &stream,
                    &input,
                    &output,
                    rows,
                    cols,
                    epsilon,
                    gamma,
                    bytes_per_iteration,
                    bench_config,
                )?;
                (baseline_report, candidate_report)
            }
        };

        baseline_report
            .metadata
            .require_compatible_environment(&candidate_report.metadata)?;
        require_same_environment(&mut expected_metadata, &baseline_report.metadata)?;
        require_same_environment(&mut expected_metadata, &candidate_report.metadata)?;

        let baseline_median = positive_median(&baseline_report)?;
        let candidate_median = positive_median(&candidate_report)?;
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
            "baseline_report": baseline_report,
            "candidate_report": candidate_report,
        }));
    }

    let median_baseline_ms = median(&baseline_round_medians)?;
    let median_candidate_ms = median(&candidate_round_medians)?;
    let median_paired_relative_improvement = median(&paired_relative_improvements)?;
    let aggregate_ratio = median_baseline_ms / median_candidate_ms;

    let result = json!({
        "schema_version": 1,
        "campaign_kind": "forge_nnis_canonical_rmsnorm_paired_requalification_v1",
        "run_context_id": run_context,
        "rows": rows,
        "cols": cols,
        "epsilon": epsilon,
        "gamma": gamma,
        "atol": atol,
        "rtol": rtol,
        "baseline_block_size": BASELINE_BLOCK_SIZE,
        "candidate_block_size": DISCOVERY_WINNER_BLOCK_SIZE,
        "rounds": rounds,
        "warmup_iterations_per_observation": warmups,
        "measured_iterations_per_observation": iterations,
        "baseline_verification": baseline_verification.to_json(),
        "candidate_verification": candidate_verification.to_json(),
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "median_baseline_ms": median_baseline_ms,
        "median_candidate_ms": median_candidate_ms,
        "median_paired_relative_improvement": median_paired_relative_improvement,
        "aggregate_microbenchmark_speedup_ratio": aggregate_ratio,
        "claim_boundary": "paired canonical NNIS F32RmsNorm fused-path launch requalification only; no generated-kernel claim, no cross-shape policy claim, no end-to-end NNIS speedup claim, and no automatic production-default promotion",
        "observations": observations,
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn measure_family(
    rms_norm: &F32RmsNorm,
    block_size: u32,
    context: &std::sync::Arc<nnis_rt::Context>,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
    epsilon: f32,
    gamma: f32,
    bytes_per_iteration: u64,
    bench_config: BenchConfig,
) -> Result<BenchmarkReport, Box<dyn Error>> {
    let case = BenchmarkCase::new("nnis_f32_rmsnorm_fused", "f32")
        .with_dimension("rows", rows as u64)
        .with_dimension("cols", cols as u64)
        .with_dimension("block_size", u64::from(block_size))
        .with_work_items((rows * cols) as u64)
        .with_bytes_per_iteration(bytes_per_iteration);
    let report = benchmark_gpu(context, stream, case, bench_config, || {
        // SAFETY: all buffers, the stream and this kernel family outlive the
        // benchmark call; benchmark_gpu synchronizes every measured launch.
        unsafe { rms_norm.enqueue_fused_rows(stream, input, output, rows, cols, epsilon, gamma) }
    })?;
    report
        .metadata
        .require_compatible_environment(&report.metadata)?;
    Ok(report)
}

fn require_same_environment(
    expected: &mut Option<BenchmarkMetadata>,
    metadata: &BenchmarkMetadata,
) -> Result<(), Box<dyn Error>> {
    match expected {
        Some(reference) => {
            reference.require_compatible_environment(metadata)?;
            Ok(())
        }
        None => {
            metadata.require_compatible_environment(metadata)?;
            *expected = Some(metadata.clone());
            Ok(())
        }
    }
}

fn positive_median(report: &BenchmarkReport) -> Result<f64, Box<dyn Error>> {
    let value = report.statistics.median_ms;
    if !value.is_finite() || value <= 0.0 {
        return Err("benchmark median_ms must be finite and positive".into());
    }
    Ok(value)
}

#[derive(Debug, Clone, Copy)]
struct VerificationSummary {
    passed: bool,
    max_abs_error: f64,
    max_rel_error: f64,
}

impl VerificationSummary {
    fn to_json(self) -> Value {
        json!({
            "passed": self.passed,
            "oracle_id": "forge-nnis/canonical-rmsnorm-f64-host-oracle-v1",
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_family(
    rms_norm: &F32RmsNorm,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    input_host: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
    gamma: f32,
    atol: f64,
    rtol: f64,
) -> Result<VerificationSummary, Box<dyn Error>> {
    rms_norm.fused_normalize_rows(stream, input, output, rows, cols, epsilon, gamma)?;
    let actual = output.to_vec(stream)?;
    Ok(verify_rmsnorm(
        input_host, &actual, rows, cols, epsilon, gamma, atol, rtol,
    ))
}

#[allow(clippy::too_many_arguments)]
fn verify_rmsnorm(
    input: &[f32],
    actual: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
    gamma: f32,
    atol: f64,
    rtol: f64,
) -> VerificationSummary {
    if input.len() != rows.saturating_mul(cols) || actual.len() != input.len() {
        return VerificationSummary {
            passed: false,
            max_abs_error: f64::INFINITY,
            max_rel_error: f64::INFINITY,
        };
    }

    let mut passed = true;
    let mut max_abs_error = 0.0_f64;
    let mut max_rel_error = 0.0_f64;
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let sumsq: f64 = input[start..end]
            .iter()
            .map(|value| {
                let value = f64::from(*value);
                value * value
            })
            .sum();
        let scale = f64::from(gamma) / (sumsq / cols as f64 + f64::from(epsilon)).sqrt();
        for index in start..end {
            let expected = f64::from(input[index]) * scale;
            let observed = f64::from(actual[index]);
            if !observed.is_finite() {
                passed = false;
                continue;
            }
            let abs_error = (observed - expected).abs();
            let rel_error = abs_error / expected.abs().max(f64::from(f32::MIN_POSITIVE));
            max_abs_error = max_abs_error.max(abs_error);
            max_rel_error = max_rel_error.max(rel_error);
            if abs_error > atol + rtol * expected.abs() {
                passed = false;
            }
        }
    }
    VerificationSummary {
        passed,
        max_abs_error,
        max_rel_error,
    }
}

fn deterministic_input(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let mixed = (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bucket = ((mixed >> 32) % 4093) as i32 - 2046;
            (bucket as f32) * (1.0 / 1024.0) + ((index % 7) as f32 - 3.0) * 0.03125
        })
        .collect()
}

fn median(values: &[f64]) -> Result<f64, Box<dyn Error>> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("median requires non-empty finite values".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let midpoint = sorted.len() / 2;
    Ok(if sorted.len().is_multiple_of(2) {
        (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
    } else {
        sorted[midpoint]
    })
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(default))
}

fn env_f32(name: &str, default: f32) -> Result<f32, Box<dyn Error>> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse::<f32>())
        .transpose()?
        .unwrap_or(default))
}

fn env_f64(name: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse::<f64>())
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
    fn discovery_winner_is_requalified_against_the_original_baseline() {
        assert_eq!(BASELINE_BLOCK_SIZE, 256);
        assert_eq!(DISCOVERY_WINNER_BLOCK_SIZE, 512);
    }

    #[test]
    fn oracle_accepts_exact_known_rmsnorm() {
        let input = vec![3.0_f32, 4.0_f32];
        let rms = ((9.0_f64 + 16.0) / 2.0).sqrt();
        let actual = vec![(3.0 / rms) as f32, (4.0 / rms) as f32];
        let summary = verify_rmsnorm(&input, &actual, 1, 2, 0.0, 1.0, 1.0e-6, 1.0e-6);
        assert!(summary.passed);
    }
}
