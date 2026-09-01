use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkMetadata, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_model::{F32DecoderKernels, F32WeightedRmsNormCandidate};
use nnis_rt::{gpu_context, DeviceBuffer, Stream};
use serde_json::{json, Value};
use std::error::Error;

const ROWS: usize = 1;
const COLS: usize = 576;
const BASELINE_BLOCK_SIZE: u32 = 256;
const CANDIDATE_BLOCK_SIZE: u32 = 512;
const RMS_NORM_EPSILON: f32 = 1.0e-5;
const DEFAULT_WARMUPS: usize = 20;
const DEFAULT_ITERATIONS: usize = 100;
const DEFAULT_ROUNDS: usize = 4;
const DEFAULT_ATOL: f64 = 5.0e-5;
const DEFAULT_RTOL: f64 = 5.0e-5;

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
    let run_context_id = std::env::var("NNIS_BENCH_RUN_CONTEXT_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or("NNIS_BENCH_RUN_CONTEXT_ID must be set for fail-closed benchmark evidence")?;
    let warmups = env_usize("FORGE_NNIS_WEIGHTED_RMSNORM_WARMUPS", DEFAULT_WARMUPS)?;
    let iterations = env_usize("FORGE_NNIS_WEIGHTED_RMSNORM_ITERATIONS", DEFAULT_ITERATIONS)?;
    let rounds = env_usize("FORGE_NNIS_WEIGHTED_RMSNORM_ROUNDS", DEFAULT_ROUNDS)?;
    let atol = env_f64("FORGE_NNIS_WEIGHTED_RMSNORM_ATOL", DEFAULT_ATOL)?;
    let rtol = env_f64("FORGE_NNIS_WEIGHTED_RMSNORM_RTOL", DEFAULT_RTOL)?;
    if rounds < 2 {
        return Err("FORGE_NNIS_WEIGHTED_RMSNORM_ROUNDS must be at least 2".into());
    }
    if !atol.is_finite() || atol < 0.0 || !rtol.is_finite() || rtol < 0.0 {
        return Err("weighted RMSNorm tolerances must be finite and non-negative".into());
    }

    let bench_config = BenchConfig::new(warmups, iterations);
    bench_config.validate()?;
    let context = gpu_context().ok_or("no CUDA device is available")?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let baseline = F32DecoderKernels::load(&context, &compiler)?;
    let candidate = F32WeightedRmsNormCandidate::load(&context, &compiler, CANDIDATE_BLOCK_SIZE)?;

    let input_host = deterministic_input(COLS);
    let weight_host = deterministic_weight(COLS);
    let input = DeviceBuffer::from_host(&context, &stream, &input_host)?;
    let weight = DeviceBuffer::from_host(&context, &stream, &weight_host)?;
    let output = DeviceBuffer::<f32>::new(&context, COLS)?;

    let baseline_verification = verify_baseline(
        &baseline,
        &stream,
        &input,
        &weight,
        &output,
        &input_host,
        &weight_host,
        atol,
        rtol,
    )?;
    let candidate_verification = verify_candidate(
        &candidate,
        &stream,
        &input,
        &weight,
        &output,
        &input_host,
        &weight_host,
        atol,
        rtol,
    )?;
    if !baseline_verification.passed || !candidate_verification.passed {
        return Err("both weighted RMSNorm launch variants must verify before measurement".into());
    }

    let bytes_per_iteration = (COLS as u64)
        .checked_mul(3 * std::mem::size_of::<f32>() as u64)
        .ok_or("weighted RMSNorm byte count overflows u64")?;
    let mut expected_metadata: Option<BenchmarkMetadata> = None;
    let mut baseline_round_medians = Vec::with_capacity(rounds);
    let mut candidate_round_medians = Vec::with_capacity(rounds);
    let mut paired_relative_improvements = Vec::with_capacity(rounds);
    let mut candidate_round_wins = 0usize;
    let mut baseline_round_wins = 0usize;
    let mut round_ties = 0usize;
    let mut observations = Vec::<Value>::with_capacity(rounds);

    for round in 0..rounds {
        let order = PairOrder::for_round(round);
        let (baseline_report, candidate_report) = match order {
            PairOrder::BaselineThenCandidate => (
                measure_baseline(
                    &baseline,
                    &context,
                    &stream,
                    &input,
                    &weight,
                    &output,
                    bytes_per_iteration,
                    bench_config,
                )?,
                measure_candidate(
                    &candidate,
                    &context,
                    &stream,
                    &input,
                    &weight,
                    &output,
                    bytes_per_iteration,
                    bench_config,
                )?,
            ),
            PairOrder::CandidateThenBaseline => {
                let candidate_report = measure_candidate(
                    &candidate,
                    &context,
                    &stream,
                    &input,
                    &weight,
                    &output,
                    bytes_per_iteration,
                    bench_config,
                )?;
                let baseline_report = measure_baseline(
                    &baseline,
                    &context,
                    &stream,
                    &input,
                    &weight,
                    &output,
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

        let baseline_median_ms = positive_median(&baseline_report)?;
        let candidate_median_ms = positive_median(&candidate_report)?;
        let relative_improvement = (baseline_median_ms - candidate_median_ms) / baseline_median_ms;
        match candidate_median_ms.total_cmp(&baseline_median_ms) {
            std::cmp::Ordering::Less => candidate_round_wins += 1,
            std::cmp::Ordering::Greater => baseline_round_wins += 1,
            std::cmp::Ordering::Equal => round_ties += 1,
        }
        baseline_round_medians.push(baseline_median_ms);
        candidate_round_medians.push(candidate_median_ms);
        paired_relative_improvements.push(relative_improvement);
        observations.push(json!({
            "round": round,
            "order": order.as_str(),
            "baseline_median_ms": baseline_median_ms,
            "candidate_median_ms": candidate_median_ms,
            "relative_improvement": relative_improvement,
            "baseline_report": baseline_report,
            "candidate_report": candidate_report,
        }));
    }

    let median_baseline_ms = median(&baseline_round_medians)?;
    let median_candidate_ms = median(&candidate_round_medians)?;
    let result = json!({
        "schema_version": 1,
        "campaign_kind": "forge_nnis_smollm2_weighted_rmsnorm_launch_requalification_v1",
        "run_context_id": run_context_id,
        "model_scope": {
            "model": "HuggingFaceTB/SmolLM2-135M",
            "hidden_size": COLS,
            "num_hidden_layers": 30,
            "runtime_rows": ROWS,
            "weighted_rmsnorm_launches_per_token": 61,
            "rms_norm_eps": RMS_NORM_EPSILON,
        },
        "baseline": {
            "implementation": "nnis-model/F32DecoderKernels::weighted_rms_norm",
            "block_size": BASELINE_BLOCK_SIZE,
        },
        "candidate": {
            "implementation": "nnis-model/F32WeightedRmsNormCandidate",
            "block_size": CANDIDATE_BLOCK_SIZE,
        },
        "warmup_iterations_per_observation": warmups,
        "measured_iterations_per_observation": iterations,
        "rounds": rounds,
        "atol": atol,
        "rtol": rtol,
        "baseline_verification": baseline_verification.to_json(),
        "candidate_verification": candidate_verification.to_json(),
        "candidate_round_wins": candidate_round_wins,
        "baseline_round_wins": baseline_round_wins,
        "round_ties": round_ties,
        "median_baseline_ms": median_baseline_ms,
        "median_candidate_ms": median_candidate_ms,
        "median_paired_relative_improvement": median(&paired_relative_improvements)?,
        "aggregate_microbenchmark_speedup_ratio": median_baseline_ms / median_candidate_ms,
        "claim_boundary": "actual NNIS decoder weighted-RMSNorm 1x576 isolated launch comparison only; no end-to-end SmolLM2 speedup claim and no runtime-default promotion",
        "observations": observations,
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn measure_baseline(
    baseline: &F32DecoderKernels,
    context: &std::sync::Arc<nnis_rt::Context>,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    bytes_per_iteration: u64,
    bench_config: BenchConfig,
) -> Result<BenchmarkReport, Box<dyn Error>> {
    measure(
        BASELINE_BLOCK_SIZE,
        context,
        stream,
        bytes_per_iteration,
        bench_config,
        || {
            // SAFETY: all resources outlive benchmark_gpu and launches are
            // synchronized by the benchmark harness.
            unsafe {
                baseline.enqueue_weighted_rms_norm(
                    stream,
                    input,
                    weight,
                    output,
                    ROWS,
                    COLS,
                    RMS_NORM_EPSILON,
                )
            }
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn measure_candidate(
    candidate: &F32WeightedRmsNormCandidate,
    context: &std::sync::Arc<nnis_rt::Context>,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    bytes_per_iteration: u64,
    bench_config: BenchConfig,
) -> Result<BenchmarkReport, Box<dyn Error>> {
    measure(
        CANDIDATE_BLOCK_SIZE,
        context,
        stream,
        bytes_per_iteration,
        bench_config,
        || {
            // SAFETY: all resources outlive benchmark_gpu and launches are
            // synchronized by the benchmark harness.
            unsafe {
                candidate.enqueue_weighted_rms_norm(
                    stream,
                    input,
                    weight,
                    output,
                    ROWS,
                    COLS,
                    RMS_NORM_EPSILON,
                )
            }
        },
    )
}

fn measure<F>(
    block_size: u32,
    context: &std::sync::Arc<nnis_rt::Context>,
    stream: &Stream,
    bytes_per_iteration: u64,
    bench_config: BenchConfig,
    operation: F,
) -> Result<BenchmarkReport, Box<dyn Error>>
where
    F: FnMut() -> nnis_rt::Result<()>,
{
    let case = BenchmarkCase::new("nnis_model_weighted_rmsnorm_f32", "f32")
        .with_dimension("rows", ROWS as u64)
        .with_dimension("cols", COLS as u64)
        .with_dimension("block_size", u64::from(block_size))
        .with_work_items((ROWS * COLS) as u64)
        .with_bytes_per_iteration(bytes_per_iteration);
    let report = benchmark_gpu(context, stream, case, bench_config, operation)?;
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

#[derive(Clone, Copy, Debug)]
struct VerificationSummary {
    passed: bool,
    max_abs_error: f64,
    max_rel_error: f64,
}

impl VerificationSummary {
    fn to_json(self) -> Value {
        json!({
            "passed": self.passed,
            "oracle_id": "forge-nnis/smollm2-weighted-rmsnorm-f64-host-oracle-v1",
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_baseline(
    baseline: &F32DecoderKernels,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    input_host: &[f32],
    weight_host: &[f32],
    atol: f64,
    rtol: f64,
) -> Result<VerificationSummary, Box<dyn Error>> {
    baseline.weighted_rms_norm(stream, input, weight, output, ROWS, COLS, RMS_NORM_EPSILON)?;
    verify_output(stream, output, input_host, weight_host, atol, rtol)
}

#[allow(clippy::too_many_arguments)]
fn verify_candidate(
    candidate: &F32WeightedRmsNormCandidate,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    input_host: &[f32],
    weight_host: &[f32],
    atol: f64,
    rtol: f64,
) -> Result<VerificationSummary, Box<dyn Error>> {
    candidate.weighted_rms_norm(stream, input, weight, output, ROWS, COLS, RMS_NORM_EPSILON)?;
    verify_output(stream, output, input_host, weight_host, atol, rtol)
}

fn verify_output(
    stream: &Stream,
    output: &DeviceBuffer<f32>,
    input: &[f32],
    weight: &[f32],
    atol: f64,
    rtol: f64,
) -> Result<VerificationSummary, Box<dyn Error>> {
    let actual = output.to_vec(stream)?;
    Ok(verify_weighted_rmsnorm(
        input,
        weight,
        &actual,
        RMS_NORM_EPSILON,
        atol,
        rtol,
    ))
}

fn verify_weighted_rmsnorm(
    input: &[f32],
    weight: &[f32],
    actual: &[f32],
    epsilon: f32,
    atol: f64,
    rtol: f64,
) -> VerificationSummary {
    if input.len() != COLS || weight.len() != COLS || actual.len() != COLS {
        return VerificationSummary {
            passed: false,
            max_abs_error: f64::INFINITY,
            max_rel_error: f64::INFINITY,
        };
    }
    let sumsq: f64 = input
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum();
    let scale = 1.0_f64 / (sumsq / COLS as f64 + f64::from(epsilon)).sqrt();
    let mut passed = true;
    let mut max_abs_error = 0.0_f64;
    let mut max_rel_error = 0.0_f64;
    for index in 0..COLS {
        let expected = f64::from(input[index]) * scale * f64::from(weight[index]);
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

fn deterministic_weight(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| 0.75 + (index % 37) as f32 * (0.5 / 37.0))
        .collect()
}

fn positive_median(report: &BenchmarkReport) -> Result<f64, Box<dyn Error>> {
    let value = report.statistics.median_ms;
    if !value.is_finite() || value <= 0.0 {
        return Err("benchmark median_ms must be finite and positive".into());
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
    fn pair_order_alternates() {
        assert_eq!(PairOrder::for_round(0), PairOrder::BaselineThenCandidate);
        assert_eq!(PairOrder::for_round(1), PairOrder::CandidateThenBaseline);
        assert_eq!(PairOrder::for_round(2), PairOrder::BaselineThenCandidate);
    }

    #[test]
    fn weighted_oracle_accepts_known_values() {
        let input = vec![0.5_f32; COLS];
        let weight = vec![2.0_f32; COLS];
        let scale = 1.0_f64 / (0.25_f64 + f64::from(RMS_NORM_EPSILON)).sqrt();
        let actual = vec![(0.5_f64 * scale * 2.0) as f32; COLS];
        let verification =
            verify_weighted_rmsnorm(&input, &weight, &actual, RMS_NORM_EPSILON, 1.0e-6, 1.0e-6);
        assert!(verification.passed);
    }

    #[test]
    fn weighted_oracle_rejects_wrong_output() {
        let input = vec![0.5_f32; COLS];
        let weight = vec![1.0_f32; COLS];
        let actual = vec![0.0_f32; COLS];
        assert!(
            !verify_weighted_rmsnorm(&input, &weight, &actual, RMS_NORM_EPSILON, 1.0e-6, 1.0e-6,)
                .passed
        );
    }

    #[test]
    fn campaign_constants_match_qualified_smollm2_runtime_shape() {
        assert_eq!((ROWS, COLS), (1, 576));
        assert_eq!(BASELINE_BLOCK_SIZE, 256);
        assert_eq!(CANDIDATE_BLOCK_SIZE, 512);
        assert_eq!(RMS_NORM_EPSILON, 1.0e-5);
    }
}
