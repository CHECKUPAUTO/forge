use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkMetadata, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::F32RmsNorm;
use nnis_rt::{gpu_context, DeviceBuffer, Stream};
use serde_json::{json, Value};
use std::error::Error;

const BASELINE_BLOCK_SIZE: u32 = 256;
const CANDIDATE_BLOCK_SIZE: u32 = 512;
const DEFAULT_SHAPES: &str = "1x2048,1x4096,1x5120,1x8192,8x2048,8x4096,8x5120,8x8192,32x2048,32x4096,32x5120,32x8192,128x2048,128x4096,128x5120,128x8192";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
    rows: usize,
    cols: usize,
}

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
    let shapes = parse_shapes(
        &std::env::var("FORGE_NNIS_RMSNORM_SHAPES").unwrap_or_else(|_| DEFAULT_SHAPES.into()),
    )?;
    let warmups = env_usize("FORGE_NNIS_RMSNORM_WARMUPS", 20)?;
    let iterations = env_usize("FORGE_NNIS_RMSNORM_ITERATIONS", 100)?;
    let rounds = env_usize("FORGE_NNIS_RMSNORM_MATRIX_ROUNDS", 4)?;
    let epsilon = env_f32("FORGE_NNIS_RMSNORM_EPSILON", 1.0e-6)?;
    let gamma = env_f32("FORGE_NNIS_RMSNORM_GAMMA", 1.0)?;
    let atol = env_f64("FORGE_NNIS_RMSNORM_ATOL", 5.0e-5)?;
    let rtol = env_f64("FORGE_NNIS_RMSNORM_RTOL", 5.0e-5)?;

    if rounds < 2 {
        return Err("FORGE_NNIS_RMSNORM_MATRIX_ROUNDS must be at least 2".into());
    }
    if !epsilon.is_finite() || epsilon <= 0.0 || !gamma.is_finite() {
        return Err("RMSNorm epsilon/gamma must be finite and epsilon must be positive".into());
    }
    if !atol.is_finite() || atol < 0.0 || !rtol.is_finite() || rtol < 0.0 {
        return Err("RMSNorm tolerances must be finite and non-negative".into());
    }

    let bench_config = BenchConfig::new(warmups, iterations);
    bench_config.validate()?;
    let context = gpu_context().ok_or("no CUDA device is available")?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let baseline = F32RmsNorm::load_with_block_size(&context, &compiler, BASELINE_BLOCK_SIZE)?;
    let candidate = F32RmsNorm::load_with_block_size(&context, &compiler, CANDIDATE_BLOCK_SIZE)?;

    let mut expected_metadata: Option<BenchmarkMetadata> = None;
    let mut shape_results = Vec::<Value>::with_capacity(shapes.len());
    let mut candidate_shape_wins = 0usize;
    let mut baseline_shape_wins = 0usize;
    let mut shape_ties = 0usize;

    for shape in &shapes {
        if !baseline.fused_available(shape.cols) {
            return Err(format!(
                "canonical block-{BASELINE_BLOCK_SIZE} RMSNorm fused path is unavailable for {}x{}",
                shape.rows, shape.cols
            )
            .into());
        }
        if !candidate.fused_available(shape.cols) {
            return Err(format!(
                "canonical block-{CANDIDATE_BLOCK_SIZE} RMSNorm fused path is unavailable for {}x{}",
                shape.rows, shape.cols
            )
            .into());
        }

        let result = evaluate_shape(
            *shape,
            &baseline,
            &candidate,
            &context,
            &stream,
            bench_config,
            rounds,
            epsilon,
            gamma,
            atol,
            rtol,
            &mut expected_metadata,
        )?;

        match result
            .median_candidate_ms
            .total_cmp(&result.median_baseline_ms)
        {
            std::cmp::Ordering::Less => candidate_shape_wins += 1,
            std::cmp::Ordering::Greater => baseline_shape_wins += 1,
            std::cmp::Ordering::Equal => shape_ties += 1,
        }
        shape_results.push(result.to_json());
    }

    let result = json!({
        "schema_version": 1,
        "campaign_kind": "forge_nnis_canonical_rmsnorm_shape_matrix_v1",
        "run_context_id": run_context,
        "baseline_block_size": BASELINE_BLOCK_SIZE,
        "candidate_block_size": CANDIDATE_BLOCK_SIZE,
        "shape_count": shapes.len(),
        "shapes": shapes.iter().map(|shape| json!({"rows": shape.rows, "cols": shape.cols})).collect::<Vec<_>>(),
        "rounds_per_shape": rounds,
        "warmup_iterations_per_observation": warmups,
        "measured_iterations_per_observation": iterations,
        "epsilon": epsilon,
        "gamma": gamma,
        "atol": atol,
        "rtol": rtol,
        "candidate_shape_wins": candidate_shape_wins,
        "baseline_shape_wins": baseline_shape_wins,
        "shape_ties": shape_ties,
        "claim_boundary": "paired canonical NNIS F32RmsNorm fused-path launch comparison across an explicit shape matrix only; no generated-kernel claim, no end-to-end NNIS speedup claim, and no automatic production-default promotion",
        "results": shape_results,
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[derive(Debug)]
struct ShapeResult {
    shape: Shape,
    baseline_verification: VerificationSummary,
    candidate_verification: VerificationSummary,
    candidate_round_wins: usize,
    baseline_round_wins: usize,
    round_ties: usize,
    median_baseline_ms: f64,
    median_candidate_ms: f64,
    median_paired_relative_improvement: f64,
    aggregate_microbenchmark_speedup_ratio: f64,
    observations: Vec<Value>,
}

impl ShapeResult {
    fn to_json(self) -> Value {
        json!({
            "rows": self.shape.rows,
            "cols": self.shape.cols,
            "baseline_verification": self.baseline_verification.to_json(),
            "candidate_verification": self.candidate_verification.to_json(),
            "candidate_round_wins": self.candidate_round_wins,
            "baseline_round_wins": self.baseline_round_wins,
            "round_ties": self.round_ties,
            "median_baseline_ms": self.median_baseline_ms,
            "median_candidate_ms": self.median_candidate_ms,
            "median_paired_relative_improvement": self.median_paired_relative_improvement,
            "aggregate_microbenchmark_speedup_ratio": self.aggregate_microbenchmark_speedup_ratio,
            "observations": self.observations,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_shape(
    shape: Shape,
    baseline: &F32RmsNorm,
    candidate: &F32RmsNorm,
    context: &std::sync::Arc<nnis_rt::Context>,
    stream: &Stream,
    bench_config: BenchConfig,
    rounds: usize,
    epsilon: f32,
    gamma: f32,
    atol: f64,
    rtol: f64,
    expected_metadata: &mut Option<BenchmarkMetadata>,
) -> Result<ShapeResult, Box<dyn Error>> {
    let elements = shape
        .rows
        .checked_mul(shape.cols)
        .ok_or("RMSNorm shape overflows usize")?;
    let bytes_per_iteration = (elements as u64)
        .checked_mul(2 * std::mem::size_of::<f32>() as u64)
        .ok_or("RMSNorm byte count overflows u64")?;
    let input_host = deterministic_input(elements, shape.rows as u64 ^ ((shape.cols as u64) << 32));
    let input = DeviceBuffer::from_host(context, stream, &input_host)?;
    let output = DeviceBuffer::<f32>::new(context, elements)?;

    let baseline_verification = verify_family(
        baseline,
        stream,
        &input,
        &output,
        &input_host,
        shape,
        epsilon,
        gamma,
        atol,
        rtol,
    )?;
    let candidate_verification = verify_family(
        candidate,
        stream,
        &input,
        &output,
        &input_host,
        shape,
        epsilon,
        gamma,
        atol,
        rtol,
    )?;
    if !baseline_verification.passed || !candidate_verification.passed {
        return Err(format!(
            "RMSNorm shape {}x{} requires both launch variants to verify",
            shape.rows, shape.cols
        )
        .into());
    }

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
                measure_family(
                    baseline,
                    BASELINE_BLOCK_SIZE,
                    context,
                    stream,
                    &input,
                    &output,
                    shape,
                    epsilon,
                    gamma,
                    bytes_per_iteration,
                    bench_config,
                )?,
                measure_family(
                    candidate,
                    CANDIDATE_BLOCK_SIZE,
                    context,
                    stream,
                    &input,
                    &output,
                    shape,
                    epsilon,
                    gamma,
                    bytes_per_iteration,
                    bench_config,
                )?,
            ),
            PairOrder::CandidateThenBaseline => {
                let candidate_report = measure_family(
                    candidate,
                    CANDIDATE_BLOCK_SIZE,
                    context,
                    stream,
                    &input,
                    &output,
                    shape,
                    epsilon,
                    gamma,
                    bytes_per_iteration,
                    bench_config,
                )?;
                let baseline_report = measure_family(
                    baseline,
                    BASELINE_BLOCK_SIZE,
                    context,
                    stream,
                    &input,
                    &output,
                    shape,
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
        require_same_environment(expected_metadata, &baseline_report.metadata)?;
        require_same_environment(expected_metadata, &candidate_report.metadata)?;

        let baseline_median = positive_median(&baseline_report)?;
        let candidate_median = positive_median(&candidate_report)?;
        let relative_improvement = (baseline_median - candidate_median) / baseline_median;
        match candidate_median.total_cmp(&baseline_median) {
            std::cmp::Ordering::Less => candidate_round_wins += 1,
            std::cmp::Ordering::Greater => baseline_round_wins += 1,
            std::cmp::Ordering::Equal => round_ties += 1,
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
    Ok(ShapeResult {
        shape,
        baseline_verification,
        candidate_verification,
        candidate_round_wins,
        baseline_round_wins,
        round_ties,
        median_baseline_ms,
        median_candidate_ms,
        median_paired_relative_improvement: median(&paired_relative_improvements)?,
        aggregate_microbenchmark_speedup_ratio: median_baseline_ms / median_candidate_ms,
        observations,
    })
}

#[allow(clippy::too_many_arguments)]
fn measure_family(
    rms_norm: &F32RmsNorm,
    block_size: u32,
    context: &std::sync::Arc<nnis_rt::Context>,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    shape: Shape,
    epsilon: f32,
    gamma: f32,
    bytes_per_iteration: u64,
    bench_config: BenchConfig,
) -> Result<BenchmarkReport, Box<dyn Error>> {
    let case = BenchmarkCase::new("nnis_f32_rmsnorm_fused", "f32")
        .with_dimension("rows", shape.rows as u64)
        .with_dimension("cols", shape.cols as u64)
        .with_dimension("block_size", u64::from(block_size))
        .with_work_items((shape.rows * shape.cols) as u64)
        .with_bytes_per_iteration(bytes_per_iteration);
    let report = benchmark_gpu(context, stream, case, bench_config, || {
        // SAFETY: buffers, stream, and kernel family outlive the benchmark;
        // benchmark_gpu synchronizes each measured launch.
        unsafe {
            rms_norm.enqueue_fused_rows(
                stream, input, output, shape.rows, shape.cols, epsilon, gamma,
            )
        }
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
    shape: Shape,
    epsilon: f32,
    gamma: f32,
    atol: f64,
    rtol: f64,
) -> Result<VerificationSummary, Box<dyn Error>> {
    rms_norm.fused_normalize_rows(
        stream, input, output, shape.rows, shape.cols, epsilon, gamma,
    )?;
    let actual = output.to_vec(stream)?;
    Ok(verify_rmsnorm(
        input_host, &actual, shape, epsilon, gamma, atol, rtol,
    ))
}

fn verify_rmsnorm(
    input: &[f32],
    actual: &[f32],
    shape: Shape,
    epsilon: f32,
    gamma: f32,
    atol: f64,
    rtol: f64,
) -> VerificationSummary {
    if input.len() != shape.rows.saturating_mul(shape.cols) || actual.len() != input.len() {
        return VerificationSummary {
            passed: false,
            max_abs_error: f64::INFINITY,
            max_rel_error: f64::INFINITY,
        };
    }

    let mut passed = true;
    let mut max_abs_error = 0.0_f64;
    let mut max_rel_error = 0.0_f64;
    for row in 0..shape.rows {
        let start = row * shape.cols;
        let end = start + shape.cols;
        let sumsq: f64 = input[start..end]
            .iter()
            .map(|value| {
                let value = f64::from(*value);
                value * value
            })
            .sum();
        let scale = f64::from(gamma) / (sumsq / shape.cols as f64 + f64::from(epsilon)).sqrt();
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

fn deterministic_input(elements: usize, seed: u64) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let mixed = (index as u64)
                .wrapping_add(seed)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bucket = ((mixed >> 32) % 4093) as i32 - 2046;
            (bucket as f32) * (1.0 / 1024.0) + ((index % 7) as f32 - 3.0) * 0.03125
        })
        .collect()
}

fn parse_shapes(value: &str) -> Result<Vec<Shape>, Box<dyn Error>> {
    let mut shapes = Vec::new();
    for raw in value.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let (rows, cols) = raw
            .split_once(['x', 'X'])
            .ok_or_else(|| format!("invalid RMSNorm shape '{raw}', expected ROWSxCOLS"))?;
        let rows = rows.trim().parse::<usize>()?;
        let cols = cols.trim().parse::<usize>()?;
        if rows == 0 || cols == 0 {
            return Err(format!("RMSNorm shape '{raw}' must be non-zero").into());
        }
        let shape = Shape { rows, cols };
        if shapes.contains(&shape) {
            return Err(format!("duplicate RMSNorm shape '{raw}'").into());
        }
        shapes.push(shape);
    }
    if shapes.is_empty() {
        return Err("RMSNorm shape matrix must contain at least one shape".into());
    }
    Ok(shapes)
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
    fn default_matrix_covers_rows_and_hidden_widths_without_duplicates() {
        let shapes = parse_shapes(DEFAULT_SHAPES).unwrap();
        assert_eq!(shapes.len(), 16);
        assert!(shapes.contains(&Shape {
            rows: 1,
            cols: 2048
        }));
        assert!(shapes.contains(&Shape {
            rows: 32,
            cols: 4096
        }));
        assert!(shapes.contains(&Shape {
            rows: 128,
            cols: 8192
        }));
    }

    #[test]
    fn shape_parser_is_case_insensitive_and_rejects_duplicates() {
        assert_eq!(
            parse_shapes("1x2048,8X4096").unwrap(),
            vec![
                Shape {
                    rows: 1,
                    cols: 2048
                },
                Shape {
                    rows: 8,
                    cols: 4096
                },
            ]
        );
        assert!(parse_shapes("1x2048,1x2048").is_err());
        assert!(parse_shapes("0x4096").is_err());
        assert!(parse_shapes("4096").is_err());
    }

    #[test]
    fn pair_order_alternates_deterministically() {
        assert_eq!(PairOrder::for_round(0), PairOrder::BaselineThenCandidate);
        assert_eq!(PairOrder::for_round(1), PairOrder::CandidateThenBaseline);
        assert_eq!(PairOrder::for_round(2), PairOrder::BaselineThenCandidate);
    }

    #[test]
    fn median_handles_even_and_odd_counts() {
        assert_eq!(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
    }

    #[test]
    fn oracle_accepts_exact_known_rmsnorm() {
        let shape = Shape { rows: 1, cols: 2 };
        let input = [3.0_f32, 4.0_f32];
        let epsilon = 1.0e-6_f32;
        let gamma = 1.0_f32;
        let scale = 1.0_f64 / ((25.0_f64 / 2.0) + f64::from(epsilon)).sqrt();
        let actual = [
            (f64::from(input[0]) * scale) as f32,
            (f64::from(input[1]) * scale) as f32,
        ];
        let verification = verify_rmsnorm(&input, &actual, shape, epsilon, gamma, 1.0e-6, 1.0e-6);
        assert!(verification.passed);
    }
}
