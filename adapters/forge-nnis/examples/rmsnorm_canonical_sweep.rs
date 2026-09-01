use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::F32RmsNorm;
use nnis_rt::{gpu_context, DeviceBuffer, Stream};
use serde_json::{json, Value};
use std::error::Error;

const BASELINE_BLOCK_SIZE: u32 = 256;
const BLOCK_SIZES: [u32; 5] = [256, 64, 128, 512, 1024];

fn main() -> Result<(), Box<dyn Error>> {
    let run_context = std::env::var("NNIS_BENCH_RUN_CONTEXT_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or("NNIS_BENCH_RUN_CONTEXT_ID must be set for fail-closed benchmark evidence")?;
    let rows = env_usize("FORGE_NNIS_RMSNORM_ROWS", 32)?;
    let cols = env_usize("FORGE_NNIS_RMSNORM_COLS", 4096)?;
    let warmups = env_usize("FORGE_NNIS_RMSNORM_WARMUPS", 20)?;
    let iterations = env_usize("FORGE_NNIS_RMSNORM_ITERATIONS", 100)?;
    let epsilon = env_f32("FORGE_NNIS_RMSNORM_EPSILON", 1.0e-6)?;
    let gamma = env_f32("FORGE_NNIS_RMSNORM_GAMMA", 1.0)?;
    let atol = env_f64("FORGE_NNIS_RMSNORM_ATOL", 5.0e-5)?;
    let rtol = env_f64("FORGE_NNIS_RMSNORM_RTOL", 5.0e-5)?;

    if rows == 0 || cols == 0 {
        return Err("RMSNorm rows and cols must be non-zero".into());
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

    let context = gpu_context().ok_or("no CUDA device is available")?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let bench_config = BenchConfig::new(warmups, iterations);
    bench_config.validate()?;
    let input_host = deterministic_input(elements);
    let input = DeviceBuffer::from_host(&context, &stream, &input_host)?;

    let mut baseline: Option<BenchmarkReport> = None;
    let mut results = Vec::<Value>::new();
    let mut best: Option<(u32, f64)> = None;

    for block_size in BLOCK_SIZES {
        let rms_norm = F32RmsNorm::load_with_block_size(&context, &compiler, block_size)?;
        if !rms_norm.fused_available(cols) {
            results.push(json!({
                "block_size": block_size,
                "status": "rejected",
                "reason": "canonical NNIS fused RMSNorm path is unavailable for this shape/block",
            }));
            continue;
        }

        let output = DeviceBuffer::<f32>::new(&context, elements)?;
        rms_norm.fused_normalize_rows(&stream, &input, &output, rows, cols, epsilon, gamma)?;
        let actual = output.to_vec(&stream)?;
        let verification =
            verify_rmsnorm(&input_host, &actual, rows, cols, epsilon, gamma, atol, rtol);
        if !verification.passed {
            results.push(json!({
                "block_size": block_size,
                "status": "rejected",
                "reason": "verification_failed",
                "verification": verification.to_json(),
            }));
            continue;
        }

        let case = BenchmarkCase::new("nnis_f32_rmsnorm_fused", "f32")
            .with_dimension("rows", rows as u64)
            .with_dimension("cols", cols as u64)
            .with_dimension("block_size", u64::from(block_size))
            .with_work_items(elements as u64)
            .with_bytes_per_iteration(bytes_per_iteration);
        let report = benchmark_gpu(&context, &stream, case, bench_config, || {
            // SAFETY: input/output/rms_norm/stream remain alive until every
            // benchmark launch is synchronized by benchmark_gpu.
            unsafe {
                rms_norm.enqueue_fused_rows(&stream, &input, &output, rows, cols, epsilon, gamma)
            }
        })?;
        report
            .metadata
            .require_compatible_environment(&report.metadata)?;
        if let Some(reference) = &baseline {
            reference
                .metadata
                .require_compatible_environment(&report.metadata)?;
        }
        if block_size == BASELINE_BLOCK_SIZE {
            baseline = Some(report.clone());
        }

        let median_ms = report.statistics.median_ms;
        if best
            .as_ref()
            .map(|(_, best_median)| median_ms < *best_median)
            .unwrap_or(true)
        {
            best = Some((block_size, median_ms));
        }
        results.push(json!({
            "block_size": block_size,
            "status": "measured",
            "verification": verification.to_json(),
            "report": report,
        }));
    }

    let baseline = baseline.ok_or("block-256 canonical RMSNorm baseline was not measured")?;
    for result in &results {
        if let Some(report_value) = result.get("report") {
            let report: BenchmarkReport = serde_json::from_value(report_value.clone())?;
            baseline
                .metadata
                .require_compatible_environment(&report.metadata)?;
        }
    }
    let (best_block_size, best_median_ms) = best.ok_or("no RMSNorm block was measurable")?;
    let baseline_median_ms = baseline.statistics.median_ms;
    let speedup = if best_median_ms < baseline_median_ms {
        Some(baseline_median_ms / best_median_ms)
    } else {
        None
    };

    let output = json!({
        "schema_version": 1,
        "campaign_kind": "forge_nnis_canonical_rmsnorm_fused_launch_sweep_v1",
        "run_context_id": run_context,
        "rows": rows,
        "cols": cols,
        "epsilon": epsilon,
        "gamma": gamma,
        "atol": atol,
        "rtol": rtol,
        "baseline_block_size": BASELINE_BLOCK_SIZE,
        "baseline_median_ms": baseline_median_ms,
        "best_block_size": best_block_size,
        "best_median_ms": best_median_ms,
        "microbenchmark_speedup_over_block_256": speedup,
        "claim_boundary": "canonical NNIS F32RmsNorm fused-path launch sweep only; no generated-kernel claim and no end-to-end NNIS speedup claim",
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
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
    if input.len() != rows * cols || actual.len() != input.len() {
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
    fn deterministic_input_is_stable() {
        assert_eq!(deterministic_input(64), deterministic_input(64));
    }

    #[test]
    fn oracle_accepts_exact_known_rmsnorm() {
        let input = vec![3.0_f32, 4.0_f32];
        let rms = ((9.0_f64 + 16.0) / 2.0).sqrt();
        let actual = vec![(3.0 / rms) as f32, (4.0 / rms) as f32];
        let summary = verify_rmsnorm(&input, &actual, 1, 2, 0.0, 1.0, 1.0e-6, 1.0e-6);
        assert!(summary.passed);
    }

    #[test]
    fn canonical_schedule_keeps_block_256_as_baseline() {
        assert_eq!(BLOCK_SIZES[0], BASELINE_BLOCK_SIZE);
        assert_eq!(BLOCK_SIZES, [256, 64, 128, 512, 1024]);
    }
}
