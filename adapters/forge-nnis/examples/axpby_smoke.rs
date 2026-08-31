use forge_kernel_agent::{
    evaluate_candidate, KernelCandidate, KernelSourceLanguage, KernelTask, NumericalContract,
};
use forge_nnis::NnisAxpbyBackend;
use nnis_bench::BenchConfig;
use serde_json::json;
use std::error::Error;

const WRONG_CANDIDATE: &str = r#"
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
        output[index] = 0.0f;
    }
}
"#;

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

fn main() -> Result<(), Box<dyn Error>> {
    let run_context = std::env::var("NNIS_BENCH_RUN_CONTEXT_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or("NNIS_BENCH_RUN_CONTEXT_ID must be set for fail-closed benchmark evidence")?;

    let elements = std::env::var("FORGE_NNIS_AXPBY_ELEMENTS")
        .ok()
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(1 << 20);
    let warmups = std::env::var("FORGE_NNIS_AXPBY_WARMUPS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(10);
    let iterations = std::env::var("FORGE_NNIS_AXPBY_ITERATIONS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(50);

    let task = KernelTask::new(
        format!("axpby-f32-n{elements}"),
        "axpby",
        NumericalContract::f32_strict(1.0e-6, 1.0e-6),
    )
    .with_dimension("elements", elements);

    let backend = NnisAxpbyBackend::first()?
        .with_bench_config(BenchConfig::new(warmups, iterations))?;

    let wrong = KernelCandidate::new(KernelSourceLanguage::CudaCpp, WRONG_CANDIDATE);
    let wrong_evaluation = evaluate_candidate(&backend, &task, &wrong)?;
    if wrong_evaluation.verification.passed || wrong_evaluation.measurement.is_some() {
        return Err("wrong candidate escaped the verification gate".into());
    }

    let correct = KernelCandidate::new(KernelSourceLanguage::CudaCpp, CORRECT_CANDIDATE);
    let correct_evaluation = evaluate_candidate(&backend, &task, &correct)?;
    if !correct_evaluation.is_selectable() {
        return Err("correct candidate did not produce selectable evidence".into());
    }

    let measurement = correct_evaluation
        .measurement
        .as_ref()
        .ok_or("correct candidate has no measurement evidence")?;
    let result = json!({
        "schema_version": 1,
        "run_context_id": run_context,
        "task_id": task.task_id,
        "operation": task.operation,
        "elements": elements,
        "wrong_candidate": {
            "candidate_id": wrong_evaluation.candidate_id,
            "verification_passed": wrong_evaluation.verification.passed,
            "measured": wrong_evaluation.measurement.is_some(),
        },
        "correct_candidate": {
            "candidate_id": correct_evaluation.candidate_id,
            "artifact_id": correct_evaluation.compile.artifact_id,
            "verification_passed": correct_evaluation.verification.passed,
            "max_abs_error": correct_evaluation.verification.max_abs_error,
            "max_rel_error": correct_evaluation.verification.max_rel_error,
            "environment_id": measurement.environment_id,
            "samples_ms": measurement.samples_ms,
            "metrics": measurement.metrics,
        }
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
