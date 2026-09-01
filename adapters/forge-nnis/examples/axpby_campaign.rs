use forge_kernel_agent::{
    KernelCandidate, KernelLaunchPolicy, KernelSourceLanguage, KernelTask, NumericalContract,
};
use forge_kernel_search::{
    run_kernel_campaign, KernelCampaignConfig, KernelCampaignReport, KernelMutator,
};
use forge_nnis::NnisAxpbyBackend;
use nnis_bench::BenchConfig;
use serde_json::json;
use std::error::Error;
use std::io;

const BASELINE_BLOCK_SIZE: u32 = 256;
const CANDIDATE_BLOCK_SIZES: [u32; 4] = [64, 128, 512, 1024];
const CAMPAIGN_SEED: u64 = 20_260_901;

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

struct LaunchPolicyMutator {
    block_sizes: &'static [u32],
    cursor: usize,
}

impl LaunchPolicyMutator {
    fn new(block_sizes: &'static [u32]) -> Self {
        Self {
            block_sizes,
            cursor: 0,
        }
    }
}

impl KernelMutator for LaunchPolicyMutator {
    type Error = io::Error;

    fn mutate(
        &mut self,
        _seed: u64,
        _generation: u32,
        _ordinal: u32,
        parent: &KernelCandidate,
    ) -> Result<KernelCandidate, Self::Error> {
        let block_size = self
            .block_sizes
            .get(self.cursor)
            .copied()
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "launch schedule exhausted"))?;
        self.cursor += 1;
        Ok(KernelCandidate::new(
            parent.source_language.clone(),
            parent.source.clone(),
        )
        .with_launch_policy(KernelLaunchPolicy::block_x(block_size)))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let run_context = std::env::var("NNIS_BENCH_RUN_CONTEXT_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or("NNIS_BENCH_RUN_CONTEXT_ID must be set for fail-closed benchmark evidence")?;

    let elements = env_u64("FORGE_NNIS_AXPBY_ELEMENTS", 1 << 20)?;
    let warmups = env_usize("FORGE_NNIS_AXPBY_WARMUPS", 10)?;
    let iterations = env_usize("FORGE_NNIS_AXPBY_ITERATIONS", 50)?;

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
    let mut mutator = LaunchPolicyMutator::new(&CANDIDATE_BLOCK_SIZES);
    let candidates_per_generation = u32::try_from(CANDIDATE_BLOCK_SIZES.len())?;

    let report = run_kernel_campaign(
        &backend,
        &mut mutator,
        &task,
        &baseline,
        KernelCampaignConfig::new(
            CAMPAIGN_SEED,
            1,
            candidates_per_generation,
            "median_ms",
        ),
    )?;

    emit_report(run_context, report)?;
    Ok(())
}

fn emit_report(run_context: String, report: KernelCampaignReport) -> Result<(), Box<dyn Error>> {
    let winner_attempt = report.winner_candidate_id.and_then(|winner_id| {
        report
            .attempts
            .iter()
            .find(|attempt| attempt.candidate_id == Some(winner_id))
    });
    let winner_block_size = winner_attempt
        .and_then(|attempt| attempt.candidate.as_ref())
        .and_then(|candidate| candidate.launch_policy)
        .map(|policy| policy.block[0]);
    let microbenchmark_speedup = report.winner_primary_metric.and_then(|winner_metric| {
        (winner_metric > 0.0).then_some(report.baseline_primary_metric / winner_metric)
    });
    let eligible_candidates = report.eligible_attempts().count();
    let rejected_candidates = report.rejected_attempts().count();

    let result = json!({
        "schema_version": 1,
        "campaign_kind": "forge_nnis_axpby_launch_policy_search_v1",
        "run_context_id": run_context,
        "baseline_block_size": BASELINE_BLOCK_SIZE,
        "candidate_block_sizes": CANDIDATE_BLOCK_SIZES,
        "winner_block_size": winner_block_size,
        "microbenchmark_speedup_over_baseline": microbenchmark_speedup,
        "eligible_candidates": eligible_candidates,
        "rejected_candidates": rejected_candidates,
        "claim_boundary": "isolated AXPBY kernel microbenchmark only; not an end-to-end NNIS speedup claim",
        "report": report,
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
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
    fn launch_schedule_preserves_source_and_changes_candidate_identity() {
        let parent = KernelCandidate::new(KernelSourceLanguage::CudaCpp, CORRECT_CANDIDATE)
            .with_launch_policy(KernelLaunchPolicy::block_x(BASELINE_BLOCK_SIZE));
        let mut mutator = LaunchPolicyMutator::new(&CANDIDATE_BLOCK_SIZES);
        let mut ids = Vec::new();

        for (ordinal, expected_block_size) in CANDIDATE_BLOCK_SIZES.iter().copied().enumerate() {
            let candidate = mutator
                .mutate(
                    CAMPAIGN_SEED,
                    0,
                    u32::try_from(ordinal).unwrap(),
                    &parent,
                )
                .unwrap();
            assert_eq!(candidate.source, parent.source);
            assert_eq!(
                candidate.launch_policy,
                Some(KernelLaunchPolicy::block_x(expected_block_size))
            );
            assert_ne!(candidate.id, parent.id);
            ids.push(candidate.id);
        }

        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), CANDIDATE_BLOCK_SIZES.len());
    }
}
