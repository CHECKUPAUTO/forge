//! NNIS backend for Forge kernel-agent campaigns.
//!
//! The first qualified slice is intentionally narrow: one f32 AXPBY task with
//! a fixed ABI and an explicit versioned 1-D launch policy. Forge owns search;
//! NNIS owns native CUDA compilation/execution and benchmark evidence.
//! Verification is performed against a host oracle before benchmarking.

use forge_kernel_agent::{
    CompileEvidence, ContractError, KernelBackend, KernelCandidate, KernelLaunchPolicy,
    KernelSourceLanguage, KernelTask, MeasurementEvidence, VerificationEvidence,
    MEASUREMENT_EVIDENCE_SCHEMA_VERSION,
};
use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkMetadata};
use nnis_jit::{
    CompileOptions, CompiledCode, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig,
    Module,
};
use nnis_rt::{gpu_context, Context, DeviceBuffer, NnisError, Stream};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub const AXPBY_ENTRYPOINT: &str = "forge_axpby_f32";
pub const DEFAULT_BLOCK_SIZE: u32 = 256;
pub const DEFAULT_VERIFY_TRIALS: u32 = 3;

#[derive(Debug)]
pub struct NnisArtifact {
    code: Arc<CompiledCode>,
    elements: usize,
    block_size: u32,
}

impl NnisArtifact {
    pub fn artifact_id(&self) -> String {
        self.code.key().hex()
    }

    pub fn elements(&self) -> usize {
        self.elements
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }
}

struct AxpbyInvocation<'a> {
    left: &'a DeviceBuffer<f32>,
    right: &'a DeviceBuffer<f32>,
    output: &'a DeviceBuffer<f32>,
    alpha: f32,
    beta: f32,
    elements: usize,
    block_size: u32,
}

pub struct NnisAxpbyBackend {
    context: Arc<Context>,
    stream: Stream,
    compiler: JitCompiler,
    bench_config: BenchConfig,
    block_size: u32,
    verify_trials: u32,
}

impl NnisAxpbyBackend {
    pub fn first() -> Result<Self, NnisBackendError> {
        let context = gpu_context().ok_or(NnisBackendError::NoCudaDevice)?;
        let stream = Stream::new(&context)?;
        Ok(Self {
            context,
            stream,
            compiler: JitCompiler::new(),
            bench_config: BenchConfig::default(),
            block_size: DEFAULT_BLOCK_SIZE,
            verify_trials: DEFAULT_VERIFY_TRIALS,
        })
    }

    pub fn with_bench_config(mut self, config: BenchConfig) -> Result<Self, NnisBackendError> {
        config.validate()?;
        self.bench_config = config;
        Ok(self)
    }

    /// Set the adapter default used by legacy candidates without an explicit
    /// launch policy. Search candidates should prefer `KernelLaunchPolicy` so
    /// the launch shape participates in candidate identity.
    pub fn with_block_size(mut self, block_size: u32) -> Result<Self, NnisBackendError> {
        if block_size == 0 || block_size > self.context.props().max_threads_per_block {
            return Err(NnisBackendError::InvalidTask(format!(
                "block_size {block_size} exceeds device/kernel launch policy"
            )));
        }
        self.block_size = block_size;
        Ok(self)
    }

    pub fn with_verify_trials(mut self, trials: u32) -> Result<Self, NnisBackendError> {
        if trials == 0 {
            return Err(NnisBackendError::InvalidTask(
                "verification requires at least one trial".to_string(),
            ));
        }
        self.verify_trials = trials;
        Ok(self)
    }

    fn validate_task(task: &KernelTask) -> Result<usize, NnisBackendError> {
        task.validate().map_err(NnisBackendError::Contract)?;
        if task.operation != "axpby" {
            return Err(NnisBackendError::InvalidTask(format!(
                "forge-nnis v0.1 supports operation axpby, got {:?}",
                task.operation
            )));
        }
        if task.dimensions.len() != 1 {
            return Err(NnisBackendError::InvalidTask(
                "axpby task requires exactly one dimension: elements".to_string(),
            ));
        }
        let elements = task.dimensions.get("elements").copied().ok_or_else(|| {
            NnisBackendError::InvalidTask("missing elements dimension".to_string())
        })?;
        if elements == 0 || elements > i32::MAX as u64 {
            return Err(NnisBackendError::InvalidTask(format!(
                "elements must be in 1..={}, got {elements}",
                i32::MAX
            )));
        }
        if task.numerical.storage_dtype != "f32"
            || task.numerical.compute_dtype != "f32"
            || task.numerical.accumulator_dtype != "f32"
            || task.numerical.allow_tf32
        {
            return Err(NnisBackendError::InvalidTask(
                "initial NNIS AXPBY backend requires strict f32/f32/f32 with TF32 disabled"
                    .to_string(),
            ));
        }
        usize::try_from(elements).map_err(|_| {
            NnisBackendError::InvalidTask("elements does not fit host usize".to_string())
        })
    }

    fn resolve_block_size(&self, candidate: &KernelCandidate) -> Result<u32, NnisBackendError> {
        resolve_block_size(
            candidate.launch_policy,
            self.block_size,
            self.context.props().max_threads_per_block,
        )
    }

    fn load_kernel(&self, artifact: &NnisArtifact) -> Result<(Module, Kernel), NnisBackendError> {
        let module = Module::load(&self.context, &artifact.code)?;
        let kernel = module.get_function(AXPBY_ENTRYPOINT)?;
        Ok((module, kernel))
    }

    fn enqueue_axpby(
        &self,
        kernel: &Kernel,
        invocation: AxpbyInvocation<'_>,
    ) -> Result<(), NnisBackendError> {
        let config = LaunchConfig::for_num_elements(invocation.elements, invocation.block_size)?;
        let launch = KernelLaunch::new(kernel, &self.stream, config);
        let mut arguments = KernelArgs::with_capacity(6, 3);
        arguments
            .push_buffer(invocation.left)
            .push_buffer(invocation.right)
            .push_buffer(invocation.output)
            .push(invocation.alpha)
            .push(invocation.beta)
            .push(invocation.elements as i32);
        // SAFETY: the fixed AXPBY ABI is part of this backend contract. The
        // argument order and widths above exactly match `AXPBY_ENTRYPOINT`, and
        // all buffers/kernel/stream outlive synchronization by the caller.
        unsafe { launch.launch(&mut arguments) }?;
        Ok(())
    }
}

impl KernelBackend for NnisAxpbyBackend {
    type Artifact = NnisArtifact;
    type Error = NnisBackendError;

    fn compile(
        &self,
        task: &KernelTask,
        candidate: &KernelCandidate,
    ) -> Result<(Self::Artifact, CompileEvidence), Self::Error> {
        let elements = Self::validate_task(task)?;
        candidate.validate().map_err(NnisBackendError::Contract)?;
        if candidate.source_language != KernelSourceLanguage::CudaCpp {
            return Err(NnisBackendError::UnsupportedSourceLanguage);
        }
        let block_size = self.resolve_block_size(candidate)?;

        let options = CompileOptions::for_device(&self.context);
        let code = self.compiler.compile_ptx(&candidate.source, &options)?;
        let artifact_id = code.key().hex();
        let mut compile_options = Vec::with_capacity(options.extra_options().len() + 1);
        compile_options.push(format!("--gpu-architecture={}", options.architecture()));
        compile_options.extend(options.extra_options().iter().cloned());

        Ok((
            NnisArtifact {
                code,
                elements,
                block_size,
            },
            CompileEvidence {
                artifact_id,
                compiler_id: "nnis-jit/nvrtc".to_string(),
                compile_options,
            },
        ))
    }

    fn verify(
        &self,
        task: &KernelTask,
        artifact: &Self::Artifact,
    ) -> Result<VerificationEvidence, Self::Error> {
        let elements = Self::validate_task(task)?;
        if artifact.elements != elements {
            return Err(NnisBackendError::InvalidTask(
                "artifact/task element count mismatch".to_string(),
            ));
        }
        let (_module, kernel) = self.load_kernel(artifact)?;
        let alpha = 1.25_f32;
        let beta = -0.75_f32;
        let mut max_abs_error = 0.0_f64;
        let mut max_rel_error = 0.0_f64;
        let mut passed = true;

        for trial in 0..self.verify_trials {
            let (left_host, right_host) = verification_inputs(elements, trial);
            let left = DeviceBuffer::from_host(&self.context, &self.stream, &left_host)?;
            let right = DeviceBuffer::from_host(&self.context, &self.stream, &right_host)?;
            let output = DeviceBuffer::<f32>::new(&self.context, elements)?;

            self.enqueue_axpby(
                &kernel,
                AxpbyInvocation {
                    left: &left,
                    right: &right,
                    output: &output,
                    alpha,
                    beta,
                    elements,
                    block_size: artifact.block_size,
                },
            )?;
            self.stream.synchronize()?;
            let actual = output.to_vec(&self.stream)?;

            for ((&left_value, &right_value), &observed) in
                left_host.iter().zip(&right_host).zip(&actual)
            {
                let expected = alpha.mul_add(left_value, beta * right_value);
                if !observed.is_finite() {
                    passed = false;
                    continue;
                }
                let abs_error = f64::from((observed - expected).abs());
                let rel_error =
                    abs_error / f64::from(expected.abs()).max(f64::from(f32::MIN_POSITIVE));
                max_abs_error = max_abs_error.max(abs_error);
                max_rel_error = max_rel_error.max(rel_error);
                let allowed = task.numerical.atol + task.numerical.rtol * f64::from(expected.abs());
                if abs_error > allowed {
                    passed = false;
                }
            }
        }

        Ok(VerificationEvidence {
            schema_version: forge_kernel_agent::VERIFICATION_EVIDENCE_SCHEMA_VERSION,
            passed,
            oracle_id: "forge-nnis/axpby-host-oracle-v1".to_string(),
            max_abs_error: Some(max_abs_error),
            max_rel_error: Some(max_rel_error),
        })
    }

    fn measure(
        &self,
        task: &KernelTask,
        artifact: &Self::Artifact,
    ) -> Result<MeasurementEvidence, Self::Error> {
        let elements = Self::validate_task(task)?;
        if artifact.elements != elements {
            return Err(NnisBackendError::InvalidTask(
                "artifact/task element count mismatch".to_string(),
            ));
        }
        let (_module, kernel) = self.load_kernel(artifact)?;
        let alpha = 1.25_f32;
        let beta = -0.75_f32;
        let (left_host, right_host) = verification_inputs(elements, u32::MAX);
        let left = DeviceBuffer::from_host(&self.context, &self.stream, &left_host)?;
        let right = DeviceBuffer::from_host(&self.context, &self.stream, &right_host)?;
        let output = DeviceBuffer::<f32>::new(&self.context, elements)?;
        let config = LaunchConfig::for_num_elements(elements, artifact.block_size)?;
        let launch = KernelLaunch::new(&kernel, &self.stream, config);
        let mut arguments = KernelArgs::with_capacity(6, 3);
        arguments
            .push_buffer(&left)
            .push_buffer(&right)
            .push_buffer(&output)
            .push(alpha)
            .push(beta)
            .push(elements as i32);

        let bytes_per_iteration = (elements as u64)
            .checked_mul(3 * std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| NnisBackendError::InvalidTask("byte count overflow".to_string()))?;
        let case = BenchmarkCase::new("forge_nnis_axpby_f32", "f32")
            .with_dimension("elements", elements as u64)
            .with_dimension("block_size", u64::from(artifact.block_size))
            .with_work_items(elements as u64)
            .with_bytes_per_iteration(bytes_per_iteration);

        let report = benchmark_gpu(&self.context, &self.stream, case, self.bench_config, || {
            // SAFETY: fixed ABI and stable argument pack; benchmark_gpu
            // drains/synchronizes the stream around all measured launches.
            unsafe { launch.launch(&mut arguments) }
        })?;

        // Self-comparison intentionally exercises NNIS's fail-closed completeness
        // gate. A missing run context, GPU UUID, driver/NVRTC version, host
        // kernel, or Jetson power/clock evidence makes this measurement unusable
        // for candidate selection instead of silently manufacturing an identity.
        report
            .metadata
            .require_compatible_environment(&report.metadata)?;
        let environment_id = compatible_environment_id(&report.metadata)?;

        let mut metrics = BTreeMap::new();
        metrics.insert("block_size".to_string(), f64::from(artifact.block_size));
        metrics.insert("min_ms".to_string(), report.statistics.min_ms);
        metrics.insert("median_ms".to_string(), report.statistics.median_ms);
        metrics.insert("mean_ms".to_string(), report.statistics.mean_ms);
        metrics.insert("p95_ms".to_string(), report.statistics.p95_ms);
        metrics.insert("p99_ms".to_string(), report.statistics.p99_ms);
        metrics.insert("max_ms".to_string(), report.statistics.max_ms);
        metrics.insert("stddev_ms".to_string(), report.statistics.stddev_ms);
        metrics.insert(
            "warmup_iterations".to_string(),
            report.config.warmup_iterations as f64,
        );
        metrics.insert("iterations".to_string(), report.config.iterations as f64);
        if let Some(throughput) = &report.throughput {
            if let Some(value) = throughput.items_per_second {
                metrics.insert("items_per_second".to_string(), value);
            }
            if let Some(value) = throughput.gigabytes_per_second {
                metrics.insert("gigabytes_per_second".to_string(), value);
            }
        }

        Ok(MeasurementEvidence {
            schema_version: MEASUREMENT_EVIDENCE_SCHEMA_VERSION,
            environment_id,
            samples_ms: report.samples_ms,
            metrics,
        })
    }
}

fn resolve_block_size(
    launch_policy: Option<KernelLaunchPolicy>,
    default_block_size: u32,
    device_max_threads_per_block: u32,
) -> Result<u32, NnisBackendError> {
    let Some(policy) = launch_policy else {
        return Ok(default_block_size);
    };
    policy.validate().map_err(NnisBackendError::Contract)?;
    if policy.block[1] != 1 || policy.block[2] != 1 {
        return Err(NnisBackendError::InvalidTask(
            "initial AXPBY backend supports one-dimensional thread blocks only".to_string(),
        ));
    }
    if policy.dynamic_shared_memory_bytes != 0 {
        return Err(NnisBackendError::InvalidTask(
            "initial AXPBY backend does not use dynamic shared memory".to_string(),
        ));
    }
    let block_size = policy.block[0];
    if block_size > device_max_threads_per_block {
        return Err(NnisBackendError::InvalidTask(format!(
            "candidate block_size {block_size} exceeds device limit {device_max_threads_per_block}"
        )));
    }
    Ok(block_size)
}

fn compatible_environment_id(metadata: &BenchmarkMetadata) -> Result<String, NnisBackendError> {
    let identity = serde_json::json!({
        "host_arch": metadata.host_arch,
        "host_os": metadata.host_os,
        "gpu_ordinal": metadata.gpu_ordinal,
        "gpu_name": metadata.gpu_name,
        "gpu_uuid": metadata.gpu_uuid,
        "compute_capability_major": metadata.compute_capability_major,
        "compute_capability_minor": metadata.compute_capability_minor,
        "multiprocessor_count": metadata.multiprocessor_count,
        "driver_version": metadata.driver_version,
        "nvrtc_version": metadata.nvrtc_version,
        "environment_fingerprint": metadata.environment_fingerprint,
    });
    serde_json::to_string(&identity).map_err(NnisBackendError::Json)
}

fn verification_inputs(elements: usize, trial: u32) -> (Vec<f32>, Vec<f32>) {
    let mut state =
        0xD1B5_4A32_D192_ED03_u64 ^ u64::from(trial).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut left = Vec::with_capacity(elements);
    let mut right = Vec::with_capacity(elements);
    for _ in 0..elements {
        left.push(unit_f32(splitmix64(&mut state)) * 2.0 - 1.0);
        right.push(unit_f32(splitmix64(&mut state)) * 2.0 - 1.0);
    }
    (left, right)
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn unit_f32(value: u64) -> f32 {
    ((value >> 40) as u32) as f32 / ((1_u32 << 24) as f32)
}

#[derive(Debug)]
pub enum NnisBackendError {
    NoCudaDevice,
    UnsupportedSourceLanguage,
    InvalidTask(String),
    Contract(ContractError),
    Nnis(NnisError),
    Json(serde_json::Error),
}

impl Display for NnisBackendError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCudaDevice => formatter.write_str("no CUDA device is available"),
            Self::UnsupportedSourceLanguage => {
                formatter.write_str("initial NNIS backend accepts CUDA C++ source only")
            }
            Self::InvalidTask(message) => write!(formatter, "invalid NNIS kernel task: {message}"),
            Self::Contract(error) => Display::fmt(error, formatter),
            Self::Nnis(error) => Display::fmt(error, formatter),
            Self::Json(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for NnisBackendError {}

impl From<NnisError> for NnisBackendError {
    fn from(value: NnisError) -> Self {
        Self::Nnis(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_kernel_agent::NumericalContract;

    fn strict_task() -> KernelTask {
        KernelTask::new(
            "axpby-f32-n1024",
            "axpby",
            NumericalContract::f32_strict(1.0e-6, 1.0e-6),
        )
        .with_dimension("elements", 1024)
    }

    #[test]
    fn task_gate_accepts_only_initial_strict_f32_contract() {
        assert_eq!(
            NnisAxpbyBackend::validate_task(&strict_task()).unwrap(),
            1024
        );
        let mut tf32 = strict_task();
        tf32.numerical.allow_tf32 = true;
        assert!(NnisAxpbyBackend::validate_task(&tf32).is_err());

        let mut wrong_operation = strict_task();
        wrong_operation.operation = "softmax".to_string();
        assert!(NnisAxpbyBackend::validate_task(&wrong_operation).is_err());
    }

    #[test]
    fn launch_policy_is_explicit_and_fail_closed() {
        assert_eq!(resolve_block_size(None, 256, 1024).unwrap(), 256);
        assert_eq!(
            resolve_block_size(Some(KernelLaunchPolicy::block_x(128)), 256, 1024).unwrap(),
            128
        );
        assert!(resolve_block_size(Some(KernelLaunchPolicy::block_x(2048)), 256, 1024).is_err());

        let mut non_1d = KernelLaunchPolicy::block_x(128);
        non_1d.block[1] = 2;
        assert!(resolve_block_size(Some(non_1d), 256, 1024).is_err());

        let mut shared = KernelLaunchPolicy::block_x(128);
        shared.dynamic_shared_memory_bytes = 256;
        assert!(resolve_block_size(Some(shared), 256, 1024).is_err());
    }

    #[test]
    fn verification_inputs_are_deterministic_and_trial_specific() {
        let first = verification_inputs(64, 7);
        let replay = verification_inputs(64, 7);
        let other = verification_inputs(64, 8);
        assert_eq!(first, replay);
        assert_ne!(first, other);
        assert!(first
            .0
            .iter()
            .chain(&first.1)
            .all(|value| value.is_finite()));
    }

    #[test]
    fn fixed_entrypoint_contract_is_stable() {
        assert_eq!(AXPBY_ENTRYPOINT, "forge_axpby_f32");
        assert_eq!(DEFAULT_BLOCK_SIZE, 256);
    }
}
