//! Backend-neutral contracts for execution-driven kernel search.
//!
//! The crate deliberately does not compile CUDA itself. It defines the stable
//! task/candidate/numerical/evidence boundary that adapters such as a future
//! `forge-nnis` backend can implement. Evaluation is fail-closed: compilation
//! is followed by independent verification, and measurement is only invoked
//! after verification passes.

use forge_core::candidate::{fnv1a, Candidate, CandidateId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const KERNEL_TASK_SCHEMA_VERSION: u32 = 1;
pub const NUMERICAL_CONTRACT_SCHEMA_VERSION: u32 = 1;
pub const VERIFICATION_EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const MEASUREMENT_EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelSourceLanguage {
    CudaCpp,
    Ptx,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelCandidate {
    pub source_language: KernelSourceLanguage,
    pub source: String,
    pub id: CandidateId,
}

impl KernelCandidate {
    pub fn new(source_language: KernelSourceLanguage, source: impl Into<String>) -> Self {
        let source = source.into();
        let identity = format!("{:?}\0{}", source_language, source);
        Self {
            source_language,
            source,
            id: fnv1a(&identity),
        }
    }
}

impl Candidate for KernelCandidate {
    fn id(&self) -> CandidateId {
        self.id
    }

    fn repr(&self) -> String {
        self.source.clone()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumericalContract {
    pub schema_version: u32,
    pub storage_dtype: String,
    pub compute_dtype: String,
    pub accumulator_dtype: String,
    pub allow_tf32: bool,
    pub atol: f64,
    pub rtol: f64,
}

impl NumericalContract {
    pub fn f32_strict(atol: f64, rtol: f64) -> Self {
        Self {
            schema_version: NUMERICAL_CONTRACT_SCHEMA_VERSION,
            storage_dtype: "f32".to_string(),
            compute_dtype: "f32".to_string(),
            accumulator_dtype: "f32".to_string(),
            allow_tf32: false,
            atol,
            rtol,
        }
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != NUMERICAL_CONTRACT_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                kind: "numerical_contract",
                version: self.schema_version,
            });
        }
        for (name, value) in [
            ("storage_dtype", self.storage_dtype.as_str()),
            ("compute_dtype", self.compute_dtype.as_str()),
            ("accumulator_dtype", self.accumulator_dtype.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ContractError::InvalidField(name));
            }
        }
        for (name, value) in [("atol", self.atol), ("rtol", self.rtol)] {
            if !value.is_finite() || value < 0.0 {
                return Err(ContractError::InvalidNumericField(name));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelTask {
    pub schema_version: u32,
    pub task_id: String,
    pub operation: String,
    pub dimensions: BTreeMap<String, u64>,
    pub numerical: NumericalContract,
}

impl KernelTask {
    pub fn new(
        task_id: impl Into<String>,
        operation: impl Into<String>,
        numerical: NumericalContract,
    ) -> Self {
        Self {
            schema_version: KERNEL_TASK_SCHEMA_VERSION,
            task_id: task_id.into(),
            operation: operation.into(),
            dimensions: BTreeMap::new(),
            numerical,
        }
    }

    pub fn with_dimension(mut self, name: impl Into<String>, value: u64) -> Self {
        self.dimensions.insert(name.into(), value);
        self
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != KERNEL_TASK_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                kind: "kernel_task",
                version: self.schema_version,
            });
        }
        if self.task_id.trim().is_empty() {
            return Err(ContractError::InvalidField("task_id"));
        }
        if self.operation.trim().is_empty() {
            return Err(ContractError::InvalidField("operation"));
        }
        self.numerical.validate()
    }

    /// Require exact semantic comparability before a candidate/baseline speed
    /// comparison. Numerical policy changes are intentionally separate tasks.
    pub fn require_comparable_to(&self, other: &Self) -> Result<(), ContractError> {
        self.validate()?;
        other.validate()?;
        if self.operation != other.operation {
            return Err(ContractError::NotComparable("operation"));
        }
        if self.dimensions != other.dimensions {
            return Err(ContractError::NotComparable("dimensions"));
        }
        if self.numerical != other.numerical {
            return Err(ContractError::NotComparable("numerical_contract"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompileEvidence {
    pub artifact_id: String,
    pub compiler_id: String,
    pub compile_options: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerificationEvidence {
    pub schema_version: u32,
    pub passed: bool,
    pub oracle_id: String,
    pub max_abs_error: Option<f64>,
    pub max_rel_error: Option<f64>,
}

impl VerificationEvidence {
    pub fn passed(oracle_id: impl Into<String>) -> Self {
        Self {
            schema_version: VERIFICATION_EVIDENCE_SCHEMA_VERSION,
            passed: true,
            oracle_id: oracle_id.into(),
            max_abs_error: None,
            max_rel_error: None,
        }
    }

    pub fn failed(oracle_id: impl Into<String>) -> Self {
        Self {
            schema_version: VERIFICATION_EVIDENCE_SCHEMA_VERSION,
            passed: false,
            oracle_id: oracle_id.into(),
            max_abs_error: None,
            max_rel_error: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasurementEvidence {
    pub schema_version: u32,
    pub environment_id: String,
    pub samples_ms: Vec<f64>,
    pub metrics: BTreeMap<String, f64>,
}

impl MeasurementEvidence {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != MEASUREMENT_EVIDENCE_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                kind: "measurement_evidence",
                version: self.schema_version,
            });
        }
        if self.environment_id.trim().is_empty() {
            return Err(ContractError::InvalidField("environment_id"));
        }
        if self.samples_ms.is_empty()
            || self
                .samples_ms
                .iter()
                .any(|sample| !sample.is_finite() || *sample < 0.0)
        {
            return Err(ContractError::InvalidNumericField("samples_ms"));
        }
        if self
            .metrics
            .values()
            .any(|value| !value.is_finite())
        {
            return Err(ContractError::InvalidNumericField("metrics"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelEvaluation {
    pub candidate_id: CandidateId,
    pub compile: CompileEvidence,
    pub verification: VerificationEvidence,
    pub measurement: Option<MeasurementEvidence>,
}

impl KernelEvaluation {
    pub fn is_selectable(&self) -> bool {
        self.verification.passed && self.measurement.is_some()
    }
}

/// Adapter boundary implemented by execution backends. Forge owns search;
/// backends own compilation/execution details and should call destination-owned
/// oracles where appropriate.
pub trait KernelBackend {
    type Artifact;
    type Error: Error + Send + Sync + 'static;

    fn compile(
        &self,
        task: &KernelTask,
        candidate: &KernelCandidate,
    ) -> Result<(Self::Artifact, CompileEvidence), Self::Error>;

    fn verify(
        &self,
        task: &KernelTask,
        artifact: &Self::Artifact,
    ) -> Result<VerificationEvidence, Self::Error>;

    fn measure(
        &self,
        task: &KernelTask,
        artifact: &Self::Artifact,
    ) -> Result<MeasurementEvidence, Self::Error>;
}

/// Execute the authoritative ordering. A failed verification returns normally
/// with no measurement and therefore cannot be selected by downstream search.
pub fn evaluate_candidate<B: KernelBackend>(
    backend: &B,
    task: &KernelTask,
    candidate: &KernelCandidate,
) -> Result<KernelEvaluation, EvaluationError<B::Error>> {
    task.validate().map_err(EvaluationError::Contract)?;
    let (artifact, compile) = backend
        .compile(task, candidate)
        .map_err(EvaluationError::Backend)?;
    let verification = backend
        .verify(task, &artifact)
        .map_err(EvaluationError::Backend)?;

    if verification.schema_version != VERIFICATION_EVIDENCE_SCHEMA_VERSION {
        return Err(EvaluationError::Contract(ContractError::UnsupportedSchema {
            kind: "verification_evidence",
            version: verification.schema_version,
        }));
    }
    if verification.oracle_id.trim().is_empty() {
        return Err(EvaluationError::Contract(ContractError::InvalidField(
            "oracle_id",
        )));
    }

    let measurement = if verification.passed {
        let evidence = backend
            .measure(task, &artifact)
            .map_err(EvaluationError::Backend)?;
        evidence.validate().map_err(EvaluationError::Contract)?;
        Some(evidence)
    } else {
        None
    };

    Ok(KernelEvaluation {
        candidate_id: candidate.id,
        compile,
        verification,
        measurement,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractError {
    UnsupportedSchema { kind: &'static str, version: u32 },
    InvalidField(&'static str),
    InvalidNumericField(&'static str),
    NotComparable(&'static str),
}

impl Display for ContractError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema { kind, version } => {
                write!(f, "unsupported {kind} schema version {version}")
            }
            Self::InvalidField(name) => write!(f, "invalid or missing field {name}"),
            Self::InvalidNumericField(name) => write!(f, "invalid numeric field {name}"),
            Self::NotComparable(name) => write!(f, "kernel tasks are not comparable at {name}"),
        }
    }
}

impl Error for ContractError {}

#[derive(Debug)]
pub enum EvaluationError<E> {
    Contract(ContractError),
    Backend(E),
}

impl<E: Display> Display for EvaluationError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contract(error) => Display::fmt(error, f),
            Self::Backend(error) => write!(f, "kernel backend error: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for EvaluationError<E> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::convert::Infallible;

    struct MockBackend {
        calls: RefCell<Vec<&'static str>>,
        verification_passes: bool,
    }

    impl KernelBackend for MockBackend {
        type Artifact = String;
        type Error = Infallible;

        fn compile(
            &self,
            _task: &KernelTask,
            candidate: &KernelCandidate,
        ) -> Result<(Self::Artifact, CompileEvidence), Self::Error> {
            self.calls.borrow_mut().push("compile");
            Ok((
                candidate.source.clone(),
                CompileEvidence {
                    artifact_id: "artifact-1".to_string(),
                    compiler_id: "mock-compiler".to_string(),
                    compile_options: vec!["-O3".to_string()],
                },
            ))
        }

        fn verify(
            &self,
            _task: &KernelTask,
            _artifact: &Self::Artifact,
        ) -> Result<VerificationEvidence, Self::Error> {
            self.calls.borrow_mut().push("verify");
            Ok(if self.verification_passes {
                VerificationEvidence::passed("independent-oracle-v1")
            } else {
                VerificationEvidence::failed("independent-oracle-v1")
            })
        }

        fn measure(
            &self,
            _task: &KernelTask,
            _artifact: &Self::Artifact,
        ) -> Result<MeasurementEvidence, Self::Error> {
            self.calls.borrow_mut().push("measure");
            Ok(MeasurementEvidence {
                schema_version: MEASUREMENT_EVIDENCE_SCHEMA_VERSION,
                environment_id: "gpu-env-1".to_string(),
                samples_ms: vec![1.0, 0.9, 1.1],
                metrics: BTreeMap::new(),
            })
        }
    }

    fn task() -> KernelTask {
        KernelTask::new(
            "axpby-f32-n1024",
            "axpby",
            NumericalContract::f32_strict(1.0e-6, 1.0e-6),
        )
        .with_dimension("elements", 1024)
    }

    #[test]
    fn verification_failure_prevents_measurement() {
        let backend = MockBackend {
            calls: RefCell::new(Vec::new()),
            verification_passes: false,
        };
        let candidate = KernelCandidate::new(
            KernelSourceLanguage::CudaCpp,
            "extern \"C\" __global__ void axpby() {}",
        );
        let result = evaluate_candidate(&backend, &task(), &candidate).unwrap();
        assert_eq!(*backend.calls.borrow(), vec!["compile", "verify"]);
        assert!(!result.is_selectable());
        assert!(result.measurement.is_none());
    }

    #[test]
    fn successful_evaluation_orders_compile_verify_measure() {
        let backend = MockBackend {
            calls: RefCell::new(Vec::new()),
            verification_passes: true,
        };
        let candidate = KernelCandidate::new(KernelSourceLanguage::CudaCpp, "kernel");
        let result = evaluate_candidate(&backend, &task(), &candidate).unwrap();
        assert_eq!(
            *backend.calls.borrow(),
            vec!["compile", "verify", "measure"]
        );
        assert!(result.is_selectable());
    }

    #[test]
    fn tf32_change_is_not_comparable_to_strict_f32() {
        let baseline = task();
        let mut candidate_task = task();
        candidate_task.numerical.allow_tf32 = true;
        assert_eq!(
            baseline.require_comparable_to(&candidate_task),
            Err(ContractError::NotComparable("numerical_contract"))
        );
    }

    #[test]
    fn tolerance_change_is_not_comparable() {
        let baseline = task();
        let mut candidate_task = task();
        candidate_task.numerical.atol = 1.0e-2;
        assert_eq!(
            baseline.require_comparable_to(&candidate_task),
            Err(ContractError::NotComparable("numerical_contract"))
        );
    }

    #[test]
    fn candidate_identity_changes_with_source_language() {
        let source = "same source";
        let cuda = KernelCandidate::new(KernelSourceLanguage::CudaCpp, source);
        let ptx = KernelCandidate::new(KernelSourceLanguage::Ptx, source);
        assert_ne!(cuda.id, ptx.id);
    }
}
