//! Closed-loop search orchestration for Forge kernel-agent campaigns.
//!
//! This crate deliberately sits above `forge-kernel-agent`: contracts remain
//! small and backend-neutral, while campaign state, rejection retention and
//! winner selection live here. Selection is fail-closed against the baseline
//! environment and reuses Forge's minimization/Pareto `Score` semantics.

use forge_core::{CandidateId, Score};
use forge_kernel_agent::{
    evaluate_candidate, KernelBackend, KernelCandidate, KernelEvaluation, KernelTask,
    MeasurementEvidence,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const KERNEL_CAMPAIGN_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelCampaignConfig {
    pub schema_version: u32,
    pub seed: u64,
    pub generations: u32,
    pub candidates_per_generation: u32,
    pub primary_metric: String,
}

impl KernelCampaignConfig {
    pub fn new(
        seed: u64,
        generations: u32,
        candidates_per_generation: u32,
        primary_metric: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: KERNEL_CAMPAIGN_SCHEMA_VERSION,
            seed,
            generations,
            candidates_per_generation,
            primary_metric: primary_metric.into(),
        }
    }

    pub fn validate(&self) -> Result<(), CampaignError> {
        if self.schema_version != KERNEL_CAMPAIGN_SCHEMA_VERSION {
            return Err(CampaignError::UnsupportedSchema(self.schema_version));
        }
        if self.generations == 0 {
            return Err(CampaignError::InvalidConfig(
                "generations must be greater than zero".to_string(),
            ));
        }
        if self.candidates_per_generation == 0 {
            return Err(CampaignError::InvalidConfig(
                "candidates_per_generation must be greater than zero".to_string(),
            ));
        }
        if self.primary_metric.trim().is_empty() {
            return Err(CampaignError::InvalidConfig(
                "primary_metric must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

pub trait KernelMutator {
    type Error: Error + Send + Sync + 'static;

    fn mutate(
        &mut self,
        seed: u64,
        generation: u32,
        ordinal: u32,
        parent: &KernelCandidate,
    ) -> Result<KernelCandidate, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CandidateRejection {
    MutationFailed {
        message: String,
    },
    EvaluationFailed {
        message: String,
    },
    VerificationFailed,
    MissingMeasurement,
    IncompatibleEnvironment {
        baseline_environment_id: String,
        candidate_environment_id: String,
    },
    MissingPrimaryMetric {
        metric: String,
    },
    InvalidPrimaryMetric {
        metric: String,
        value: f64,
    },
    NotBetterThanBaseline {
        baseline_value: f64,
        candidate_value: f64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateAttempt {
    pub generation: u32,
    pub ordinal: u32,
    pub parent_candidate_id: CandidateId,
    pub candidate_id: Option<CandidateId>,
    pub candidate: Option<KernelCandidate>,
    pub evaluation: Option<KernelEvaluation>,
    pub rejection: Option<CandidateRejection>,
    pub eligible_for_selection: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelCampaignReport {
    pub schema_version: u32,
    pub config: KernelCampaignConfig,
    pub task: KernelTask,
    pub baseline_candidate_id: CandidateId,
    pub baseline_candidate: KernelCandidate,
    pub baseline_evaluation: KernelEvaluation,
    pub baseline_primary_metric: f64,
    pub baseline_environment_id: String,
    pub attempts: Vec<CandidateAttempt>,
    pub winner_candidate_id: Option<CandidateId>,
    pub winner_primary_metric: Option<f64>,
}

impl KernelCampaignReport {
    pub fn rejected_attempts(&self) -> impl Iterator<Item = &CandidateAttempt> {
        self.attempts
            .iter()
            .filter(|attempt| attempt.rejection.is_some())
    }

    pub fn eligible_attempts(&self) -> impl Iterator<Item = &CandidateAttempt> {
        self.attempts
            .iter()
            .filter(|attempt| attempt.eligible_for_selection)
    }
}

pub fn run_kernel_campaign<B, M>(
    backend: &B,
    mutator: &mut M,
    task: &KernelTask,
    baseline: &KernelCandidate,
    config: KernelCampaignConfig,
) -> Result<KernelCampaignReport, CampaignError>
where
    B: KernelBackend,
    M: KernelMutator,
{
    config.validate()?;
    task.validate()
        .map_err(|error| CampaignError::InvalidTask(error.to_string()))?;
    baseline
        .validate()
        .map_err(|error| CampaignError::InvalidBaseline(error.to_string()))?;

    let baseline_evaluation = evaluate_candidate(backend, task, baseline)
        .map_err(|error| CampaignError::BaselineEvaluation(error.to_string()))?;
    if !baseline_evaluation.verification.passed {
        return Err(CampaignError::BaselineRejected);
    }

    let baseline_measurement = baseline_evaluation
        .measurement
        .as_ref()
        .ok_or(CampaignError::BaselineMissingMeasurement)?;
    let baseline_primary_metric = metric_value(baseline_measurement, &config.primary_metric)
        .map_err(CampaignError::BaselineMetric)?;
    let baseline_environment_id = baseline_measurement.environment_id.clone();
    let baseline_score = Score::valid(vec![baseline_primary_metric]);

    let capacity = usize::try_from(config.generations)
        .unwrap_or(0)
        .saturating_mul(usize::try_from(config.candidates_per_generation).unwrap_or(0));
    let mut attempts = Vec::with_capacity(capacity);
    let mut parent = baseline.clone();
    let mut winner: Option<(KernelCandidate, f64)> = None;

    for generation in 0..config.generations {
        let mut generation_best: Option<(KernelCandidate, f64)> = None;

        for ordinal in 0..config.candidates_per_generation {
            let parent_candidate_id = parent.id;
            let candidate = match mutator.mutate(config.seed, generation, ordinal, &parent) {
                Ok(candidate) => candidate,
                Err(error) => {
                    attempts.push(CandidateAttempt {
                        generation,
                        ordinal,
                        parent_candidate_id,
                        candidate_id: None,
                        candidate: None,
                        evaluation: None,
                        rejection: Some(CandidateRejection::MutationFailed {
                            message: error.to_string(),
                        }),
                        eligible_for_selection: false,
                    });
                    continue;
                }
            };
            let candidate_id = candidate.id;

            let evaluation = match evaluate_candidate(backend, task, &candidate) {
                Ok(evaluation) => evaluation,
                Err(error) => {
                    attempts.push(CandidateAttempt {
                        generation,
                        ordinal,
                        parent_candidate_id,
                        candidate_id: Some(candidate_id),
                        candidate: Some(candidate),
                        evaluation: None,
                        rejection: Some(CandidateRejection::EvaluationFailed {
                            message: error.to_string(),
                        }),
                        eligible_for_selection: false,
                    });
                    continue;
                }
            };

            if !evaluation.verification.passed {
                attempts.push(CandidateAttempt {
                    generation,
                    ordinal,
                    parent_candidate_id,
                    candidate_id: Some(candidate_id),
                    candidate: Some(candidate),
                    evaluation: Some(evaluation),
                    rejection: Some(CandidateRejection::VerificationFailed),
                    eligible_for_selection: false,
                });
                continue;
            }

            let Some(measurement) = evaluation.measurement.as_ref() else {
                attempts.push(CandidateAttempt {
                    generation,
                    ordinal,
                    parent_candidate_id,
                    candidate_id: Some(candidate_id),
                    candidate: Some(candidate),
                    evaluation: Some(evaluation),
                    rejection: Some(CandidateRejection::MissingMeasurement),
                    eligible_for_selection: false,
                });
                continue;
            };

            let candidate_environment_id = measurement.environment_id.clone();
            if candidate_environment_id.as_str() != baseline_environment_id.as_str() {
                attempts.push(CandidateAttempt {
                    generation,
                    ordinal,
                    parent_candidate_id,
                    candidate_id: Some(candidate_id),
                    candidate: Some(candidate),
                    evaluation: Some(evaluation),
                    rejection: Some(CandidateRejection::IncompatibleEnvironment {
                        baseline_environment_id: baseline_environment_id.clone(),
                        candidate_environment_id,
                    }),
                    eligible_for_selection: false,
                });
                continue;
            }

            let candidate_metric = match metric_value(measurement, &config.primary_metric) {
                Ok(value) => value,
                Err(rejection) => {
                    attempts.push(CandidateAttempt {
                        generation,
                        ordinal,
                        parent_candidate_id,
                        candidate_id: Some(candidate_id),
                        candidate: Some(candidate),
                        evaluation: Some(evaluation),
                        rejection: Some(rejection),
                        eligible_for_selection: false,
                    });
                    continue;
                }
            };
            let candidate_score = Score::valid(vec![candidate_metric]);

            if !candidate_score.dominates(&baseline_score) {
                attempts.push(CandidateAttempt {
                    generation,
                    ordinal,
                    parent_candidate_id,
                    candidate_id: Some(candidate_id),
                    candidate: Some(candidate),
                    evaluation: Some(evaluation),
                    rejection: Some(CandidateRejection::NotBetterThanBaseline {
                        baseline_value: baseline_primary_metric,
                        candidate_value: candidate_metric,
                    }),
                    eligible_for_selection: false,
                });
                continue;
            }

            attempts.push(CandidateAttempt {
                generation,
                ordinal,
                parent_candidate_id,
                candidate_id: Some(candidate_id),
                candidate: Some(candidate.clone()),
                evaluation: Some(evaluation),
                rejection: None,
                eligible_for_selection: true,
            });

            if is_better_candidate(candidate_metric, candidate_id, generation_best.as_ref()) {
                generation_best = Some((candidate.clone(), candidate_metric));
            }
            if is_better_candidate(candidate_metric, candidate_id, winner.as_ref()) {
                winner = Some((candidate, candidate_metric));
            }
        }

        if let Some((candidate, _)) = generation_best {
            parent = candidate;
        }
    }

    Ok(KernelCampaignReport {
        schema_version: KERNEL_CAMPAIGN_SCHEMA_VERSION,
        config,
        task: task.clone(),
        baseline_candidate_id: baseline.id,
        baseline_candidate: baseline.clone(),
        baseline_evaluation,
        baseline_primary_metric,
        baseline_environment_id,
        attempts,
        winner_candidate_id: winner.as_ref().map(|(candidate, _)| candidate.id),
        winner_primary_metric: winner.as_ref().map(|(_, metric)| *metric),
    })
}

fn metric_value(
    measurement: &MeasurementEvidence,
    metric: &str,
) -> Result<f64, CandidateRejection> {
    let value = measurement.metrics.get(metric).copied().ok_or_else(|| {
        CandidateRejection::MissingPrimaryMetric {
            metric: metric.to_string(),
        }
    })?;

    if !value.is_finite() || value < 0.0 {
        return Err(CandidateRejection::InvalidPrimaryMetric {
            metric: metric.to_string(),
            value,
        });
    }
    Ok(value)
}

fn is_better_candidate(
    metric: f64,
    candidate_id: CandidateId,
    current: Option<&(KernelCandidate, f64)>,
) -> bool {
    match current {
        None => true,
        Some((current_candidate, current_metric)) => metric
            .total_cmp(current_metric)
            .then_with(|| candidate_id.cmp(&current_candidate.id))
            .is_lt(),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CampaignError {
    UnsupportedSchema(u32),
    InvalidConfig(String),
    InvalidTask(String),
    InvalidBaseline(String),
    BaselineEvaluation(String),
    BaselineRejected,
    BaselineMissingMeasurement,
    BaselineMetric(CandidateRejection),
}

impl Display for CampaignError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(version) => {
                write!(
                    formatter,
                    "unsupported kernel campaign schema version {version}"
                )
            }
            Self::InvalidConfig(message) => write!(formatter, "invalid campaign config: {message}"),
            Self::InvalidTask(message) => write!(formatter, "invalid campaign task: {message}"),
            Self::InvalidBaseline(message) => {
                write!(formatter, "invalid campaign baseline: {message}")
            }
            Self::BaselineEvaluation(message) => {
                write!(formatter, "baseline evaluation failed: {message}")
            }
            Self::BaselineRejected => write!(formatter, "baseline failed verification"),
            Self::BaselineMissingMeasurement => {
                write!(formatter, "baseline has no measurement evidence")
            }
            Self::BaselineMetric(rejection) => {
                write!(
                    formatter,
                    "baseline primary metric is unusable: {rejection:?}"
                )
            }
        }
    }
}

impl Error for CampaignError {}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_kernel_agent::{
        CompileEvidence, KernelLaunchPolicy, KernelSourceLanguage, NumericalContract,
        VerificationEvidence, MEASUREMENT_EVIDENCE_SCHEMA_VERSION,
    };
    use std::collections::BTreeMap;
    use std::convert::Infallible;
    use std::io;

    struct MockBackend;

    impl KernelBackend for MockBackend {
        type Artifact = String;
        type Error = Infallible;

        fn compile(
            &self,
            _task: &KernelTask,
            candidate: &KernelCandidate,
        ) -> Result<(Self::Artifact, CompileEvidence), Self::Error> {
            Ok((
                candidate.source.clone(),
                CompileEvidence {
                    artifact_id: candidate.id.to_string(),
                    compiler_id: "mock".to_string(),
                    compile_options: Vec::new(),
                },
            ))
        }

        fn verify(
            &self,
            _task: &KernelTask,
            artifact: &Self::Artifact,
        ) -> Result<VerificationEvidence, Self::Error> {
            Ok(if artifact == "bad" {
                VerificationEvidence::failed("oracle")
            } else {
                VerificationEvidence::passed("oracle")
            })
        }

        fn measure(
            &self,
            _task: &KernelTask,
            artifact: &Self::Artifact,
        ) -> Result<MeasurementEvidence, Self::Error> {
            let (environment_id, median_ms) = match artifact.as_str() {
                "baseline" => ("env-a", 10.0),
                "slow" => ("env-a", 12.0),
                "fast" => ("env-a", 8.0),
                "faster" => ("env-a", 6.0),
                "wrong-env" => ("env-b", 5.0),
                _ => ("env-a", 9.0),
            };
            let mut metrics = BTreeMap::new();
            metrics.insert("median_ms".to_string(), median_ms);
            Ok(MeasurementEvidence {
                schema_version: MEASUREMENT_EVIDENCE_SCHEMA_VERSION,
                environment_id: environment_id.to_string(),
                samples_ms: vec![median_ms],
                metrics,
            })
        }
    }

    struct ScriptedMutator {
        script: Vec<&'static str>,
        cursor: usize,
    }

    impl ScriptedMutator {
        fn new(script: Vec<&'static str>) -> Self {
            Self { script, cursor: 0 }
        }
    }

    impl KernelMutator for ScriptedMutator {
        type Error = io::Error;

        fn mutate(
            &mut self,
            _seed: u64,
            _generation: u32,
            _ordinal: u32,
            _parent: &KernelCandidate,
        ) -> Result<KernelCandidate, Self::Error> {
            let source =
                self.script.get(self.cursor).copied().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "script exhausted")
                })?;
            self.cursor += 1;
            Ok(KernelCandidate::new(KernelSourceLanguage::CudaCpp, source))
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
    fn campaign_retains_rejections_and_selects_best_compatible_candidate() {
        let baseline = KernelCandidate::new(KernelSourceLanguage::CudaCpp, "baseline");
        let mut mutator =
            ScriptedMutator::new(vec!["bad", "wrong-env", "slow", "fast", "faster", "fast"]);
        let report = run_kernel_campaign(
            &MockBackend,
            &mut mutator,
            &task(),
            &baseline,
            KernelCampaignConfig::new(7, 2, 3, "median_ms"),
        )
        .unwrap();

        assert_eq!(report.attempts.len(), 6);
        assert_eq!(report.rejected_attempts().count(), 3);
        assert_eq!(report.eligible_attempts().count(), 3);
        assert_eq!(report.winner_primary_metric, Some(6.0));
        assert_eq!(report.baseline_candidate, baseline);
        assert!(report
            .attempts
            .iter()
            .filter_map(|attempt| attempt.candidate.as_ref())
            .all(|candidate| candidate.id == candidate.id()));
        let winner_id = KernelCandidate::new(KernelSourceLanguage::CudaCpp, "faster").id;
        assert_eq!(report.winner_candidate_id, Some(winner_id));
        assert!(matches!(
            report.attempts[0].rejection,
            Some(CandidateRejection::VerificationFailed)
        ));
        assert!(matches!(
            report.attempts[1].rejection,
            Some(CandidateRejection::IncompatibleEnvironment { .. })
        ));
        assert!(matches!(
            report.attempts[2].rejection,
            Some(CandidateRejection::NotBetterThanBaseline { .. })
        ));
    }

    #[test]
    fn same_seed_and_script_are_deterministic() {
        fn run() -> KernelCampaignReport {
            let baseline = KernelCandidate::new(KernelSourceLanguage::CudaCpp, "baseline")
                .with_launch_policy(KernelLaunchPolicy::block_x(256));
            let mut mutator = ScriptedMutator::new(vec!["fast", "faster"]);
            run_kernel_campaign(
                &MockBackend,
                &mut mutator,
                &task(),
                &baseline,
                KernelCampaignConfig::new(1234, 1, 2, "median_ms"),
            )
            .unwrap()
        }

        let first = run();
        let second = run();
        assert_eq!(first, second);
    }

    #[test]
    fn mutation_failure_is_retained_in_campaign_evidence() {
        let baseline = KernelCandidate::new(KernelSourceLanguage::CudaCpp, "baseline");
        let mut mutator = ScriptedMutator::new(Vec::new());
        let report = run_kernel_campaign(
            &MockBackend,
            &mut mutator,
            &task(),
            &baseline,
            KernelCampaignConfig::new(1, 1, 1, "median_ms"),
        )
        .unwrap();

        assert_eq!(report.attempts.len(), 1);
        assert!(report.attempts[0].candidate.is_none());
        assert!(matches!(
            report.attempts[0].rejection,
            Some(CandidateRejection::MutationFailed { .. })
        ));
        assert!(report.winner_candidate_id.is_none());
    }
}