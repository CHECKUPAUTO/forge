//! Versioned executable campaign envelope for the SOUP post-training search domain.
//!
//! This module turns the already-published `soup_posttrain` typed domain into a
//! narrow process-friendly contract without moving SOUP semantics into Forge.
//! The evaluator remains authoritative for verification and measured metrics;
//! Forge only searches, applies the verify-before-measure gate and selects from
//! executed evidence.

use std::collections::BTreeMap;

use forge_core::{Candidate, Config, Engine, FailureDiagnostics, Score};
use serde::{Deserialize, Serialize};

use crate::soup_posttrain::{
    SoupEvaluator, SoupPostTrainDomain, SoupRecipeCandidate, SoupSearchSpace,
};
use crate::{ExternalDomainManifestV1, ObjectiveDirection};

pub const SOUP_CAMPAIGN_SCHEMA_VERSION: u16 = 1;
pub const SOUP_DOMAIN_SOURCE_MERGE: &str = "1385c71a541419f15a558a5e94bc8a4a60567a4a";
const MAX_GENERATIONS: u64 = 10_000;
const MAX_POPULATION: usize = 4_096;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SoupEngineConfigV1 {
    pub generations: u64,
    pub population: usize,
    pub survivors: usize,
    pub base_seed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SoupCampaignSpecV1 {
    pub schema_version: u16,
    pub external_domain: ExternalDomainManifestV1,
    pub dimensions: BTreeMap<String, Vec<String>>,
    pub baseline: BTreeMap<String, String>,
    pub engine: SoupEngineConfigV1,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SoupObjectiveValueV1 {
    pub name: String,
    pub direction: ObjectiveDirection,
    pub value: f64,
    pub forge_minimized_value: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SoupScoreV1 {
    pub valid: bool,
    pub objectives: Vec<SoupObjectiveValueV1>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SoupCandidateResultV1 {
    pub candidate_id: u64,
    pub values: BTreeMap<String, String>,
    pub score: SoupScoreV1,
    pub holdout_score: Option<SoupScoreV1>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SoupCampaignReportV1 {
    pub schema_version: u16,
    pub forge_domain_source_merge: String,
    pub domain_id: String,
    pub upstream_repository: String,
    pub upstream_commit_id: String,
    pub upstream_contract_sha256: String,
    pub verification_adapter_id: String,
    pub verification_adapter_sha256: String,
    pub engine: SoupEngineConfigV1,
    pub best: Option<SoupCandidateResultV1>,
    pub final_baseline: Option<SoupScoreV1>,
    pub holdout_best: Option<SoupScoreV1>,
    pub holdout_baseline: Option<SoupScoreV1>,
    pub history: Vec<f64>,
    pub failure_diagnostics: Vec<FailureDiagnostics>,
    pub final_front: Vec<SoupCandidateResultV1>,
}

impl SoupCampaignSpecV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SOUP_CAMPAIGN_SCHEMA_VERSION {
            return Err(format!(
                "unsupported SOUP campaign schema_version {}; expected {}",
                self.schema_version, SOUP_CAMPAIGN_SCHEMA_VERSION
            ));
        }
        self.external_domain
            .validate()
            .map_err(|error| format!("external domain manifest: {error}"))?;
        if self.engine.generations == 0 || self.engine.generations > MAX_GENERATIONS {
            return Err(format!(
                "engine.generations must be in 1..={MAX_GENERATIONS}"
            ));
        }
        if self.engine.population == 0 || self.engine.population > MAX_POPULATION {
            return Err(format!(
                "engine.population must be in 1..={MAX_POPULATION}"
            ));
        }
        if self.engine.survivors == 0 || self.engine.survivors > self.engine.population {
            return Err("engine.survivors must be in 1..=engine.population".to_string());
        }
        if self.dimensions.keys().any(|name| name.trim().is_empty()) {
            return Err("campaign dimension names must be non-empty".to_string());
        }
        Ok(())
    }
}

pub fn run_soup_campaign<E: SoupEvaluator>(
    spec: SoupCampaignSpecV1,
    evaluator: E,
    isolation_available: bool,
) -> Result<SoupCampaignReportV1, String> {
    spec.validate()?;

    let domain_id = spec.external_domain.domain_id.clone();
    let upstream_repository = spec.external_domain.upstream.repository.clone();
    let upstream_commit_id = spec.external_domain.upstream.commit_id.clone();
    let upstream_contract_sha256 = spec.external_domain.upstream.contract_sha256.clone();
    let verification_adapter_id = spec.external_domain.verification.adapter_id.clone();
    let verification_adapter_sha256 = spec.external_domain.verification.adapter_sha256.clone();
    let objective_specs = spec.external_domain.objectives.clone();
    let engine_config = spec.engine.clone();

    let search = SoupSearchSpace {
        dimensions: spec.dimensions,
        baseline: SoupRecipeCandidate {
            values: spec.baseline,
        },
    };
    let domain = SoupPostTrainDomain::new(
        spec.external_domain,
        search,
        evaluator,
        isolation_available,
    )
    .map_err(|error| error.to_string())?;

    let report = Engine::new(
        domain,
        Config {
            generations: engine_config.generations,
            population: engine_config.population,
            survivors: engine_config.survivors,
            base_seed: engine_config.base_seed,
            // The v1 process contract is deliberately local. Distributed worker
            // trust/placement remains a separate Forge/Hub contract.
            worker_addresses: None,
        },
    )
    .run()
    .map_err(|error| error.to_string())?;

    let final_front = report
        .final_front
        .into_iter()
        .zip(report.final_front_holdout)
        .map(|(individual, holdout)| SoupCandidateResultV1 {
            candidate_id: individual.cand.id(),
            values: individual.cand.values,
            score: score_to_wire(individual.score, &objective_specs),
            holdout_score: holdout.map(|score| score_to_wire(score, &objective_specs)),
        })
        .collect();

    let best = report.best.map(|individual| SoupCandidateResultV1 {
        candidate_id: individual.cand.id(),
        values: individual.cand.values,
        score: score_to_wire(individual.score, &objective_specs),
        // `Report::best` and `Report::holdout_best` are separate fields. Avoid
        // inventing an association that the engine does not expose.
        holdout_score: None,
    });

    Ok(SoupCampaignReportV1 {
        schema_version: SOUP_CAMPAIGN_SCHEMA_VERSION,
        forge_domain_source_merge: SOUP_DOMAIN_SOURCE_MERGE.to_string(),
        domain_id,
        upstream_repository,
        upstream_commit_id,
        upstream_contract_sha256,
        verification_adapter_id,
        verification_adapter_sha256,
        engine: engine_config,
        best,
        final_baseline: report
            .final_baseline
            .map(|score| score_to_wire(score, &objective_specs)),
        holdout_best: report
            .holdout_best
            .map(|score| score_to_wire(score, &objective_specs)),
        holdout_baseline: report
            .holdout_baseline
            .map(|score| score_to_wire(score, &objective_specs)),
        history: report.history,
        failure_diagnostics: report.failure_diagnostics,
        final_front,
    })
}

fn score_to_wire(score: Score, specs: &[crate::ObjectiveSpecV1]) -> SoupScoreV1 {
    if !score.valid {
        return SoupScoreV1 {
            valid: false,
            objectives: Vec::new(),
        };
    }

    let objectives = specs
        .iter()
        .zip(score.objectives)
        .map(|(spec, minimized)| SoupObjectiveValueV1 {
            name: spec.name.clone(),
            direction: spec.direction,
            value: match spec.direction {
                ObjectiveDirection::Minimize => minimized,
                ObjectiveDirection::Maximize => -minimized,
            },
            forge_minimized_value: minimized,
        })
        .collect();
    SoupScoreV1 {
        valid: true,
        objectives,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::soup_posttrain::{
        SoupEvaluatorRequest, SoupMeasurementEvidence, SoupVerificationEvidence,
        EVALUATOR_SCHEMA_VERSION,
    };
    use crate::{
        DataBoundaryV1, EnvironmentPolicyV1, ObjectiveSpecV1, UpstreamContractRefV1,
        VerificationBindingV1, EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION,
    };

    #[derive(Clone, Default)]
    struct FakeEvaluator {
        phases: Arc<Mutex<Vec<String>>>,
    }

    impl SoupEvaluator for FakeEvaluator {
        fn verify(
            &self,
            request: &SoupEvaluatorRequest,
        ) -> std::result::Result<SoupVerificationEvidence, String> {
            self.phases.lock().unwrap().push(request.phase.clone());
            Ok(SoupVerificationEvidence {
                schema_version: EVALUATOR_SCHEMA_VERSION,
                candidate_id: request.candidate_id,
                trial_seed: request.trial_seed,
                passed: true,
                evidence_id: format!("verify-{}", request.candidate_id),
                environment_fingerprint: "cpu=test".to_string(),
            })
        }

        fn measure(
            &self,
            request: &SoupEvaluatorRequest,
        ) -> std::result::Result<SoupMeasurementEvidence, String> {
            self.phases.lock().unwrap().push(request.phase.clone());
            let quality = if request.candidate.values["recipe.rank"] == "16" {
                0.9
            } else {
                0.8
            };
            Ok(SoupMeasurementEvidence {
                schema_version: EVALUATOR_SCHEMA_VERSION,
                candidate_id: request.candidate_id,
                trial_seed: request.trial_seed,
                evidence_id: format!("measure-{}", request.candidate_id),
                environment_fingerprint: "cpu=test".to_string(),
                metrics: BTreeMap::from([
                    ("quality".to_string(), quality),
                    ("wall_ms".to_string(), 10.0),
                ]),
            })
        }
    }

    fn spec() -> SoupCampaignSpecV1 {
        SoupCampaignSpecV1 {
            schema_version: SOUP_CAMPAIGN_SCHEMA_VERSION,
            external_domain: ExternalDomainManifestV1 {
                schema_version: EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION,
                domain_id: "soup/posttrain-v1".to_string(),
                upstream: UpstreamContractRefV1 {
                    repository: "MakazhanAlpamys/Soup".to_string(),
                    commit_id: "05b646523727925990530667e7012ede50bd30b2".to_string(),
                    contract_sha256:
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                            .to_string(),
                },
                allowed_candidate_dimensions: vec!["recipe.rank".to_string()],
                data_boundary: DataBoundaryV1 {
                    generation_sources: vec!["train".to_string()],
                    verification_sources: vec!["validation".to_string()],
                    final_holdout_sources: vec!["holdout".to_string()],
                },
                verification: VerificationBindingV1 {
                    adapter_id: "hub/soup-eval-v1".to_string(),
                    adapter_sha256:
                        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                            .to_string(),
                },
                objectives: vec![
                    ObjectiveSpecV1 {
                        name: "quality".to_string(),
                        direction: ObjectiveDirection::Maximize,
                    },
                    ObjectiveSpecV1 {
                        name: "wall_ms".to_string(),
                        direction: ObjectiveDirection::Minimize,
                    },
                ],
                environment: EnvironmentPolicyV1 {
                    fingerprint_required: true,
                    isolation_required: false,
                },
            },
            dimensions: BTreeMap::from([(
                "recipe.rank".to_string(),
                vec!["8".to_string(), "16".to_string()],
            )]),
            baseline: BTreeMap::from([("recipe.rank".to_string(), "8".to_string())]),
            engine: SoupEngineConfigV1 {
                generations: 2,
                population: 4,
                survivors: 2,
                base_seed: 7,
            },
        }
    }

    #[test]
    fn campaign_runs_verify_before_measure_and_restores_objective_direction() {
        let evaluator = FakeEvaluator::default();
        let phases = evaluator.phases.clone();
        let report = run_soup_campaign(spec(), evaluator, false).expect("campaign");
        assert_eq!(report.schema_version, SOUP_CAMPAIGN_SCHEMA_VERSION);
        assert_eq!(report.domain_id, "soup/posttrain-v1");
        assert!(!report.final_front.is_empty());
        let first = &report.final_front[0].score;
        assert!(first.valid);
        assert_eq!(first.objectives[0].name, "quality");
        assert!(first.objectives[0].value >= 0.0);
        assert!(first.objectives[0].forge_minimized_value <= 0.0);

        let phases = phases.lock().unwrap();
        assert!(!phases.is_empty());
        for measure_index in phases
            .iter()
            .enumerate()
            .filter_map(|(index, phase)| (phase == "measure").then_some(index))
        {
            assert!(
                phases[..measure_index].iter().any(|phase| phase == "verify"),
                "measurement must never be the first evaluator phase"
            );
        }
    }

    #[test]
    fn invalid_engine_bounds_fail_before_evaluation() {
        let mut invalid = spec();
        invalid.engine.survivors = invalid.engine.population + 1;
        let error = run_soup_campaign(invalid, FakeEvaluator::default(), false)
            .expect_err("invalid bounds must fail");
        assert!(error.contains("survivors"));
    }

    #[test]
    fn required_isolation_fails_closed() {
        let mut isolated = spec();
        isolated.external_domain.environment.isolation_required = true;
        let error = run_soup_campaign(isolated, FakeEvaluator::default(), false)
            .expect_err("missing isolation must fail");
        assert!(error.contains("isolation"));
    }
}
