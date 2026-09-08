//! Pont typé Forge ↔ autres briques de l'écosystème.
//!
//! Ce crate expose les types de `forge-core` et une façade fonctionnelle pour
//! lancer une campagne sans imposer de transport particulier. Un éventuel
//! service HTTP doit vivre dans un binaire dédié et n'est pas fourni ici.

use forge_core::{Candidate, Config, Domain, Score};
use serde::{Deserialize, Serialize};
use tracing::info;

pub mod binpack_demo;
pub mod candidate_envelope;
pub mod external_domain;
pub mod llm_ollama;
pub mod scientific_domain;
pub mod scientific_domain_access;
pub mod soup_campaign;
pub mod soup_posttrain;

pub use candidate_envelope::{
    CandidateEnvelopeError, CandidateEnvelopeV1, CANDIDATE_ENVELOPE_SCHEMA_VERSION,
};
pub use external_domain::{
    DataBoundaryV1, EnvironmentPolicyV1, ExternalDomainManifestError, ExternalDomainManifestV1,
    ObjectiveDirection, ObjectiveSpecV1, UpstreamContractRefV1, VerificationBindingV1,
    EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION,
};
pub use scientific_domain::{
    ScientificExternalDomainError, ScientificExternalDomainManifestV1,
    SCIENTIFIC_EXTERNAL_DOMAIN_SCHEMA_VERSION,
};
pub use scientific_domain_access::{
    scientific_generation_view, scientific_measurement_permit, scientific_verification_view,
    ScientificGenerationViewV1, ScientificMeasurementPermitV1, ScientificVerificationEvidenceError,
    ScientificVerificationEvidenceV1, ScientificVerificationViewV1,
};

pub type ForgeConfig = Config;

pub struct ForgeCampaign<D: Domain>
where
    D::Cand: Serialize + for<'a> Deserialize<'a>,
{
    pub config: ForgeConfig,
    pub domain: D,
}

impl<D: Domain> ForgeCampaign<D>
where
    D::Cand: Serialize + for<'a> Deserialize<'a>,
{
    pub fn new(config: ForgeConfig, domain: D) -> Self {
        Self { config, domain }
    }

    pub fn run(self) -> forge_core::Report<D::Cand> {
        info!(target: "forge-bridge", "lancement campagne domaine={}", self.domain.name());
        let engine = forge_core::Engine::new(self.domain, self.config);
        match engine.run() {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(target: "forge-bridge", "campagne en echec: {e}");
                forge_core::Report {
                    best: None,
                    final_baseline: None,
                    holdout_best: None,
                    holdout_baseline: None,
                    history: Vec::new(),
                    failure_diagnostics: Vec::new(),
                    final_front: Vec::new(),
                    final_front_holdout: Vec::new(),
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ScoreDto {
    pub objectives: Vec<f64>,
    pub valid: bool,
}

impl From<Score> for ScoreDto {
    fn from(s: Score) -> Self {
        ScoreDto {
            objectives: s.objectives,
            valid: s.valid,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct CandidateDto {
    pub id: u64,
    pub repr: String,
}

impl<T: Candidate> From<&T> for CandidateDto {
    fn from(c: &T) -> Self {
        CandidateDto {
            id: c.id(),
            repr: c.repr(),
        }
    }
}
