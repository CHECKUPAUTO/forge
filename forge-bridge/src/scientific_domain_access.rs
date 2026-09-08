//! Capability-separated views for scientific external domains.
//!
//! The validated scientific manifest contains development, validation and
//! confirmatory source identities in one administrative record. Search/mutation
//! code should not receive that whole record. These projections expose only the
//! identities required by one phase and deliberately omit confirmatory sources.
//!
//! Verification evidence must bind the exact upstream contract and verifier
//! identity before a measurement permit can be produced. The permit is a
//! structural verify-before-measure capability; it is not a scientific verdict
//! and never grants confirmatory/final-holdout access.

use serde::{Deserialize, Serialize};

use crate::{
    EnvironmentPolicyV1, ObjectiveSpecV1, ScientificExternalDomainError,
    ScientificExternalDomainManifestV1, UpstreamContractRefV1, VerificationBindingV1,
};

/// Data visible to candidate proposal/mutation for a scientific domain.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ScientificGenerationViewV1 {
    /// Frozen upstream semantic/scientific contract identity.
    pub upstream: UpstreamContractRefV1,
    /// Candidate dimensions explicitly declared search-safe.
    pub allowed_candidate_dimensions: Vec<String>,
    /// Development-only source identities available to generation/mutation.
    pub generation_sources: Vec<String>,
}

/// Data visible to independent ordinary verification for a scientific domain.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ScientificVerificationViewV1 {
    /// Frozen upstream semantic/scientific contract identity.
    pub upstream: UpstreamContractRefV1,
    /// Independent verification adapter binding.
    pub verification: VerificationBindingV1,
    /// Validation-only source identities available to verification.
    pub verification_sources: Vec<String>,
    /// Objective directions declared by the external-domain contract.
    pub objectives: Vec<ObjectiveSpecV1>,
    /// Environment requirements for evidence collection.
    pub environment: EnvironmentPolicyV1,
}

/// Executed ordinary-verification evidence supplied before scientific measurement.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ScientificVerificationEvidenceV1 {
    /// Exact upstream contract identity observed by the verifier.
    pub upstream: UpstreamContractRefV1,
    /// Exact verifier/oracle binding that produced the evidence.
    pub verification: VerificationBindingV1,
    /// Declared Validation source actually used for this verification.
    pub verification_source: String,
    /// Stable candidate identity verified by the adapter.
    pub candidate_id: String,
    /// Whether the prerequisite correctness/validity gate passed.
    pub passed: bool,
    /// Non-empty identity of the executed verification evidence.
    pub evidence_id: String,
    /// Executed environment identity when required by the domain policy.
    pub environment_fingerprint: String,
}

/// Capability proving ordinary verification passed before measurement.
///
/// This value intentionally carries no Development or confirmatory source
/// identities. Measurement code may require this capability but must still obey
/// the objective/environment contract in the verification view.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ScientificMeasurementPermitV1 {
    /// Exact upstream contract identity authorized for measurement.
    pub upstream: UpstreamContractRefV1,
    /// Exact verifier/oracle binding that passed.
    pub verification: VerificationBindingV1,
    /// Candidate identity covered by the verification evidence.
    pub candidate_id: String,
    /// Executed verification evidence identity establishing the prerequisite.
    pub verification_evidence_id: String,
    /// Environment fingerprint inherited from verification evidence.
    pub environment_fingerprint: String,
}

/// Fail-closed reasons why ordinary scientific verification cannot authorize measurement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScientificVerificationEvidenceError {
    /// The source scientific manifest failed validation before capability projection.
    InvalidManifest(ScientificExternalDomainError),
    /// The verification capability does not contain a usable upstream identity.
    MissingUpstreamIdentity,
    /// The verification capability does not contain a usable verifier/oracle identity.
    MissingVerificationBinding,
    /// Executed evidence names a different upstream contract.
    UpstreamIdentityMismatch,
    /// Executed evidence names a different verifier/oracle binding.
    VerificationBindingMismatch,
    /// Evidence used a source outside the Validation-only capability.
    UndeclaredVerificationSource { source: String },
    /// Evidence does not identify the candidate it verified.
    EmptyCandidateId,
    /// Verification executed but did not pass the prerequisite gate.
    VerificationFailed,
    /// Verification passed without a stable executed-evidence identity.
    EmptyEvidenceId,
    /// The domain requires an environment fingerprint but evidence omitted it.
    MissingEnvironmentFingerprint,
}

impl std::fmt::Display for ScientificVerificationEvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ScientificVerificationEvidenceError {}

impl From<ScientificExternalDomainError> for ScientificVerificationEvidenceError {
    fn from(error: ScientificExternalDomainError) -> Self {
        Self::InvalidManifest(error)
    }
}

/// Project a validated scientific manifest into the generation/mutation capability.
///
/// Validation and confirmatory source identities are intentionally absent.
pub fn scientific_generation_view(
    manifest: &ScientificExternalDomainManifestV1,
) -> Result<ScientificGenerationViewV1, ScientificExternalDomainError> {
    manifest.validate()?;
    let external = &manifest.external_domain;
    Ok(ScientificGenerationViewV1 {
        upstream: external.upstream.clone(),
        allowed_candidate_dimensions: external.allowed_candidate_dimensions.clone(),
        generation_sources: external.data_boundary.generation_sources.clone(),
    })
}

/// Project a validated scientific manifest into the independent verification capability.
///
/// Development and confirmatory source identities are intentionally absent.
pub fn scientific_verification_view(
    manifest: &ScientificExternalDomainManifestV1,
) -> Result<ScientificVerificationViewV1, ScientificExternalDomainError> {
    manifest.validate()?;
    let external = &manifest.external_domain;
    Ok(ScientificVerificationViewV1 {
        upstream: external.upstream.clone(),
        verification: external.verification.clone(),
        verification_sources: external.data_boundary.verification_sources.clone(),
        objectives: external.objectives.clone(),
        environment: external.environment,
    })
}

/// Validate executed verification evidence and mint a measurement capability.
///
/// Exact upstream and verifier/oracle identities must match the verification
/// capability. Only a declared Validation source may be referenced, the
/// prerequisite must have passed, and required environment provenance must be
/// present. No metric value participates in this gate.
pub fn scientific_measurement_permit(
    view: &ScientificVerificationViewV1,
    evidence: &ScientificVerificationEvidenceV1,
) -> Result<ScientificMeasurementPermitV1, ScientificVerificationEvidenceError> {
    if !usable_upstream_identity(&view.upstream) {
        return Err(ScientificVerificationEvidenceError::MissingUpstreamIdentity);
    }
    if !usable_verification_binding(&view.verification) {
        return Err(ScientificVerificationEvidenceError::MissingVerificationBinding);
    }
    if evidence.upstream != view.upstream {
        return Err(ScientificVerificationEvidenceError::UpstreamIdentityMismatch);
    }
    if evidence.verification != view.verification {
        return Err(ScientificVerificationEvidenceError::VerificationBindingMismatch);
    }
    if !view
        .verification_sources
        .iter()
        .any(|source| source == &evidence.verification_source)
    {
        return Err(
            ScientificVerificationEvidenceError::UndeclaredVerificationSource {
                source: evidence.verification_source.clone(),
            },
        );
    }
    if evidence.candidate_id.trim().is_empty() {
        return Err(ScientificVerificationEvidenceError::EmptyCandidateId);
    }
    if !evidence.passed {
        return Err(ScientificVerificationEvidenceError::VerificationFailed);
    }
    if evidence.evidence_id.trim().is_empty() {
        return Err(ScientificVerificationEvidenceError::EmptyEvidenceId);
    }
    if view.environment.fingerprint_required && evidence.environment_fingerprint.trim().is_empty() {
        return Err(ScientificVerificationEvidenceError::MissingEnvironmentFingerprint);
    }

    Ok(ScientificMeasurementPermitV1 {
        upstream: view.upstream.clone(),
        verification: view.verification.clone(),
        candidate_id: evidence.candidate_id.clone(),
        verification_evidence_id: evidence.evidence_id.clone(),
        environment_fingerprint: evidence.environment_fingerprint.clone(),
    })
}

fn usable_upstream_identity(upstream: &UpstreamContractRefV1) -> bool {
    !upstream.repository.trim().is_empty()
        && matches!(upstream.commit_id.len(), 40 | 64)
        && lower_hex(&upstream.commit_id)
        && upstream.contract_sha256.len() == 64
        && lower_hex(&upstream.contract_sha256)
}

fn usable_verification_binding(binding: &VerificationBindingV1) -> bool {
    !binding.adapter_id.trim().is_empty()
        && binding.adapter_sha256.len() == 64
        && lower_hex(&binding.adapter_sha256)
}

fn lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DataBoundaryV1, ExternalDomainManifestV1, ObjectiveDirection,
        EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION, SCIENTIFIC_EXTERNAL_DOMAIN_SCHEMA_VERSION,
    };

    fn sha256(fill: char) -> String {
        std::iter::repeat_n(fill, 64).collect()
    }

    fn manifest() -> ScientificExternalDomainManifestV1 {
        ScientificExternalDomainManifestV1 {
            schema_version: SCIENTIFIC_EXTERNAL_DOMAIN_SCHEMA_VERSION,
            external_domain: ExternalDomainManifestV1 {
                schema_version: EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION,
                domain_id: "scientific/example-v1".to_string(),
                upstream: UpstreamContractRefV1 {
                    repository: "Memorithm/TDI".to_string(),
                    commit_id: "98b56dfb52acd7efa903b57fb9f0df0de65377a7".to_string(),
                    contract_sha256: sha256('a'),
                },
                allowed_candidate_dimensions: vec!["schedule".to_string()],
                data_boundary: DataBoundaryV1 {
                    generation_sources: vec!["development-secret".to_string()],
                    verification_sources: vec!["validation-secret".to_string()],
                    final_holdout_sources: vec!["confirmatory-secret".to_string()],
                },
                verification: VerificationBindingV1 {
                    adapter_id: "validator-v1".to_string(),
                    adapter_sha256: sha256('b'),
                },
                objectives: vec![ObjectiveSpecV1 {
                    name: "correctness".to_string(),
                    direction: ObjectiveDirection::Maximize,
                }],
                environment: EnvironmentPolicyV1 {
                    fingerprint_required: true,
                    isolation_required: true,
                },
            },
        }
    }

    fn evidence(view: &ScientificVerificationViewV1) -> ScientificVerificationEvidenceV1 {
        ScientificVerificationEvidenceV1 {
            upstream: view.upstream.clone(),
            verification: view.verification.clone(),
            verification_source: "validation-secret".to_string(),
            candidate_id: "candidate:7".to_string(),
            passed: true,
            evidence_id: "verification:7".to_string(),
            environment_fingerprint: "cpu=test;toolchain=test".to_string(),
        }
    }

    #[test]
    fn generation_view_cannot_serialize_validation_or_confirmatory_sources() {
        let view = scientific_generation_view(&manifest()).expect("valid manifest");
        let json = serde_json::to_string(&view).expect("view serializes");
        assert!(json.contains("development-secret"));
        assert!(!json.contains("validation-secret"));
        assert!(!json.contains("confirmatory-secret"));
    }

    #[test]
    fn verification_view_cannot_serialize_development_or_confirmatory_sources() {
        let view = scientific_verification_view(&manifest()).expect("valid manifest");
        let json = serde_json::to_string(&view).expect("view serializes");
        assert!(json.contains("validation-secret"));
        assert!(!json.contains("development-secret"));
        assert!(!json.contains("confirmatory-secret"));
    }

    #[test]
    fn projections_fail_closed_when_manifest_is_invalid() {
        let mut invalid = manifest();
        invalid.external_domain.data_boundary.verification_sources =
            vec!["development-secret".to_string()];
        assert!(scientific_generation_view(&invalid).is_err());
        assert!(scientific_verification_view(&invalid).is_err());
    }

    #[test]
    fn exact_passed_verification_mints_measurement_permit() {
        let view = scientific_verification_view(&manifest()).expect("verification view");
        let evidence = evidence(&view);
        let permit = scientific_measurement_permit(&view, &evidence).expect("measurement permit");
        assert_eq!(permit.upstream, view.upstream);
        assert_eq!(permit.verification, view.verification);
        assert_eq!(permit.candidate_id, "candidate:7");
        assert_eq!(permit.verification_evidence_id, "verification:7");
    }

    #[test]
    fn mismatched_upstream_or_verifier_identity_fails_closed() {
        let view = scientific_verification_view(&manifest()).expect("verification view");
        let mut wrong_upstream = evidence(&view);
        wrong_upstream.upstream.contract_sha256 = sha256('c');
        assert_eq!(
            scientific_measurement_permit(&view, &wrong_upstream),
            Err(ScientificVerificationEvidenceError::UpstreamIdentityMismatch)
        );

        let mut wrong_verifier = evidence(&view);
        wrong_verifier.verification.adapter_sha256 = sha256('d');
        assert_eq!(
            scientific_measurement_permit(&view, &wrong_verifier),
            Err(ScientificVerificationEvidenceError::VerificationBindingMismatch)
        );
    }

    #[test]
    fn development_or_confirmatory_source_cannot_authorize_measurement() {
        let view = scientific_verification_view(&manifest()).expect("verification view");
        for forbidden in ["development-secret", "confirmatory-secret"] {
            let mut forbidden_evidence = evidence(&view);
            forbidden_evidence.verification_source = forbidden.to_string();
            assert_eq!(
                scientific_measurement_permit(&view, &forbidden_evidence),
                Err(
                    ScientificVerificationEvidenceError::UndeclaredVerificationSource {
                        source: forbidden.to_string(),
                    }
                )
            );
        }
    }

    #[test]
    fn failed_or_unidentified_verification_cannot_authorize_measurement() {
        let view = scientific_verification_view(&manifest()).expect("verification view");
        let mut failed = evidence(&view);
        failed.passed = false;
        assert_eq!(
            scientific_measurement_permit(&view, &failed),
            Err(ScientificVerificationEvidenceError::VerificationFailed)
        );

        let mut unidentified = evidence(&view);
        unidentified.evidence_id.clear();
        assert_eq!(
            scientific_measurement_permit(&view, &unidentified),
            Err(ScientificVerificationEvidenceError::EmptyEvidenceId)
        );
    }

    #[test]
    fn required_environment_fingerprint_cannot_be_omitted() {
        let view = scientific_verification_view(&manifest()).expect("verification view");
        let mut missing = evidence(&view);
        missing.environment_fingerprint.clear();
        assert_eq!(
            scientific_measurement_permit(&view, &missing),
            Err(ScientificVerificationEvidenceError::MissingEnvironmentFingerprint)
        );
    }

    #[test]
    fn forged_empty_verifier_capability_fails_closed() {
        let mut view = scientific_verification_view(&manifest()).expect("verification view");
        view.verification.adapter_id.clear();
        let evidence = evidence(&view);
        assert_eq!(
            scientific_measurement_permit(&view, &evidence),
            Err(ScientificVerificationEvidenceError::MissingVerificationBinding)
        );
    }
}
