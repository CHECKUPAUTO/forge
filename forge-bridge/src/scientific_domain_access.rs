//! Capability-separated views for scientific external domains.
//!
//! The validated scientific manifest contains development, validation and
//! confirmatory source identities in one administrative record. Search/mutation
//! code should not receive that whole record. These projections expose only the
//! identities required by one phase and deliberately omit confirmatory sources.

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
        environment: external.environment.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DataBoundaryV1, ExternalDomainManifestV1, ObjectiveDirection,
        SCIENTIFIC_EXTERNAL_DOMAIN_SCHEMA_VERSION, EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION,
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
}
