//! Stronger split contract for scientific external Forge domains.
//!
//! [`ExternalDomainManifestV1`](crate::ExternalDomainManifestV1) is intentionally
//! generic because production domains may legitimately reuse ordinary verification
//! inputs during candidate construction. Scientific campaigns often require a
//! stricter development/validation boundary. This module provides that stronger,
//! opt-in contract without changing the existing wire semantics used by non-scientific
//! integrations.
//!
//! The wrapper validates the base external-domain manifest first, then requires
//! non-empty generation and verification source sets and rejects any source identity
//! shared by both. Final/confirmatory holdouts remain protected by the base manifest.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{ExternalDomainManifestError, ExternalDomainManifestV1};

/// Wire/schema version for [`ScientificExternalDomainManifestV1`].
pub const SCIENTIFIC_EXTERNAL_DOMAIN_SCHEMA_VERSION: u16 = 1;

/// A scientific Forge domain with an explicit development/validation split.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ScientificExternalDomainManifestV1 {
    /// Must equal [`SCIENTIFIC_EXTERNAL_DOMAIN_SCHEMA_VERSION`].
    pub schema_version: u16,
    /// Generic Forge external-domain contract.
    pub external_domain: ExternalDomainManifestV1,
}

/// Fail-closed validation failures specific to scientific split discipline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScientificExternalDomainError {
    /// Unsupported wrapper schema version.
    UnsupportedSchemaVersion(u16),
    /// The wrapped generic external-domain manifest is invalid.
    ExternalDomain(ExternalDomainManifestError),
    /// Candidate generation has no declared development source.
    NoGenerationSources,
    /// Independent ordinary verification has no declared validation source.
    NoVerificationSources,
    /// A source identity is visible to both generation and verification.
    DevelopmentValidationLeak { source: String },
    /// JSON serialization failed after validation.
    Json(String),
}

impl std::fmt::Display for ScientificExternalDomainError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ScientificExternalDomainError {}

impl From<ExternalDomainManifestError> for ScientificExternalDomainError {
    fn from(error: ExternalDomainManifestError) -> Self {
        Self::ExternalDomain(error)
    }
}

impl ScientificExternalDomainManifestV1 {
    /// Validate the generic external-domain contract and the stronger scientific
    /// development/validation separation.
    pub fn validate(&self) -> Result<(), ScientificExternalDomainError> {
        if self.schema_version != SCIENTIFIC_EXTERNAL_DOMAIN_SCHEMA_VERSION {
            return Err(ScientificExternalDomainError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }

        self.external_domain.validate()?;

        let boundary = &self.external_domain.data_boundary;
        if boundary.generation_sources.is_empty() {
            return Err(ScientificExternalDomainError::NoGenerationSources);
        }
        if boundary.verification_sources.is_empty() {
            return Err(ScientificExternalDomainError::NoVerificationSources);
        }

        let generation: BTreeSet<&str> = boundary
            .generation_sources
            .iter()
            .map(String::as_str)
            .collect();
        for source in &boundary.verification_sources {
            if generation.contains(source.as_str()) {
                return Err(ScientificExternalDomainError::DevelopmentValidationLeak {
                    source: source.clone(),
                });
            }
        }

        Ok(())
    }

    /// Serialize only a validated scientific-domain manifest.
    pub fn to_json(&self) -> Result<String, ScientificExternalDomainError> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| ScientificExternalDomainError::Json(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DataBoundaryV1, EnvironmentPolicyV1, ObjectiveDirection, ObjectiveSpecV1,
        UpstreamContractRefV1, VerificationBindingV1, EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION,
    };

    fn sha256(fill: char) -> String {
        std::iter::repeat_n(fill, 64).collect()
    }

    fn valid_manifest() -> ScientificExternalDomainManifestV1 {
        ScientificExternalDomainManifestV1 {
            schema_version: SCIENTIFIC_EXTERNAL_DOMAIN_SCHEMA_VERSION,
            external_domain: ExternalDomainManifestV1 {
                schema_version: EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION,
                domain_id: "tdi/assr-development-v1".to_string(),
                upstream: UpstreamContractRefV1 {
                    repository: "Memorithm/TDI".to_string(),
                    commit_id: "98b56dfb52acd7efa903b57fb9f0df0de65377a7".to_string(),
                    contract_sha256: sha256('a'),
                },
                allowed_candidate_dimensions: vec!["implementation_schedule".to_string()],
                data_boundary: DataBoundaryV1 {
                    generation_sources: vec!["development-v1".to_string()],
                    verification_sources: vec!["validation-v1".to_string()],
                    final_holdout_sources: vec!["confirmatory-v1".to_string()],
                },
                verification: VerificationBindingV1 {
                    adapter_id: "tdi-assr-reference-validator-v1".to_string(),
                    adapter_sha256: sha256('b'),
                },
                objectives: vec![ObjectiveSpecV1 {
                    name: "task_quality".to_string(),
                    direction: ObjectiveDirection::Maximize,
                }],
                environment: EnvironmentPolicyV1 {
                    fingerprint_required: true,
                    isolation_required: false,
                },
            },
        }
    }

    #[test]
    fn disjoint_development_validation_and_holdout_sources_are_accepted() {
        let manifest = valid_manifest();
        assert_eq!(manifest.validate(), Ok(()));
        let json = manifest.to_json().expect("valid scientific manifest serializes");
        assert!(json.contains("development-v1"));
        assert!(json.contains("validation-v1"));
        assert!(json.contains("confirmatory-v1"));
    }

    #[test]
    fn development_source_cannot_be_reused_for_validation() {
        let mut manifest = valid_manifest();
        manifest.external_domain.data_boundary.verification_sources =
            vec!["development-v1".to_string()];
        assert_eq!(
            manifest.validate(),
            Err(ScientificExternalDomainError::DevelopmentValidationLeak {
                source: "development-v1".to_string(),
            })
        );
    }

    #[test]
    fn scientific_campaign_requires_both_development_and_validation_sources() {
        let mut manifest = valid_manifest();
        manifest.external_domain.data_boundary.generation_sources.clear();
        assert_eq!(
            manifest.validate(),
            Err(ScientificExternalDomainError::NoGenerationSources)
        );

        let mut manifest = valid_manifest();
        manifest.external_domain.data_boundary.verification_sources.clear();
        assert_eq!(
            manifest.validate(),
            Err(ScientificExternalDomainError::NoVerificationSources)
        );
    }

    #[test]
    fn confirmatory_holdout_guard_is_inherited_from_base_manifest() {
        let mut manifest = valid_manifest();
        manifest
            .external_domain
            .data_boundary
            .generation_sources
            .push("confirmatory-v1".to_string());
        assert!(matches!(
            manifest.validate(),
            Err(ScientificExternalDomainError::ExternalDomain(
                ExternalDomainManifestError::HoldoutLeak { .. }
            ))
        ));
    }
}