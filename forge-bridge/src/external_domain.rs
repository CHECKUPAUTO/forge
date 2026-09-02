//! Versioned, fail-closed contract for external Forge search domains.
//!
//! The manifest binds a Forge campaign to an upstream semantic/scientific
//! contract without importing that upstream repository's domain ownership into
//! Forge. It is deliberately generic: TDI, ADA, FLAT, NNIS or another producer
//! may provide the upstream contract, while Forge remains responsible only for
//! candidate search and executed evidence.
//!
//! Final-holdout identities are represented explicitly and validation rejects
//! any overlap with candidate-generation or ordinary verification sources.
//! This is a structural anti-leakage guard, not authorization to access a
//! confirmatory holdout.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Wire/schema version for [`ExternalDomainManifestV1`].
pub const EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION: u16 = 1;

/// Direction of one independently reported search objective.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    /// Lower values are preferable.
    Minimize,
    /// Higher values are preferable.
    Maximize,
}

/// One named, independently directed campaign objective.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ObjectiveSpecV1 {
    /// Stable objective name, for example `latency_ms` or `task_quality`.
    pub name: String,
    /// Optimization direction. Forge must not infer direction from the name.
    pub direction: ObjectiveDirection,
}

/// Identity of the repository-owned contract that defines admissible semantics.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct UpstreamContractRefV1 {
    /// Repository in `owner/name` form.
    pub repository: String,
    /// Exact upstream commit/object id. SHA-1 and SHA-256 Git object lengths are
    /// accepted so the bridge is not tied to one Git hash generation.
    pub commit_id: String,
    /// SHA-256 of the versioned semantic/scientific contract consumed by Forge.
    pub contract_sha256: String,
}

/// Explicit data/split boundary for candidate search.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataBoundaryV1 {
    /// Sources candidate generation or mutation may inspect.
    pub generation_sources: Vec<String>,
    /// Sources the ordinary independent verification harness may inspect.
    pub verification_sources: Vec<String>,
    /// Final/confirmatory sources that are forbidden to both paths above.
    pub final_holdout_sources: Vec<String>,
}

/// Identity of the independent verification adapter used by the domain.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VerificationBindingV1 {
    /// Stable adapter identifier.
    pub adapter_id: String,
    /// SHA-256 of the adapter contract/artifact identity supplied to the
    /// campaign. This does not replace destination-repository requalification.
    pub adapter_sha256: String,
}

/// Environment requirements that must be satisfied before campaign execution.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EnvironmentPolicyV1 {
    /// Whether a measurement environment fingerprint is mandatory.
    pub fingerprint_required: bool,
    /// Whether the domain requires an external isolation boundary for candidate
    /// execution. The manifest does not claim Forge itself supplies a sandbox.
    pub isolation_required: bool,
}

/// Generic external-domain contract consumed before Forge starts a campaign.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExternalDomainManifestV1 {
    /// Must equal [`EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION`].
    pub schema_version: u16,
    /// Stable domain/campaign-family identifier.
    pub domain_id: String,
    /// Versioned owner of semantic/scientific truth for this domain.
    pub upstream: UpstreamContractRefV1,
    /// Candidate dimensions that Forge is explicitly allowed to mutate/search.
    pub allowed_candidate_dimensions: Vec<String>,
    /// Split/data identities, including final sources that must remain outside
    /// generation and ordinary verification.
    pub data_boundary: DataBoundaryV1,
    /// Independent correctness/quality verification adapter identity.
    pub verification: VerificationBindingV1,
    /// Separately directed objectives; correctness is not encoded here as a
    /// soft objective and remains a prerequisite gate.
    pub objectives: Vec<ObjectiveSpecV1>,
    /// Environment/trust requirements for executed evidence.
    pub environment: EnvironmentPolicyV1,
}

/// Fail-closed validation errors for an external-domain manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExternalDomainManifestError {
    /// Unsupported schema version.
    UnsupportedSchemaVersion(u16),
    /// A required string field is empty or whitespace-only.
    EmptyField(&'static str),
    /// Repository is not in the expected `owner/name` form.
    InvalidRepository,
    /// Git object id is not lowercase hexadecimal of a supported length.
    InvalidCommitId,
    /// A SHA-256 field is not exactly 64 lowercase hexadecimal characters.
    InvalidSha256(&'static str),
    /// A list contains an empty entry.
    EmptyListEntry(&'static str),
    /// A list that must be unique contains a duplicate.
    DuplicateListEntry {
        /// List carrying the duplicate.
        field: &'static str,
        /// Duplicate value.
        value: String,
    },
    /// No searchable candidate dimension was declared.
    NoCandidateDimensions,
    /// No objective was declared.
    NoObjectives,
    /// A final/confirmatory source leaked into a generation or ordinary
    /// verification source set.
    HoldoutLeak {
        /// Conflicting source identity.
        source: String,
        /// Path in which the conflict was found.
        field: &'static str,
    },
    /// JSON serialization failed after validation.
    Json(String),
}

impl std::fmt::Display for ExternalDomainManifestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ExternalDomainManifestError {}

impl ExternalDomainManifestV1 {
    /// Validate identity, anti-leakage and objective invariants before a domain
    /// manifest is consumed by campaign code.
    pub fn validate(&self) -> Result<(), ExternalDomainManifestError> {
        if self.schema_version != EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION {
            return Err(ExternalDomainManifestError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        require_nonempty(&self.domain_id, "domain_id")?;
        validate_domain_id(&self.domain_id)?;
        validate_repository(&self.upstream.repository)?;
        validate_commit_id(&self.upstream.commit_id)?;
        validate_sha256(&self.upstream.contract_sha256, "upstream.contract_sha256")?;
        require_nonempty(&self.verification.adapter_id, "verification.adapter_id")?;
        validate_sha256(
            &self.verification.adapter_sha256,
            "verification.adapter_sha256",
        )?;

        if self.allowed_candidate_dimensions.is_empty() {
            return Err(ExternalDomainManifestError::NoCandidateDimensions);
        }
        validate_unique_list(
            &self.allowed_candidate_dimensions,
            "allowed_candidate_dimensions",
        )?;

        if self.objectives.is_empty() {
            return Err(ExternalDomainManifestError::NoObjectives);
        }
        let objective_names: Vec<String> = self
            .objectives
            .iter()
            .map(|objective| objective.name.clone())
            .collect();
        validate_unique_list(&objective_names, "objectives.name")?;

        validate_unique_list(
            &self.data_boundary.generation_sources,
            "data_boundary.generation_sources",
        )?;
        validate_unique_list(
            &self.data_boundary.verification_sources,
            "data_boundary.verification_sources",
        )?;
        validate_unique_list(
            &self.data_boundary.final_holdout_sources,
            "data_boundary.final_holdout_sources",
        )?;
        validate_holdout_disjointness(&self.data_boundary)?;

        Ok(())
    }

    /// Serialize a validated manifest to JSON. JSON is an interchange format;
    /// this function intentionally does not claim a canonical fingerprint.
    pub fn to_json(&self) -> Result<String, ExternalDomainManifestError> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| ExternalDomainManifestError::Json(error.to_string()))
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), ExternalDomainManifestError> {
    if value.trim().is_empty() {
        Err(ExternalDomainManifestError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_domain_id(value: &str) -> Result<(), ExternalDomainManifestError> {
    if value.bytes().all(|byte| {
        byte.is_ascii_lowercase()
            || byte.is_ascii_digit()
            || matches!(byte, b'.' | b'_' | b'-' | b'/')
    }) {
        Ok(())
    } else {
        Err(ExternalDomainManifestError::EmptyField("domain_id_format"))
    }
}

fn validate_repository(value: &str) -> Result<(), ExternalDomainManifestError> {
    let mut parts = value.split('/');
    let owner = parts.next().unwrap_or_default();
    let repository = parts.next().unwrap_or_default();
    if !owner.is_empty()
        && !repository.is_empty()
        && parts.next().is_none()
        && owner.bytes().all(valid_repository_byte)
        && repository.bytes().all(valid_repository_byte)
    {
        Ok(())
    } else {
        Err(ExternalDomainManifestError::InvalidRepository)
    }
}

fn valid_repository_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
}

fn validate_commit_id(value: &str) -> Result<(), ExternalDomainManifestError> {
    if matches!(value.len(), 40 | 64) && is_lower_hex(value) {
        Ok(())
    } else {
        Err(ExternalDomainManifestError::InvalidCommitId)
    }
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), ExternalDomainManifestError> {
    if value.len() == 64 && is_lower_hex(value) {
        Ok(())
    } else {
        Err(ExternalDomainManifestError::InvalidSha256(field))
    }
}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_unique_list(
    values: &[String],
    field: &'static str,
) -> Result<(), ExternalDomainManifestError> {
    let mut seen = BTreeSet::new();
    for value in values {
        if value.trim().is_empty() {
            return Err(ExternalDomainManifestError::EmptyListEntry(field));
        }
        if !seen.insert(value.as_str()) {
            return Err(ExternalDomainManifestError::DuplicateListEntry {
                field,
                value: value.clone(),
            });
        }
    }
    Ok(())
}

fn validate_holdout_disjointness(
    boundary: &DataBoundaryV1,
) -> Result<(), ExternalDomainManifestError> {
    let holdouts: BTreeSet<&str> = boundary
        .final_holdout_sources
        .iter()
        .map(String::as_str)
        .collect();
    for source in &boundary.generation_sources {
        if holdouts.contains(source.as_str()) {
            return Err(ExternalDomainManifestError::HoldoutLeak {
                source: source.clone(),
                field: "data_boundary.generation_sources",
            });
        }
    }
    for source in &boundary.verification_sources {
        if holdouts.contains(source.as_str()) {
            return Err(ExternalDomainManifestError::HoldoutLeak {
                source: source.clone(),
                field: "data_boundary.verification_sources",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha256(fill: char) -> String {
        std::iter::repeat_n(fill, 64).collect()
    }

    fn valid_manifest() -> ExternalDomainManifestV1 {
        ExternalDomainManifestV1 {
            schema_version: EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION,
            domain_id: "tdi/assr-reference-v1".to_string(),
            upstream: UpstreamContractRefV1 {
                repository: "Memorithm/TDI".to_string(),
                commit_id: "96deedc454f2bdff03b7ce39565e713f1992dde1".to_string(),
                contract_sha256: sha256('a'),
            },
            allowed_candidate_dimensions: vec![
                "memory_partition".to_string(),
                "implementation_schedule".to_string(),
            ],
            data_boundary: DataBoundaryV1 {
                generation_sources: vec!["train-v1".to_string()],
                verification_sources: vec!["validation-v1".to_string()],
                final_holdout_sources: vec!["final-holdout-v1".to_string()],
            },
            verification: VerificationBindingV1 {
                adapter_id: "tdi-assr-reference-validator-v1".to_string(),
                adapter_sha256: sha256('b'),
            },
            objectives: vec![
                ObjectiveSpecV1 {
                    name: "task_quality".to_string(),
                    direction: ObjectiveDirection::Maximize,
                },
                ObjectiveSpecV1 {
                    name: "memory_bits".to_string(),
                    direction: ObjectiveDirection::Minimize,
                },
            ],
            environment: EnvironmentPolicyV1 {
                fingerprint_required: true,
                isolation_required: false,
            },
        }
    }

    #[test]
    fn valid_external_domain_manifest_is_accepted() {
        let manifest = valid_manifest();
        assert_eq!(manifest.validate(), Ok(()));
        let json = manifest.to_json().expect("valid manifest should serialize");
        assert!(json.contains("\"schema_version\":1"));
        assert!(json.contains("final-holdout-v1"));
    }

    #[test]
    fn final_holdout_cannot_enter_generation_sources() {
        let mut manifest = valid_manifest();
        manifest
            .data_boundary
            .generation_sources
            .push("final-holdout-v1".to_string());
        assert_eq!(
            manifest.validate(),
            Err(ExternalDomainManifestError::HoldoutLeak {
                source: "final-holdout-v1".to_string(),
                field: "data_boundary.generation_sources",
            })
        );
    }

    #[test]
    fn final_holdout_cannot_enter_verification_sources() {
        let mut manifest = valid_manifest();
        manifest
            .data_boundary
            .verification_sources
            .push("final-holdout-v1".to_string());
        assert_eq!(
            manifest.validate(),
            Err(ExternalDomainManifestError::HoldoutLeak {
                source: "final-holdout-v1".to_string(),
                field: "data_boundary.verification_sources",
            })
        );
    }

    #[test]
    fn candidate_dimensions_must_be_unique() {
        let mut manifest = valid_manifest();
        manifest
            .allowed_candidate_dimensions
            .push("memory_partition".to_string());
        assert_eq!(
            manifest.validate(),
            Err(ExternalDomainManifestError::DuplicateListEntry {
                field: "allowed_candidate_dimensions",
                value: "memory_partition".to_string(),
            })
        );
    }

    #[test]
    fn hashes_and_commit_identity_are_fail_closed() {
        let mut manifest = valid_manifest();
        manifest.upstream.contract_sha256 = "ABC".to_string();
        assert_eq!(
            manifest.validate(),
            Err(ExternalDomainManifestError::InvalidSha256(
                "upstream.contract_sha256"
            ))
        );

        let mut manifest = valid_manifest();
        manifest.upstream.commit_id = "not-a-git-object".to_string();
        assert_eq!(
            manifest.validate(),
            Err(ExternalDomainManifestError::InvalidCommitId)
        );
    }

    #[test]
    fn search_requires_explicit_dimensions_and_objectives() {
        let mut manifest = valid_manifest();
        manifest.allowed_candidate_dimensions.clear();
        assert_eq!(
            manifest.validate(),
            Err(ExternalDomainManifestError::NoCandidateDimensions)
        );

        let mut manifest = valid_manifest();
        manifest.objectives.clear();
        assert_eq!(
            manifest.validate(),
            Err(ExternalDomainManifestError::NoObjectives)
        );
    }
}
