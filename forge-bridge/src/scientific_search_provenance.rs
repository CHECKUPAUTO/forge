//! Search-stage provenance for scientific external domains.
//!
//! These records describe Forge's own execution-driven search disposition after
//! ordinary verification has already produced a measurement permit. They are
//! deliberately not scientific verdicts and contain no objective values,
//! Development sources, Validation sources, or confirmatory/final identities.

use serde::{Deserialize, Serialize};

use crate::{
    ScientificMeasurementPermitV1, UpstreamContractRefV1, VerificationBindingV1,
};

/// Forge-local disposition of one verified candidate in an ordinary search run.
///
/// `Rejected` and `Survivor` are search-engine states only. They must not be
/// interpreted as Beneficial/Harmful, confirmed/refuted, or any other scientific
/// conclusion owned by the upstream research programme.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScientificSearchDispositionV1 {
    /// Candidate did not survive the declared Forge search/selection stage.
    Rejected,
    /// Candidate survived the declared Forge search/selection stage.
    Survivor,
}

/// Provenance for a Forge-local search disposition after verify-before-measure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ScientificSearchProvenanceV1 {
    /// Exact upstream semantic/scientific contract inherited from the permit.
    pub upstream: UpstreamContractRefV1,
    /// Exact verifier/oracle binding inherited from the permit.
    pub verification: VerificationBindingV1,
    /// Candidate identity covered by the prerequisite verification.
    pub candidate_id: String,
    /// Executed prerequisite verification evidence identity.
    pub verification_evidence_id: String,
    /// Executed environment identity inherited from the measurement permit.
    pub environment_fingerprint: String,
    /// Forge-local search disposition; never a scientific verdict.
    pub disposition: ScientificSearchDispositionV1,
    /// Stable identity of the executed Forge search/selection evidence.
    pub search_evidence_id: String,
    /// Stable machine-oriented reason code from the Forge search stage.
    pub reason_code: String,
}

/// Fail-closed reasons why search provenance cannot be emitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScientificSearchProvenanceError {
    /// The permit does not identify the verified candidate.
    EmptyCandidateId,
    /// The permit does not identify the prerequisite verification evidence.
    EmptyVerificationEvidenceId,
    /// The search/selection execution does not have a stable evidence identity.
    EmptySearchEvidenceId,
    /// The search disposition does not have a stable machine-oriented reason.
    EmptyReasonCode,
}

impl std::fmt::Display for ScientificSearchProvenanceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ScientificSearchProvenanceError {}

/// Bind a Forge-local rejection/survival decision to its prerequisite
/// verify-before-measure capability.
///
/// This constructor accepts no metric/objective values and therefore cannot
/// silently turn Forge fitness into a scientific conclusion.
pub fn scientific_search_provenance(
    permit: &ScientificMeasurementPermitV1,
    disposition: ScientificSearchDispositionV1,
    search_evidence_id: impl Into<String>,
    reason_code: impl Into<String>,
) -> Result<ScientificSearchProvenanceV1, ScientificSearchProvenanceError> {
    if permit.candidate_id.trim().is_empty() {
        return Err(ScientificSearchProvenanceError::EmptyCandidateId);
    }
    if permit.verification_evidence_id.trim().is_empty() {
        return Err(ScientificSearchProvenanceError::EmptyVerificationEvidenceId);
    }

    let search_evidence_id = search_evidence_id.into();
    if search_evidence_id.trim().is_empty() {
        return Err(ScientificSearchProvenanceError::EmptySearchEvidenceId);
    }

    let reason_code = reason_code.into();
    if reason_code.trim().is_empty() {
        return Err(ScientificSearchProvenanceError::EmptyReasonCode);
    }

    Ok(ScientificSearchProvenanceV1 {
        upstream: permit.upstream.clone(),
        verification: permit.verification.clone(),
        candidate_id: permit.candidate_id.clone(),
        verification_evidence_id: permit.verification_evidence_id.clone(),
        environment_fingerprint: permit.environment_fingerprint.clone(),
        disposition,
        search_evidence_id,
        reason_code,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha256(fill: char) -> String {
        std::iter::repeat_n(fill, 64).collect()
    }

    fn permit() -> ScientificMeasurementPermitV1 {
        ScientificMeasurementPermitV1 {
            upstream: UpstreamContractRefV1 {
                repository: "Memorithm/TDI".to_string(),
                commit_id: "98b56dfb52acd7efa903b57fb9f0df0de65377a7".to_string(),
                contract_sha256: sha256('a'),
            },
            verification: VerificationBindingV1 {
                adapter_id: "validator-v1".to_string(),
                adapter_sha256: sha256('b'),
            },
            candidate_id: "candidate:7".to_string(),
            verification_evidence_id: "verification:7".to_string(),
            environment_fingerprint: "cpu=test;toolchain=test".to_string(),
        }
    }

    #[test]
    fn rejection_and_survivor_provenance_bind_exact_prerequisite_identity() {
        for disposition in [
            ScientificSearchDispositionV1::Rejected,
            ScientificSearchDispositionV1::Survivor,
        ] {
            let record = scientific_search_provenance(
                &permit(),
                disposition,
                "search-run:12:generation:3",
                "pareto-selection-v1",
            )
            .expect("valid search provenance");
            assert_eq!(record.candidate_id, "candidate:7");
            assert_eq!(record.verification_evidence_id, "verification:7");
            assert_eq!(record.disposition, disposition);
        }
    }

    #[test]
    fn serialized_provenance_contains_no_sources_fitness_or_scientific_verdict() {
        let record = scientific_search_provenance(
            &permit(),
            ScientificSearchDispositionV1::Survivor,
            "search-run:12:generation:3",
            "pareto-selection-v1",
        )
        .expect("valid search provenance");
        let json = serde_json::to_string(&record).expect("record serializes");

        for forbidden in [
            "generation_sources",
            "verification_sources",
            "final_holdout_sources",
            "fitness",
            "objectives",
            "scientific_verdict",
            "beneficial",
            "harmful",
        ] {
            assert!(!json.contains(forbidden), "leaked forbidden field: {forbidden}");
        }
    }

    #[test]
    fn missing_executed_search_identity_or_reason_fails_closed() {
        assert_eq!(
            scientific_search_provenance(
                &permit(),
                ScientificSearchDispositionV1::Rejected,
                " ",
                "compile-rejected",
            ),
            Err(ScientificSearchProvenanceError::EmptySearchEvidenceId)
        );
        assert_eq!(
            scientific_search_provenance(
                &permit(),
                ScientificSearchDispositionV1::Rejected,
                "search-run:12",
                " ",
            ),
            Err(ScientificSearchProvenanceError::EmptyReasonCode)
        );
    }
}
