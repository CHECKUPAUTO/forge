//! Destination-owned requalification before a Forge survivor can be promoted.
//!
//! A Forge `Survivor` is only a search disposition. Promotion into another
//! repository/runtime requires fresh evidence owned by that destination. This
//! module binds destination evidence to both the exact survivor provenance and
//! a trusted destination qualification contract, and fails closed on mismatch.

use serde::{Deserialize, Serialize};

use crate::{ScientificSearchDispositionV1, ScientificSearchProvenanceV1};

/// Trusted destination-owned binding supplied by the promotion authority.
///
/// This value is intentionally separate from executed evidence so a serialized
/// evidence record cannot self-attest the repository, revision, adapter, or
/// adapter hash that is allowed to mint a promotion permit.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DestinationQualificationBindingV1 {
    /// Destination repository that owns the qualification gate.
    pub destination_repository: String,
    /// Exact destination revision containing the qualification gate/adapter.
    pub destination_commit_id: String,
    /// Stable destination-owned validator/qualification identity.
    pub qualification_adapter_id: String,
    /// Content hash of the trusted destination qualification contract/adapter.
    pub qualification_adapter_sha256: String,
}

/// Executed destination-owned qualification evidence for one Forge survivor.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DestinationRequalificationEvidenceV1 {
    /// Destination repository reported by the executed qualification.
    pub destination_repository: String,
    /// Exact destination revision reported by the executed qualification.
    pub destination_commit_id: String,
    /// Destination validator/qualification identity used for the execution.
    pub qualification_adapter_id: String,
    /// Content hash of the qualification contract/adapter used for execution.
    pub qualification_adapter_sha256: String,
    /// Exact candidate identity copied from Forge search provenance.
    pub candidate_id: String,
    /// Exact Forge search evidence identity being requalified.
    pub search_evidence_id: String,
    /// Environment fingerprint for the destination qualification execution.
    pub environment_fingerprint: String,
    /// Whether destination-owned qualification completed successfully.
    pub qualified: bool,
    /// Stable identity of the executed destination evidence record.
    pub requalification_evidence_id: String,
}

/// Capability minted only after destination-owned requalification succeeds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DestinationPromotionPermitV1 {
    pub destination_repository: String,
    pub destination_commit_id: String,
    pub candidate_id: String,
    pub search_evidence_id: String,
    pub requalification_evidence_id: String,
    pub environment_fingerprint: String,
}

/// Fail-closed reasons why a survivor cannot be promoted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DestinationRequalificationError {
    NotSurvivor,
    EmptyCandidateId,
    EmptySearchEvidenceId,
    EmptyDestinationRepository,
    EmptyDestinationCommitId,
    EmptyQualificationAdapterId,
    InvalidQualificationAdapterSha256,
    DestinationRepositoryMismatch,
    DestinationCommitMismatch,
    QualificationAdapterMismatch,
    QualificationAdapterSha256Mismatch,
    CandidateMismatch,
    SearchEvidenceMismatch,
    EmptyEnvironmentFingerprint,
    RequalificationFailed,
    EmptyRequalificationEvidenceId,
}

impl std::fmt::Display for DestinationRequalificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for DestinationRequalificationError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_binding(
    binding: &DestinationQualificationBindingV1,
) -> Result<(), DestinationRequalificationError> {
    if binding.destination_repository.trim().is_empty() {
        return Err(DestinationRequalificationError::EmptyDestinationRepository);
    }
    if binding.destination_commit_id.trim().is_empty() {
        return Err(DestinationRequalificationError::EmptyDestinationCommitId);
    }
    if binding.qualification_adapter_id.trim().is_empty() {
        return Err(DestinationRequalificationError::EmptyQualificationAdapterId);
    }
    if !valid_sha256(&binding.qualification_adapter_sha256) {
        return Err(DestinationRequalificationError::InvalidQualificationAdapterSha256);
    }
    Ok(())
}

/// Mint a destination promotion capability only when the exact Forge survivor
/// has been independently requalified by the trusted destination binding.
pub fn destination_promotion_permit(
    provenance: &ScientificSearchProvenanceV1,
    binding: &DestinationQualificationBindingV1,
    evidence: &DestinationRequalificationEvidenceV1,
) -> Result<DestinationPromotionPermitV1, DestinationRequalificationError> {
    if provenance.disposition != ScientificSearchDispositionV1::Survivor {
        return Err(DestinationRequalificationError::NotSurvivor);
    }
    if provenance.candidate_id.trim().is_empty() {
        return Err(DestinationRequalificationError::EmptyCandidateId);
    }
    if provenance.search_evidence_id.trim().is_empty() {
        return Err(DestinationRequalificationError::EmptySearchEvidenceId);
    }

    validate_binding(binding)?;

    if evidence.destination_repository != binding.destination_repository {
        return Err(DestinationRequalificationError::DestinationRepositoryMismatch);
    }
    if evidence.destination_commit_id != binding.destination_commit_id {
        return Err(DestinationRequalificationError::DestinationCommitMismatch);
    }
    if evidence.qualification_adapter_id != binding.qualification_adapter_id {
        return Err(DestinationRequalificationError::QualificationAdapterMismatch);
    }
    if evidence.qualification_adapter_sha256 != binding.qualification_adapter_sha256 {
        return Err(DestinationRequalificationError::QualificationAdapterSha256Mismatch);
    }
    if evidence.candidate_id != provenance.candidate_id {
        return Err(DestinationRequalificationError::CandidateMismatch);
    }
    if evidence.search_evidence_id != provenance.search_evidence_id {
        return Err(DestinationRequalificationError::SearchEvidenceMismatch);
    }
    if evidence.environment_fingerprint.trim().is_empty() {
        return Err(DestinationRequalificationError::EmptyEnvironmentFingerprint);
    }
    if !evidence.qualified {
        return Err(DestinationRequalificationError::RequalificationFailed);
    }
    if evidence.requalification_evidence_id.trim().is_empty() {
        return Err(DestinationRequalificationError::EmptyRequalificationEvidenceId);
    }

    Ok(DestinationPromotionPermitV1 {
        destination_repository: binding.destination_repository.clone(),
        destination_commit_id: binding.destination_commit_id.clone(),
        candidate_id: provenance.candidate_id.clone(),
        search_evidence_id: provenance.search_evidence_id.clone(),
        requalification_evidence_id: evidence.requalification_evidence_id.clone(),
        environment_fingerprint: evidence.environment_fingerprint.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{UpstreamContractRefV1, VerificationBindingV1};

    fn sha256(fill: char) -> String {
        std::iter::repeat_n(fill, 64).collect()
    }

    fn provenance(disposition: ScientificSearchDispositionV1) -> ScientificSearchProvenanceV1 {
        ScientificSearchProvenanceV1 {
            upstream: UpstreamContractRefV1 {
                repository: "Memorithm/TDI".to_string(),
                commit_id: "upstream-commit".to_string(),
                contract_sha256: sha256('a'),
            },
            verification: VerificationBindingV1 {
                adapter_id: "validator-v1".to_string(),
                adapter_sha256: sha256('b'),
            },
            candidate_id: "candidate:7".to_string(),
            verification_evidence_id: "verification:7".to_string(),
            environment_fingerprint: "forge-env".to_string(),
            disposition,
            search_evidence_id: "search:7".to_string(),
            reason_code: "pareto-survivor-v1".to_string(),
        }
    }

    fn binding() -> DestinationQualificationBindingV1 {
        DestinationQualificationBindingV1 {
            destination_repository: "Memorithm/NNIS".to_string(),
            destination_commit_id: "destination-commit".to_string(),
            qualification_adapter_id: "nnis-kernel-qualification-v1".to_string(),
            qualification_adapter_sha256: sha256('c'),
        }
    }

    fn evidence() -> DestinationRequalificationEvidenceV1 {
        DestinationRequalificationEvidenceV1 {
            destination_repository: "Memorithm/NNIS".to_string(),
            destination_commit_id: "destination-commit".to_string(),
            qualification_adapter_id: "nnis-kernel-qualification-v1".to_string(),
            qualification_adapter_sha256: sha256('c'),
            candidate_id: "candidate:7".to_string(),
            search_evidence_id: "search:7".to_string(),
            environment_fingerprint: "gpu=test;driver=test".to_string(),
            qualified: true,
            requalification_evidence_id: "nnis-qualification:7".to_string(),
        }
    }

    #[test]
    fn exact_survivor_and_trusted_destination_evidence_mint_permit() {
        let permit = destination_promotion_permit(
            &provenance(ScientificSearchDispositionV1::Survivor),
            &binding(),
            &evidence(),
        )
        .expect("matching trusted destination qualification should permit promotion");
        assert_eq!(permit.destination_repository, "Memorithm/NNIS");
        assert_eq!(permit.candidate_id, "candidate:7");
        assert_eq!(permit.search_evidence_id, "search:7");
    }

    #[test]
    fn rejected_candidate_cannot_be_promoted() {
        assert_eq!(
            destination_promotion_permit(
                &provenance(ScientificSearchDispositionV1::Rejected),
                &binding(),
                &evidence(),
            ),
            Err(DestinationRequalificationError::NotSurvivor)
        );
    }

    #[test]
    fn destination_must_requalify_exact_search_survivor() {
        let survivor = provenance(ScientificSearchDispositionV1::Survivor);

        let mut wrong_candidate = evidence();
        wrong_candidate.candidate_id = "candidate:8".to_string();
        assert_eq!(
            destination_promotion_permit(&survivor, &binding(), &wrong_candidate),
            Err(DestinationRequalificationError::CandidateMismatch)
        );

        let mut wrong_search = evidence();
        wrong_search.search_evidence_id = "search:8".to_string();
        assert_eq!(
            destination_promotion_permit(&survivor, &binding(), &wrong_search),
            Err(DestinationRequalificationError::SearchEvidenceMismatch)
        );
    }

    #[test]
    fn untrusted_destination_identity_cannot_self_attest() {
        let survivor = provenance(ScientificSearchDispositionV1::Survivor);
        let mut self_attested = evidence();
        self_attested.destination_repository = "attacker/repo".to_string();
        assert_eq!(
            destination_promotion_permit(&survivor, &binding(), &self_attested),
            Err(DestinationRequalificationError::DestinationRepositoryMismatch)
        );

        let mut wrong_hash = evidence();
        wrong_hash.qualification_adapter_sha256 = sha256('d');
        assert_eq!(
            destination_promotion_permit(&survivor, &binding(), &wrong_hash),
            Err(DestinationRequalificationError::QualificationAdapterSha256Mismatch)
        );
    }

    #[test]
    fn empty_survivor_identities_are_rejected() {
        let mut empty_candidate = provenance(ScientificSearchDispositionV1::Survivor);
        empty_candidate.candidate_id.clear();
        let mut candidate_evidence = evidence();
        candidate_evidence.candidate_id.clear();
        assert_eq!(
            destination_promotion_permit(&empty_candidate, &binding(), &candidate_evidence),
            Err(DestinationRequalificationError::EmptyCandidateId)
        );

        let mut empty_search = provenance(ScientificSearchDispositionV1::Survivor);
        empty_search.search_evidence_id.clear();
        let mut search_evidence = evidence();
        search_evidence.search_evidence_id.clear();
        assert_eq!(
            destination_promotion_permit(&empty_search, &binding(), &search_evidence),
            Err(DestinationRequalificationError::EmptySearchEvidenceId)
        );
    }

    #[test]
    fn failed_destination_qualification_never_mints_permit() {
        let survivor = provenance(ScientificSearchDispositionV1::Survivor);
        let mut failed = evidence();
        failed.qualified = false;
        assert_eq!(
            destination_promotion_permit(&survivor, &binding(), &failed),
            Err(DestinationRequalificationError::RequalificationFailed)
        );
    }
}
