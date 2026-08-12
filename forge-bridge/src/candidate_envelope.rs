//! Architecture-neutral candidate envelope shared with CCOS Research-Lab.
//!
//! This module intentionally does not depend on CCOS/RSI. Interoperability is
//! pinned by a cross-repository golden vector. JSON is transport only; the
//! candidate identity and envelope fingerprint use the canonical binary v1
//! encoding defined by `docs/CANDIDATE_ENVELOPE_V1.md`.

use std::collections::BTreeMap;

use forge_core::Candidate;
use serde_json::Value;

pub const CANDIDATE_ENVELOPE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateEnvelopeV1 {
    pub schema_version: u16,
    pub candidate_id: String,
    pub producer_candidate_id: Option<String>,
    pub parent_candidate_id: Option<String>,
    pub domain: String,
    pub source_sha256: String,
    pub proposal_sha256: Option<String>,
    pub trial_seed: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CandidateEnvelopeError {
    EmptyDomain,
    InvalidSha256(&'static str),
    Json(String),
}

impl std::fmt::Display for CandidateEnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CandidateEnvelopeError {}

impl CandidateEnvelopeV1 {
    pub fn from_candidate<C: Candidate>(
        candidate: &C,
        domain: impl Into<String>,
        parent_candidate_id: Option<String>,
        proposal_sha256: Option<String>,
        trial_seed: u64,
    ) -> Result<Self, CandidateEnvelopeError> {
        let domain = domain.into();
        if domain.trim().is_empty() {
            return Err(CandidateEnvelopeError::EmptyDomain);
        }
        validate_optional_sha256(proposal_sha256.as_deref(), "proposal_sha256")?;
        validate_optional_sha256(parent_candidate_id.as_deref(), "parent_candidate_id")?;

        let repr = candidate.repr();
        let source_sha256 = digest_hex(repr.as_bytes());
        let candidate_id = candidate_identity(&domain, &source_sha256);
        Ok(Self {
            schema_version: CANDIDATE_ENVELOPE_SCHEMA_VERSION,
            candidate_id,
            producer_candidate_id: Some(candidate.id().to_string()),
            parent_candidate_id,
            domain,
            source_sha256,
            proposal_sha256,
            trial_seed,
        })
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CandidateEnvelopeError> {
        self.validate()?;
        let mut out = b"memorithm.candidate-envelope.v1\0".to_vec();
        out.extend_from_slice(&self.schema_version.to_le_bytes());
        push_str(&mut out, &self.candidate_id);
        push_opt_str(&mut out, self.producer_candidate_id.as_deref());
        push_opt_str(&mut out, self.parent_candidate_id.as_deref());
        out.push(1); // CandidateOrigin::Forge
        push_str(&mut out, &self.domain);
        push_str(&mut out, &self.source_sha256);
        push_opt_str(&mut out, self.proposal_sha256.as_deref());
        out.extend_from_slice(&self.trial_seed.to_le_bytes());
        Ok(out)
    }

    pub fn fingerprint(&self) -> Result<String, CandidateEnvelopeError> {
        Ok(digest_hex(&self.canonical_bytes()?))
    }

    pub fn to_wire_json(&self) -> Result<String, CandidateEnvelopeError> {
        self.validate()?;
        let mut map = BTreeMap::new();
        map.insert("candidate_id", Value::String(self.candidate_id.clone()));
        map.insert("domain", Value::String(self.domain.clone()));
        map.insert("fingerprint", Value::String(self.fingerprint()?));
        map.insert("origin", Value::String("forge".to_string()));
        map.insert(
            "parent_candidate_id",
            self.parent_candidate_id
                .as_ref()
                .map(|value| Value::String(value.clone()))
                .unwrap_or(Value::Null),
        );
        map.insert(
            "producer_candidate_id",
            self.producer_candidate_id
                .as_ref()
                .map(|value| Value::String(value.clone()))
                .unwrap_or(Value::Null),
        );
        map.insert(
            "proposal_sha256",
            self.proposal_sha256
                .as_ref()
                .map(|value| Value::String(value.clone()))
                .unwrap_or(Value::Null),
        );
        map.insert(
            "schema_version",
            Value::Number(serde_json::Number::from(self.schema_version)),
        );
        map.insert("source_sha256", Value::String(self.source_sha256.clone()));
        // JSON numbers are not permitted for protocol u64 values: parsers that
        // pass through f64 would lose precision above 2^53.
        map.insert("trial_seed", Value::String(self.trial_seed.to_string()));
        serde_json::to_string(&map).map_err(|error| CandidateEnvelopeError::Json(error.to_string()))
    }

    pub fn validate(&self) -> Result<(), CandidateEnvelopeError> {
        if self.schema_version != CANDIDATE_ENVELOPE_SCHEMA_VERSION {
            return Err(CandidateEnvelopeError::Json(
                "unsupported candidate envelope schema".to_string(),
            ));
        }
        if self.domain.trim().is_empty() {
            return Err(CandidateEnvelopeError::EmptyDomain);
        }
        validate_sha256(&self.source_sha256, "source_sha256")?;
        validate_sha256(&self.candidate_id, "candidate_id")?;
        validate_optional_sha256(self.parent_candidate_id.as_deref(), "parent_candidate_id")?;
        validate_optional_sha256(self.proposal_sha256.as_deref(), "proposal_sha256")?;
        if self.candidate_id != candidate_identity(&self.domain, &self.source_sha256) {
            return Err(CandidateEnvelopeError::InvalidSha256("candidate_id"));
        }
        Ok(())
    }
}

fn candidate_identity(domain: &str, source_sha256: &str) -> String {
    let mut identity = b"memorithm.candidate.identity.v1\0".to_vec();
    identity.push(1); // CandidateOrigin::Forge
    push_str(&mut identity, domain);
    push_str(&mut identity, source_sha256);
    digest_hex(&identity)
}

fn push_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn push_opt_str(out: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            out.push(1);
            push_str(out, value);
        }
        None => out.push(0),
    }
}

fn validate_optional_sha256(
    value: Option<&str>,
    field: &'static str,
) -> Result<(), CandidateEnvelopeError> {
    if let Some(value) = value {
        validate_sha256(value, field)?;
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), CandidateEnvelopeError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CandidateEnvelopeError::InvalidSha256(field));
    }
    Ok(())
}

fn digest_hex(data: &[u8]) -> String {
    let digest = sha256(data);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

// Compact FIPS 180-4 reference implementation kept local so the bridge's wire
// contract does not expand Forge's dependency graph or Cargo.lock.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    for block in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in w.iter_mut().take(16).enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                block[offset],
                block[offset + 1],
                block[offset + 2],
                block[offset + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }

        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];
        for index in 0..64 {
            let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(sigma1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sigma0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut digest = [0u8; 32];
    for (index, word) in state.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    struct GoldenCandidate;

    impl Candidate for GoldenCandidate {
        fn id(&self) -> u64 {
            42
        }

        fn repr(&self) -> String {
            "pub fn kernel() {}".to_string()
        }
    }

    #[test]
    fn sha256_matches_nist_vector() {
        assert_eq!(
            digest_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn research_lab_candidate_v1_golden_vector_matches() {
        let envelope = CandidateEnvelopeV1::from_candidate(
            &GoldenCandidate,
            "simd_gemm",
            None,
            Some("11".repeat(32)),
            u64::MAX,
        )
        .unwrap();
        assert_eq!(
            envelope.source_sha256,
            "3b6e6e212c45273719067e12eac78aceaf44fbb2ffcafef4ab4519a64c5083e1"
        );
        assert_eq!(
            envelope.candidate_id,
            "4457784cc3119a48ab2f90fbac86d5e5c1ab0c99b46b567edd8dbd1bb3a3446f"
        );
        assert_eq!(
            envelope.fingerprint().unwrap(),
            "9a531d78fbf991077c087bdac953db53b1ede544349a71c5e6bdbe25f00e8693"
        );
        assert_eq!(
            envelope.to_wire_json().unwrap(),
            concat!(
                "{\"candidate_id\":\"4457784cc3119a48ab2f90fbac86d5e5c1ab0c99b46b567edd8dbd1bb3a3446f\",",
                "\"domain\":\"simd_gemm\",",
                "\"fingerprint\":\"9a531d78fbf991077c087bdac953db53b1ede544349a71c5e6bdbe25f00e8693\",",
                "\"origin\":\"forge\",",
                "\"parent_candidate_id\":null,",
                "\"producer_candidate_id\":\"42\",",
                "\"proposal_sha256\":\"1111111111111111111111111111111111111111111111111111111111111111\",",
                "\"schema_version\":1,",
                "\"source_sha256\":\"3b6e6e212c45273719067e12eac78aceaf44fbb2ffcafef4ab4519a64c5083e1\",",
                "\"trial_seed\":\"18446744073709551615\"}"
            )
        );
    }
}
