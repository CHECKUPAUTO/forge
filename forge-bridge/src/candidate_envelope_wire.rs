//! Strict wire JSON parser for `CandidateEnvelopeV1`.
//!
//! The parser deliberately accepts only the v1 transport shape documented in
//! `docs/CANDIDATE_ENVELOPE_V1.md`. It reconstructs the typed envelope and then
//! verifies both its structural invariants and its canonical fingerprint.

use serde_json::{Map, Value};

use crate::candidate_envelope::{
    CandidateEnvelopeError, CandidateEnvelopeV1, CANDIDATE_ENVELOPE_SCHEMA_VERSION,
};

const WIRE_FIELDS: [&str; 10] = [
    "candidate_id",
    "domain",
    "fingerprint",
    "origin",
    "parent_candidate_id",
    "producer_candidate_id",
    "proposal_sha256",
    "schema_version",
    "source_sha256",
    "trial_seed",
];

impl CandidateEnvelopeV1 {
    /// Parse and authenticate the integrity of a v1 wire JSON envelope.
    ///
    /// This verifies the exact v1 field set, Forge origin, string-encoded
    /// `trial_seed`, candidate identity invariants, and the canonical envelope
    /// fingerprint. It does not authenticate the producer cryptographically.
    pub fn from_wire_json(json: &str) -> Result<Self, CandidateEnvelopeError> {
        let value: Value = serde_json::from_str(json)
            .map_err(|error| CandidateEnvelopeError::Json(error.to_string()))?;
        let mut object = match value {
            Value::Object(object) => object,
            _ => return Err(wire_error("candidate envelope must be a JSON object")),
        };

        for field in object.keys() {
            if !WIRE_FIELDS.contains(&field.as_str()) {
                return Err(wire_error(format!(
                    "unknown candidate envelope field: {field}"
                )));
            }
        }

        let candidate_id = take_required_string(&mut object, "candidate_id")?;
        let domain = take_required_string(&mut object, "domain")?;
        let expected_fingerprint = take_required_string(&mut object, "fingerprint")?;
        if !is_sha256_hex(&expected_fingerprint) {
            return Err(CandidateEnvelopeError::InvalidSha256("fingerprint"));
        }

        let origin = take_required_string(&mut object, "origin")?;
        if origin != "forge" {
            return Err(wire_error("candidate envelope origin must be `forge`"));
        }

        let parent_candidate_id = take_optional_string(&mut object, "parent_candidate_id")?;
        let producer_candidate_id = take_optional_string(&mut object, "producer_candidate_id")?;
        let proposal_sha256 = take_optional_string(&mut object, "proposal_sha256")?;
        let schema_version = take_schema_version(&mut object)?;
        let source_sha256 = take_required_string(&mut object, "source_sha256")?;
        let trial_seed = take_trial_seed(&mut object)?;

        debug_assert!(object.is_empty());

        let envelope = Self {
            schema_version,
            candidate_id,
            producer_candidate_id,
            parent_candidate_id,
            domain,
            source_sha256,
            proposal_sha256,
            trial_seed,
        };
        envelope.validate()?;

        if envelope.fingerprint()? != expected_fingerprint {
            return Err(wire_error("candidate envelope fingerprint mismatch"));
        }

        Ok(envelope)
    }
}

fn take_required_string(
    object: &mut Map<String, Value>,
    field: &'static str,
) -> Result<String, CandidateEnvelopeError> {
    match object.remove(field) {
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(wire_error(format!(
            "candidate envelope field `{field}` must be a string"
        ))),
        None => Err(wire_error(format!(
            "candidate envelope field `{field}` is required"
        ))),
    }
}

fn take_optional_string(
    object: &mut Map<String, Value>,
    field: &'static str,
) -> Result<Option<String>, CandidateEnvelopeError> {
    match object.remove(field) {
        Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(wire_error(format!(
            "candidate envelope field `{field}` must be a string or null"
        ))),
        None => Err(wire_error(format!(
            "candidate envelope field `{field}` is required"
        ))),
    }
}

fn take_schema_version(object: &mut Map<String, Value>) -> Result<u16, CandidateEnvelopeError> {
    let version = match object.remove("schema_version") {
        Some(Value::Number(value)) => value
            .as_u64()
            .and_then(|value| u16::try_from(value).ok())
            .ok_or_else(|| wire_error("candidate envelope field `schema_version` must be a u16"))?,
        Some(_) => {
            return Err(wire_error(
                "candidate envelope field `schema_version` must be a JSON integer",
            ));
        }
        None => {
            return Err(wire_error(
                "candidate envelope field `schema_version` is required",
            ));
        }
    };

    if version != CANDIDATE_ENVELOPE_SCHEMA_VERSION {
        return Err(wire_error("unsupported candidate envelope schema"));
    }

    Ok(version)
}

fn take_trial_seed(object: &mut Map<String, Value>) -> Result<u64, CandidateEnvelopeError> {
    let seed = match object.remove("trial_seed") {
        Some(Value::String(value)) => value,
        Some(_) => {
            return Err(wire_error(
                "candidate envelope field `trial_seed` must be a decimal string",
            ));
        }
        None => {
            return Err(wire_error(
                "candidate envelope field `trial_seed` is required",
            ));
        }
    };

    seed.parse::<u64>().map_err(|_| {
        wire_error("candidate envelope field `trial_seed` is not a valid u64 decimal string")
    })
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn wire_error(message: impl Into<String>) -> CandidateEnvelopeError {
    CandidateEnvelopeError::Json(message.into())
}

#[cfg(test)]
mod tests {
    use forge_core::Candidate;
    use serde_json::json;

    use super::*;

    #[derive(Clone)]
    struct WireCandidate;

    impl Candidate for WireCandidate {
        fn id(&self) -> u64 {
            42
        }

        fn repr(&self) -> String {
            "pub fn kernel() {}".to_string()
        }
    }

    fn envelope() -> CandidateEnvelopeV1 {
        CandidateEnvelopeV1::from_candidate(
            &WireCandidate,
            "simd_gemm",
            None,
            Some("11".repeat(32)),
            u64::MAX,
        )
        .expect("golden envelope must be valid")
    }

    fn wire_value() -> Value {
        serde_json::from_str(&envelope().to_wire_json().expect("wire JSON must serialize"))
            .expect("wire JSON must parse")
    }

    #[test]
    fn wire_round_trip_preserves_u64_max() {
        let expected = envelope();
        let wire = expected.to_wire_json().expect("wire JSON must serialize");
        let parsed = CandidateEnvelopeV1::from_wire_json(&wire).expect("wire JSON must validate");
        assert_eq!(parsed, expected);
    }

    #[test]
    fn wire_parser_rejects_numeric_trial_seed() {
        let mut wire = wire_value();
        wire["trial_seed"] = json!(u64::MAX);
        let error = CandidateEnvelopeV1::from_wire_json(&wire.to_string()).unwrap_err();
        assert!(error.to_string().contains("decimal string"));
    }

    #[test]
    fn wire_parser_rejects_fingerprint_mutation() {
        let mut wire = wire_value();
        wire["fingerprint"] = Value::String("00".repeat(32));
        let error = CandidateEnvelopeV1::from_wire_json(&wire.to_string()).unwrap_err();
        assert!(error.to_string().contains("fingerprint mismatch"));
    }

    #[test]
    fn wire_parser_rejects_wrong_origin() {
        let mut wire = wire_value();
        wire["origin"] = Value::String("other".to_string());
        let error = CandidateEnvelopeV1::from_wire_json(&wire.to_string()).unwrap_err();
        assert!(error.to_string().contains("origin"));
    }

    #[test]
    fn wire_parser_rejects_unknown_field() {
        let mut wire = wire_value();
        wire["unexpected"] = Value::Bool(true);
        let error = CandidateEnvelopeV1::from_wire_json(&wire.to_string()).unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown candidate envelope field"));
    }

    #[test]
    fn wire_parser_rejects_missing_required_field() {
        let mut wire = wire_value();
        wire.as_object_mut()
            .expect("wire value must be an object")
            .remove("source_sha256");
        let error = CandidateEnvelopeV1::from_wire_json(&wire.to_string()).unwrap_err();
        assert!(error.to_string().contains("source_sha256"));
    }
}
