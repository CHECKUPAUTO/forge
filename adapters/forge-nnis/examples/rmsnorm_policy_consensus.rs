use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;

const POLICY_SCHEMA_VERSION: u64 = 1;
const POLICY_KIND: &str = "forge_nnis_rmsnorm_shape_policy_recommendation_v1";
const SOURCE_CAMPAIGN_KIND: &str = "forge_nnis_canonical_rmsnorm_shape_matrix_v1";
const CONSENSUS_SCHEMA_VERSION: u64 = 1;
const CONSENSUS_KIND: &str = "forge_nnis_rmsnorm_shape_policy_consensus_v1";
const DEFAULT_MIN_RUNS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    CandidatePreferred,
    BaselinePreferred,
    Inconclusive,
}

impl Decision {
    fn parse(value: &str) -> ConsensusResult<Self> {
        match value {
            "candidate_preferred" => Ok(Self::CandidatePreferred),
            "baseline_preferred" => Ok(Self::BaselinePreferred),
            "inconclusive" => Ok(Self::Inconclusive),
            other => Err(ConsensusError(format!("unsupported decision {other:?}"))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::CandidatePreferred => "candidate_preferred",
            Self::BaselinePreferred => "baseline_preferred",
            Self::Inconclusive => "inconclusive",
        }
    }
}

#[derive(Debug)]
struct ConsensusError(String);

impl Display for ConsensusError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ConsensusError {}

type ConsensusResult<T> = Result<T, ConsensusError>;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Thresholds {
    min_relative_improvement: f64,
    min_round_win_fraction: f64,
}

#[derive(Clone, Debug)]
struct ShapeRecommendation {
    decision: Decision,
    recommended_block_size: Option<u64>,
    median_paired_relative_improvement: f64,
    candidate_round_wins: u64,
    baseline_round_wins: u64,
    round_ties: u64,
}

#[derive(Clone, Debug)]
struct ParsedPolicy {
    run_context_id: String,
    baseline_block_size: u64,
    candidate_block_size: u64,
    rounds_per_shape: u64,
    shape_count: usize,
    thresholds: Thresholds,
    recommendations: BTreeMap<(u64, u64), ShapeRecommendation>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let paths = std::env::args().skip(1).collect::<Vec<_>>();
    let min_runs = env_usize("FORGE_NNIS_RMSNORM_CONSENSUS_MIN_RUNS", DEFAULT_MIN_RUNS)?;
    if min_runs < 2 {
        return Err("FORGE_NNIS_RMSNORM_CONSENSUS_MIN_RUNS must be at least 2".into());
    }
    if paths.len() < min_runs {
        return Err(format!(
            "usage: rmsnorm_policy_consensus <policy-1.json> <policy-2.json> [...]; at least {min_runs} independent runs are required"
        )
        .into());
    }

    let mut policies = Vec::with_capacity(paths.len());
    for path in paths {
        let value: Value = serde_json::from_str(&fs::read_to_string(&path)?)?;
        policies.push(parse_policy(&value, &path)?);
    }

    let consensus = build_consensus(&policies, min_runs)?;
    println!("{}", serde_json::to_string_pretty(&consensus)?);
    Ok(())
}

fn parse_policy(value: &Value, label: &str) -> ConsensusResult<ParsedPolicy> {
    let root = object(value, label)?;
    require_u64(root, "schema_version", POLICY_SCHEMA_VERSION)?;
    require_str(root, "recommendation_kind", POLICY_KIND)?;

    let source = object_field(root, "source_campaign")?;
    require_u64(source, "schema_version", POLICY_SCHEMA_VERSION)?;
    require_str(source, "campaign_kind", SOURCE_CAMPAIGN_KIND)?;
    let run_context_id = nonempty_string(source, "run_context_id")?;
    let baseline_block_size = positive_u64(source, "baseline_block_size")?;
    let candidate_block_size = positive_u64(source, "candidate_block_size")?;
    if baseline_block_size == candidate_block_size {
        return Err(ConsensusError(format!(
            "{label}: baseline and candidate block sizes must differ"
        )));
    }
    let rounds_per_shape = positive_u64(source, "rounds_per_shape")?;
    if rounds_per_shape < 2 {
        return Err(ConsensusError(format!(
            "{label}: rounds_per_shape must be at least 2"
        )));
    }
    let shape_count = usize_from_u64(positive_u64(source, "shape_count")?, "shape_count")?;

    let threshold_object = object_field(root, "thresholds")?;
    let thresholds = Thresholds {
        min_relative_improvement: finite_f64(threshold_object, "min_relative_improvement")?,
        min_round_win_fraction: finite_f64(threshold_object, "min_round_win_fraction")?,
    };
    validate_thresholds(thresholds, label)?;

    let declared_candidate = usize_from_u64(
        nonnegative_u64(root, "candidate_recommendations")?,
        "candidate_recommendations",
    )?;
    let declared_baseline = usize_from_u64(
        nonnegative_u64(root, "baseline_recommendations")?,
        "baseline_recommendations",
    )?;
    let declared_inconclusive =
        usize_from_u64(nonnegative_u64(root, "inconclusive")?, "inconclusive")?;
    if declared_candidate + declared_baseline + declared_inconclusive != shape_count {
        return Err(ConsensusError(format!(
            "{label}: declared recommendation counts do not sum to shape_count"
        )));
    }

    let entries = array_field(root, "recommendations")?;
    if entries.len() != shape_count {
        return Err(ConsensusError(format!(
            "{label}: shape_count {shape_count} does not match recommendations length {}",
            entries.len()
        )));
    }

    let mut recommendations = BTreeMap::new();
    let mut observed_candidate = 0usize;
    let mut observed_baseline = 0usize;
    let mut observed_inconclusive = 0usize;

    for (index, entry) in entries.iter().enumerate() {
        let entry = object(entry, &format!("{label}.recommendations[{index}]"))?;
        let rows = positive_u64(entry, "rows")?;
        let cols = positive_u64(entry, "cols")?;
        let decision = Decision::parse(
            entry
                .get("decision")
                .and_then(Value::as_str)
                .ok_or_else(|| ConsensusError(format!("{label}: decision must be a string")))?,
        )?;
        let recommended_block_size = optional_u64(entry, "recommended_block_size")?;
        match decision {
            Decision::CandidatePreferred => {
                observed_candidate += 1;
                if recommended_block_size != Some(candidate_block_size) {
                    return Err(ConsensusError(format!(
                        "{label}: candidate_preferred {rows}x{cols} must recommend block {candidate_block_size}"
                    )));
                }
            }
            Decision::BaselinePreferred => {
                observed_baseline += 1;
                if recommended_block_size != Some(baseline_block_size) {
                    return Err(ConsensusError(format!(
                        "{label}: baseline_preferred {rows}x{cols} must recommend block {baseline_block_size}"
                    )));
                }
            }
            Decision::Inconclusive => {
                observed_inconclusive += 1;
                if recommended_block_size.is_some() {
                    return Err(ConsensusError(format!(
                        "{label}: inconclusive {rows}x{cols} must not recommend a block size"
                    )));
                }
            }
        }

        let candidate_round_wins = nonnegative_u64(entry, "candidate_round_wins")?;
        let baseline_round_wins = nonnegative_u64(entry, "baseline_round_wins")?;
        let round_ties = nonnegative_u64(entry, "round_ties")?;
        if candidate_round_wins + baseline_round_wins + round_ties != rounds_per_shape {
            return Err(ConsensusError(format!(
                "{label}: {rows}x{cols} round counts do not sum to rounds_per_shape"
            )));
        }
        let median_paired_relative_improvement =
            finite_f64(entry, "median_paired_relative_improvement")?;

        if recommendations
            .insert(
                (rows, cols),
                ShapeRecommendation {
                    decision,
                    recommended_block_size,
                    median_paired_relative_improvement,
                    candidate_round_wins,
                    baseline_round_wins,
                    round_ties,
                },
            )
            .is_some()
        {
            return Err(ConsensusError(format!(
                "{label}: duplicate shape {rows}x{cols}"
            )));
        }
    }

    if observed_candidate != declared_candidate
        || observed_baseline != declared_baseline
        || observed_inconclusive != declared_inconclusive
    {
        return Err(ConsensusError(format!(
            "{label}: declared recommendation counts do not match recommendation entries"
        )));
    }

    Ok(ParsedPolicy {
        run_context_id,
        baseline_block_size,
        candidate_block_size,
        rounds_per_shape,
        shape_count,
        thresholds,
        recommendations,
    })
}

fn build_consensus(policies: &[ParsedPolicy], min_runs: usize) -> ConsensusResult<Value> {
    if policies.len() < min_runs || min_runs < 2 {
        return Err(ConsensusError(format!(
            "at least {min_runs} independent policy runs are required"
        )));
    }

    let reference = policies
        .first()
        .ok_or_else(|| ConsensusError("no policies supplied".to_string()))?;
    let mut run_context_ids = BTreeSet::new();
    for policy in policies {
        if !run_context_ids.insert(policy.run_context_id.clone()) {
            return Err(ConsensusError(format!(
                "duplicate run_context_id {:?}; consensus requires independent runs",
                policy.run_context_id
            )));
        }
        require_same_campaign(reference, policy)?;
    }

    let run_count = policies.len();
    let mut candidate_recommendations = 0usize;
    let mut baseline_recommendations = 0usize;
    let mut inconclusive = 0usize;
    let mut recommendations = Vec::with_capacity(reference.shape_count);

    for (&(rows, cols), reference_shape) in &reference.recommendations {
        let mut candidate_support_runs = 0usize;
        let mut baseline_support_runs = 0usize;
        let mut inconclusive_runs = 0usize;
        let mut min_improvement = f64::INFINITY;
        let mut max_improvement = f64::NEG_INFINITY;
        let mut per_run = Vec::with_capacity(run_count);

        for policy in policies {
            let shape = policy.recommendations.get(&(rows, cols)).ok_or_else(|| {
                ConsensusError(format!(
                    "run {:?} is missing shape {rows}x{cols}",
                    policy.run_context_id
                ))
            })?;
            match shape.decision {
                Decision::CandidatePreferred => candidate_support_runs += 1,
                Decision::BaselinePreferred => baseline_support_runs += 1,
                Decision::Inconclusive => inconclusive_runs += 1,
            }
            min_improvement = min_improvement.min(shape.median_paired_relative_improvement);
            max_improvement = max_improvement.max(shape.median_paired_relative_improvement);
            per_run.push(json!({
                "run_context_id": policy.run_context_id,
                "decision": shape.decision.as_str(),
                "recommended_block_size": shape.recommended_block_size,
                "median_paired_relative_improvement": shape.median_paired_relative_improvement,
                "candidate_round_wins": shape.candidate_round_wins,
                "baseline_round_wins": shape.baseline_round_wins,
                "round_ties": shape.round_ties,
            }));
        }

        let decision = if candidate_support_runs == run_count {
            candidate_recommendations += 1;
            Decision::CandidatePreferred
        } else if baseline_support_runs == run_count {
            baseline_recommendations += 1;
            Decision::BaselinePreferred
        } else {
            inconclusive += 1;
            Decision::Inconclusive
        };
        let recommended_block_size = match decision {
            Decision::CandidatePreferred => Some(reference.candidate_block_size),
            Decision::BaselinePreferred => Some(reference.baseline_block_size),
            Decision::Inconclusive => None,
        };

        recommendations.push(json!({
            "rows": rows,
            "cols": cols,
            "decision": decision.as_str(),
            "recommended_block_size": recommended_block_size,
            "candidate_support_runs": candidate_support_runs,
            "baseline_support_runs": baseline_support_runs,
            "inconclusive_runs": inconclusive_runs,
            "min_median_paired_relative_improvement": min_improvement,
            "max_median_paired_relative_improvement": max_improvement,
            "runs": per_run,
        }));

        if reference_shape.recommended_block_size.is_some()
            && reference_shape.decision == Decision::Inconclusive
        {
            return Err(ConsensusError(format!(
                "reference shape {rows}x{cols} is internally inconsistent"
            )));
        }
    }

    Ok(json!({
        "schema_version": CONSENSUS_SCHEMA_VERSION,
        "consensus_kind": CONSENSUS_KIND,
        "source_policy_kind": POLICY_KIND,
        "source_campaign_kind": SOURCE_CAMPAIGN_KIND,
        "run_count": run_count,
        "minimum_required_runs": min_runs,
        "run_context_ids": run_context_ids.into_iter().collect::<Vec<_>>(),
        "baseline_block_size": reference.baseline_block_size,
        "candidate_block_size": reference.candidate_block_size,
        "rounds_per_shape": reference.rounds_per_shape,
        "shape_count": reference.shape_count,
        "thresholds": {
            "min_relative_improvement": reference.thresholds.min_relative_improvement,
            "min_round_win_fraction": reference.thresholds.min_round_win_fraction,
        },
        "candidate_recommendations": candidate_recommendations,
        "baseline_recommendations": baseline_recommendations,
        "inconclusive": inconclusive,
        "recommendations": recommendations,
        "claim_boundary": "consensus qualifies repeatability of microbenchmark policy evidence only; it does not authorize NNIS runtime promotion or imply end-to-end model speedup",
    }))
}

fn require_same_campaign(reference: &ParsedPolicy, policy: &ParsedPolicy) -> ConsensusResult<()> {
    if policy.baseline_block_size != reference.baseline_block_size {
        return Err(ConsensusError(format!(
            "run {:?} baseline_block_size differs from reference",
            policy.run_context_id
        )));
    }
    if policy.candidate_block_size != reference.candidate_block_size {
        return Err(ConsensusError(format!(
            "run {:?} candidate_block_size differs from reference",
            policy.run_context_id
        )));
    }
    if policy.rounds_per_shape != reference.rounds_per_shape {
        return Err(ConsensusError(format!(
            "run {:?} rounds_per_shape differs from reference",
            policy.run_context_id
        )));
    }
    if policy.shape_count != reference.shape_count {
        return Err(ConsensusError(format!(
            "run {:?} shape_count differs from reference",
            policy.run_context_id
        )));
    }
    if policy.thresholds != reference.thresholds {
        return Err(ConsensusError(format!(
            "run {:?} thresholds differ from reference",
            policy.run_context_id
        )));
    }
    if policy.recommendations.keys().ne(reference.recommendations.keys()) {
        return Err(ConsensusError(format!(
            "run {:?} shape set differs from reference",
            policy.run_context_id
        )));
    }
    Ok(())
}

fn validate_thresholds(thresholds: Thresholds, label: &str) -> ConsensusResult<()> {
    if thresholds.min_relative_improvement <= 0.0 || thresholds.min_relative_improvement >= 1.0 {
        return Err(ConsensusError(format!(
            "{label}: min_relative_improvement must be in (0, 1)"
        )));
    }
    if thresholds.min_round_win_fraction <= 0.5 || thresholds.min_round_win_fraction > 1.0 {
        return Err(ConsensusError(format!(
            "{label}: min_round_win_fraction must be in (0.5, 1]"
        )));
    }
    Ok(())
}

fn object<'a>(value: &'a Value, label: &str) -> ConsensusResult<&'a serde_json::Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| ConsensusError(format!("{label} must be a JSON object")))
}

fn object_field<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> ConsensusResult<&'a serde_json::Map<String, Value>> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| ConsensusError(format!("{field} must be an object")))
}

fn array_field<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> ConsensusResult<&'a Vec<Value>> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| ConsensusError(format!("{field} must be an array")))
}

fn require_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
    expected: u64,
) -> ConsensusResult<()> {
    let actual = object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| ConsensusError(format!("{field} must be a non-negative integer")))?;
    if actual != expected {
        return Err(ConsensusError(format!(
            "{field} must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn require_str(
    object: &serde_json::Map<String, Value>,
    field: &str,
    expected: &str,
) -> ConsensusResult<()> {
    let actual = object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ConsensusError(format!("{field} must be a string")))?;
    if actual != expected {
        return Err(ConsensusError(format!(
            "{field} must be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn nonempty_string(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> ConsensusResult<String> {
    let value = object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ConsensusError(format!("{field} must be a string")))?;
    if value.trim().is_empty() {
        return Err(ConsensusError(format!("{field} must not be empty")));
    }
    Ok(value.to_string())
}

fn nonnegative_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> ConsensusResult<u64> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| ConsensusError(format!("{field} must be a non-negative integer")))
}

fn positive_u64(object: &serde_json::Map<String, Value>, field: &str) -> ConsensusResult<u64> {
    let value = nonnegative_u64(object, field)?;
    if value == 0 {
        return Err(ConsensusError(format!("{field} must be positive")));
    }
    Ok(value)
}

fn optional_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> ConsensusResult<Option<u64>> {
    match object.get(field) {
        Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| ConsensusError(format!("{field} must be an integer or null"))),
        None => Err(ConsensusError(format!("missing field {field}"))),
    }
}

fn finite_f64(object: &serde_json::Map<String, Value>, field: &str) -> ConsensusResult<f64> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| ConsensusError(format!("{field} must be numeric")))?;
    if !value.is_finite() {
        return Err(ConsensusError(format!("{field} must be finite")));
    }
    Ok(value)
}

fn usize_from_u64(value: u64, field: &str) -> ConsensusResult<usize> {
    usize::try_from(value).map_err(|_| ConsensusError(format!("{field} does not fit usize")))
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(default))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recommendation(
        rows: u64,
        cols: u64,
        decision: &str,
        block: Option<u64>,
        improvement: f64,
        candidate_wins: u64,
        baseline_wins: u64,
        ties: u64,
    ) -> Value {
        json!({
            "rows": rows,
            "cols": cols,
            "decision": decision,
            "recommended_block_size": block,
            "median_paired_relative_improvement": improvement,
            "candidate_round_wins": candidate_wins,
            "baseline_round_wins": baseline_wins,
            "round_ties": ties,
        })
    }

    fn policy(run: &str, recommendations: Vec<Value>) -> Value {
        let candidate = recommendations
            .iter()
            .filter(|value| value["decision"] == "candidate_preferred")
            .count();
        let baseline = recommendations
            .iter()
            .filter(|value| value["decision"] == "baseline_preferred")
            .count();
        let inconclusive = recommendations.len() - candidate - baseline;
        json!({
            "schema_version": 1,
            "recommendation_kind": POLICY_KIND,
            "source_campaign": {
                "schema_version": 1,
                "campaign_kind": SOURCE_CAMPAIGN_KIND,
                "run_context_id": run,
                "baseline_block_size": 256,
                "candidate_block_size": 512,
                "rounds_per_shape": 4,
                "shape_count": recommendations.len(),
            },
            "thresholds": {
                "min_relative_improvement": 0.03,
                "min_round_win_fraction": 1.0,
            },
            "candidate_recommendations": candidate,
            "baseline_recommendations": baseline,
            "inconclusive": inconclusive,
            "recommendations": recommendations,
        })
    }

    #[test]
    fn unanimous_candidate_is_recommended() {
        let first = parse_policy(
            &policy(
                "run-a",
                vec![recommendation(
                    32,
                    4096,
                    "candidate_preferred",
                    Some(512),
                    0.19,
                    4,
                    0,
                    0,
                )],
            ),
            "first",
        )
        .unwrap();
        let second = parse_policy(
            &policy(
                "run-b",
                vec![recommendation(
                    32,
                    4096,
                    "candidate_preferred",
                    Some(512),
                    0.18,
                    4,
                    0,
                    0,
                )],
            ),
            "second",
        )
        .unwrap();
        let consensus = build_consensus(&[first, second], 2).unwrap();
        assert_eq!(consensus["candidate_recommendations"], 1);
        assert_eq!(consensus["recommendations"][0]["decision"], "candidate_preferred");
        assert_eq!(consensus["recommendations"][0]["recommended_block_size"], 512);
    }

    #[test]
    fn disagreement_is_inconclusive() {
        let first = parse_policy(
            &policy(
                "run-a",
                vec![recommendation(
                    128,
                    2048,
                    "candidate_preferred",
                    Some(512),
                    0.05,
                    4,
                    0,
                    0,
                )],
            ),
            "first",
        )
        .unwrap();
        let second = parse_policy(
            &policy(
                "run-b",
                vec![recommendation(
                    128,
                    2048,
                    "inconclusive",
                    None,
                    -0.01,
                    0,
                    4,
                    0,
                )],
            ),
            "second",
        )
        .unwrap();
        let consensus = build_consensus(&[first, second], 2).unwrap();
        assert_eq!(consensus["inconclusive"], 1);
        assert_eq!(consensus["recommendations"][0]["decision"], "inconclusive");
        assert!(consensus["recommendations"][0]["recommended_block_size"].is_null());
    }

    #[test]
    fn duplicate_run_context_fails_closed() {
        let value = policy(
            "same-run",
            vec![recommendation(
                1,
                4096,
                "candidate_preferred",
                Some(512),
                0.2,
                4,
                0,
                0,
            )],
        );
        let first = parse_policy(&value, "first").unwrap();
        let second = parse_policy(&value, "second").unwrap();
        assert!(build_consensus(&[first, second], 2).is_err());
    }

    #[test]
    fn mismatched_thresholds_fail_closed() {
        let first_value = policy(
            "run-a",
            vec![recommendation(
                1,
                4096,
                "candidate_preferred",
                Some(512),
                0.2,
                4,
                0,
                0,
            )],
        );
        let mut second_value = policy(
            "run-b",
            vec![recommendation(
                1,
                4096,
                "candidate_preferred",
                Some(512),
                0.2,
                4,
                0,
                0,
            )],
        );
        second_value["thresholds"]["min_relative_improvement"] = json!(0.04);
        let first = parse_policy(&first_value, "first").unwrap();
        let second = parse_policy(&second_value, "second").unwrap();
        assert!(build_consensus(&[first, second], 2).is_err());
    }

    #[test]
    fn malformed_recommendation_block_fails_closed() {
        let value = policy(
            "run-a",
            vec![recommendation(
                1,
                4096,
                "candidate_preferred",
                Some(256),
                0.2,
                4,
                0,
                0,
            )],
        );
        assert!(parse_policy(&value, "policy").is_err());
    }

    #[test]
    fn missing_shape_in_other_run_fails_closed() {
        let first = parse_policy(
            &policy(
                "run-a",
                vec![recommendation(
                    1,
                    4096,
                    "candidate_preferred",
                    Some(512),
                    0.2,
                    4,
                    0,
                    0,
                )],
            ),
            "first",
        )
        .unwrap();
        let second = parse_policy(
            &policy(
                "run-b",
                vec![recommendation(
                    1,
                    8192,
                    "candidate_preferred",
                    Some(512),
                    0.2,
                    4,
                    0,
                    0,
                )],
            ),
            "second",
        )
        .unwrap();
        assert!(build_consensus(&[first, second], 2).is_err());
    }
}
