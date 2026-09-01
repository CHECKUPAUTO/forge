use serde_json::{json, Map, Value};
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
    fn parse(value: &str) -> Result<Self, ConsensusError> {
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

#[derive(Clone, Copy, Debug)]
struct Thresholds {
    min_relative_improvement: f64,
    min_round_win_fraction: f64,
}

impl Thresholds {
    fn matches(self, other: Self) -> bool {
        self.min_relative_improvement.to_bits() == other.min_relative_improvement.to_bits()
            && self.min_round_win_fraction.to_bits() == other.min_round_win_fraction.to_bits()
    }
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
    let min_runs = env_usize(
        "FORGE_NNIS_RMSNORM_CONSENSUS_MIN_RUNS",
        DEFAULT_MIN_RUNS,
    )?;
    if min_runs < 2 {
        return Err("FORGE_NNIS_RMSNORM_CONSENSUS_MIN_RUNS must be at least 2".into());
    }
    if paths.len() < min_runs {
        return Err(format!(
            "rmsnorm_policy_consensus requires at least {min_runs} independent policy files"
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
    let root = as_object(value, label)?;
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
    let shape_count = to_usize(positive_u64(source, "shape_count")?, "shape_count")?;

    let threshold_object = object_field(root, "thresholds")?;
    let thresholds = Thresholds {
        min_relative_improvement: finite_f64(
            threshold_object,
            "min_relative_improvement",
        )?,
        min_round_win_fraction: finite_f64(threshold_object, "min_round_win_fraction")?,
    };
    validate_thresholds(thresholds, label)?;

    let declared_candidate = to_usize(
        nonnegative_u64(root, "candidate_recommendations")?,
        "candidate_recommendations",
    )?;
    let declared_baseline = to_usize(
        nonnegative_u64(root, "baseline_recommendations")?,
        "baseline_recommendations",
    )?;
    let declared_inconclusive = to_usize(
        nonnegative_u64(root, "inconclusive")?,
        "inconclusive",
    )?;
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
    let mut observed = [0usize; 3];

    for (index, entry) in entries.iter().enumerate() {
        let entry_label = format!("{label}.recommendations[{index}]");
        let entry = as_object(entry, &entry_label)?;
        let rows = positive_u64(entry, "rows")?;
        let cols = positive_u64(entry, "cols")?;
        let decision = Decision::parse(string_field(entry, "decision")?)?;
        let recommended_block_size = optional_u64(entry, "recommended_block_size")?;
        let candidate_round_wins = nonnegative_u64(entry, "candidate_round_wins")?;
        let baseline_round_wins = nonnegative_u64(entry, "baseline_round_wins")?;
        let round_ties = nonnegative_u64(entry, "round_ties")?;
        if candidate_round_wins + baseline_round_wins + round_ties != rounds_per_shape {
            return Err(ConsensusError(format!(
                "{entry_label}: round counts do not sum to rounds_per_shape"
            )));
        }
        let improvement = finite_f64(entry, "median_paired_relative_improvement")?;
        let derived = derive_decision(
            improvement,
            candidate_round_wins,
            baseline_round_wins,
            rounds_per_shape,
            thresholds,
        );
        if decision != derived {
            return Err(ConsensusError(format!(
                "{entry_label}: recorded decision {:?} disagrees with threshold-derived decision {:?}",
                decision, derived
            )));
        }

        match decision {
            Decision::CandidatePreferred => {
                observed[0] += 1;
                require_block(
                    recommended_block_size,
                    candidate_block_size,
                    &entry_label,
                )?;
            }
            Decision::BaselinePreferred => {
                observed[1] += 1;
                require_block(
                    recommended_block_size,
                    baseline_block_size,
                    &entry_label,
                )?;
            }
            Decision::Inconclusive => {
                observed[2] += 1;
                if recommended_block_size.is_some() {
                    return Err(ConsensusError(format!(
                        "{entry_label}: inconclusive decision must not recommend a block"
                    )));
                }
            }
        }

        let shape = ShapeRecommendation {
            decision,
            recommended_block_size,
            median_paired_relative_improvement: improvement,
            candidate_round_wins,
            baseline_round_wins,
            round_ties,
        };
        if recommendations.insert((rows, cols), shape).is_some() {
            return Err(ConsensusError(format!(
                "{label}: duplicate shape {rows}x{cols}"
            )));
        }
    }

    if observed != [declared_candidate, declared_baseline, declared_inconclusive] {
        return Err(ConsensusError(format!(
            "{label}: declared recommendation counts do not match entries"
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

fn derive_decision(
    improvement: f64,
    candidate_wins: u64,
    baseline_wins: u64,
    rounds: u64,
    thresholds: Thresholds,
) -> Decision {
    let rounds = rounds as f64;
    let candidate_fraction = candidate_wins as f64 / rounds;
    let baseline_fraction = baseline_wins as f64 / rounds;
    if improvement >= thresholds.min_relative_improvement
        && candidate_fraction >= thresholds.min_round_win_fraction
    {
        Decision::CandidatePreferred
    } else if improvement <= -thresholds.min_relative_improvement
        && baseline_fraction >= thresholds.min_round_win_fraction
    {
        Decision::BaselinePreferred
    } else {
        Decision::Inconclusive
    }
}

fn build_consensus(policies: &[ParsedPolicy], min_runs: usize) -> ConsensusResult<Value> {
    if min_runs < 2 || policies.len() < min_runs {
        return Err(ConsensusError(format!(
            "at least {min_runs} independent policies are required"
        )));
    }
    let reference = policies
        .first()
        .ok_or_else(|| ConsensusError("no policies supplied".to_string()))?;

    let mut run_context_ids = BTreeSet::new();
    for policy in policies {
        if !run_context_ids.insert(policy.run_context_id.clone()) {
            return Err(ConsensusError(format!(
                "duplicate run_context_id {:?}",
                policy.run_context_id
            )));
        }
        require_same_campaign(reference, policy)?;
    }

    let run_count = policies.len();
    let mut counts = [0usize; 3];
    let mut recommendations = Vec::with_capacity(reference.shape_count);

    for &(rows, cols) in reference.recommendations.keys() {
        let mut support = [0usize; 3];
        let mut min_improvement = f64::INFINITY;
        let mut max_improvement = f64::NEG_INFINITY;
        let mut runs = Vec::with_capacity(run_count);

        for policy in policies {
            let shape = policy.recommendations.get(&(rows, cols)).ok_or_else(|| {
                ConsensusError(format!(
                    "run {:?} is missing shape {rows}x{cols}",
                    policy.run_context_id
                ))
            })?;
            match shape.decision {
                Decision::CandidatePreferred => support[0] += 1,
                Decision::BaselinePreferred => support[1] += 1,
                Decision::Inconclusive => support[2] += 1,
            }
            min_improvement = min_improvement.min(shape.median_paired_relative_improvement);
            max_improvement = max_improvement.max(shape.median_paired_relative_improvement);
            runs.push(json!({
                "run_context_id": policy.run_context_id,
                "decision": shape.decision.as_str(),
                "recommended_block_size": shape.recommended_block_size,
                "median_paired_relative_improvement": shape.median_paired_relative_improvement,
                "candidate_round_wins": shape.candidate_round_wins,
                "baseline_round_wins": shape.baseline_round_wins,
                "round_ties": shape.round_ties,
            }));
        }

        let decision = if support[0] == run_count {
            counts[0] += 1;
            Decision::CandidatePreferred
        } else if support[1] == run_count {
            counts[1] += 1;
            Decision::BaselinePreferred
        } else {
            counts[2] += 1;
            Decision::Inconclusive
        };
        let block = match decision {
            Decision::CandidatePreferred => Some(reference.candidate_block_size),
            Decision::BaselinePreferred => Some(reference.baseline_block_size),
            Decision::Inconclusive => None,
        };

        recommendations.push(json!({
            "rows": rows,
            "cols": cols,
            "decision": decision.as_str(),
            "recommended_block_size": block,
            "candidate_support_runs": support[0],
            "baseline_support_runs": support[1],
            "inconclusive_runs": support[2],
            "min_median_paired_relative_improvement": min_improvement,
            "max_median_paired_relative_improvement": max_improvement,
            "runs": runs,
        }));
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
        "candidate_recommendations": counts[0],
        "baseline_recommendations": counts[1],
        "inconclusive": counts[2],
        "recommendations": recommendations,
        "claim_boundary": "consensus qualifies repeatability of microbenchmark policy evidence only; it does not authorize NNIS runtime promotion or imply end-to-end model speedup",
    }))
}

fn require_same_campaign(reference: &ParsedPolicy, policy: &ParsedPolicy) -> ConsensusResult<()> {
    if policy.baseline_block_size != reference.baseline_block_size
        || policy.candidate_block_size != reference.candidate_block_size
        || policy.rounds_per_shape != reference.rounds_per_shape
        || policy.shape_count != reference.shape_count
        || !policy.thresholds.matches(reference.thresholds)
    {
        return Err(ConsensusError(format!(
            "run {:?} campaign semantics differ from reference",
            policy.run_context_id
        )));
    }
    if policy
        .recommendations
        .keys()
        .ne(reference.recommendations.keys())
    {
        return Err(ConsensusError(format!(
            "run {:?} shape set differs from reference",
            policy.run_context_id
        )));
    }
    Ok(())
}

fn require_block(actual: Option<u64>, expected: u64, label: &str) -> ConsensusResult<()> {
    if actual != Some(expected) {
        return Err(ConsensusError(format!(
            "{label}: expected recommended block {expected}, got {actual:?}"
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

fn as_object<'a>(value: &'a Value, label: &str) -> ConsensusResult<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| ConsensusError(format!("{label} must be a JSON object")))
}

fn object_field<'a>(object: &'a Map<String, Value>, field: &str) -> ConsensusResult<&'a Map<String, Value>> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| ConsensusError(format!("{field} must be an object")))
}

fn array_field<'a>(object: &'a Map<String, Value>, field: &str) -> ConsensusResult<&'a Vec<Value>> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| ConsensusError(format!("{field} must be an array")))
}

fn string_field<'a>(object: &'a Map<String, Value>, field: &str) -> ConsensusResult<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ConsensusError(format!("{field} must be a string")))
}

fn require_u64(object: &Map<String, Value>, field: &str, expected: u64) -> ConsensusResult<()> {
    let actual = nonnegative_u64(object, field)?;
    if actual != expected {
        return Err(ConsensusError(format!(
            "{field} must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn require_str(object: &Map<String, Value>, field: &str, expected: &str) -> ConsensusResult<()> {
    let actual = string_field(object, field)?;
    if actual != expected {
        return Err(ConsensusError(format!(
            "{field} must be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn nonempty_string(object: &Map<String, Value>, field: &str) -> ConsensusResult<String> {
    let value = string_field(object, field)?;
    if value.trim().is_empty() {
        return Err(ConsensusError(format!("{field} must not be empty")));
    }
    Ok(value.to_string())
}

fn nonnegative_u64(object: &Map<String, Value>, field: &str) -> ConsensusResult<u64> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| ConsensusError(format!("{field} must be a non-negative integer")))
}

fn positive_u64(object: &Map<String, Value>, field: &str) -> ConsensusResult<u64> {
    let value = nonnegative_u64(object, field)?;
    if value == 0 {
        return Err(ConsensusError(format!("{field} must be positive")));
    }
    Ok(value)
}

fn optional_u64(object: &Map<String, Value>, field: &str) -> ConsensusResult<Option<u64>> {
    match object.get(field) {
        Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| ConsensusError(format!("{field} must be an integer or null"))),
        None => Err(ConsensusError(format!("missing field {field}"))),
    }
}

fn finite_f64(object: &Map<String, Value>, field: &str) -> ConsensusResult<f64> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| ConsensusError(format!("{field} must be numeric")))?;
    if !value.is_finite() {
        return Err(ConsensusError(format!("{field} must be finite")));
    }
    Ok(value)
}

fn to_usize(value: u64, field: &str) -> ConsensusResult<usize> {
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

    fn rec(rows: u64, cols: u64, decision: &str, improvement: f64) -> Value {
        let (block, candidate_wins, baseline_wins) = match decision {
            "candidate_preferred" => (Some(512), 4, 0),
            "baseline_preferred" => (Some(256), 0, 4),
            "inconclusive" => (None, 2, 2),
            _ => panic!("unsupported test decision"),
        };
        json!({
            "rows": rows,
            "cols": cols,
            "decision": decision,
            "recommended_block_size": block,
            "median_paired_relative_improvement": improvement,
            "candidate_round_wins": candidate_wins,
            "baseline_round_wins": baseline_wins,
            "round_ties": 0,
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

    fn parsed(run: &str, recommendation: Value) -> ParsedPolicy {
        parse_policy(&policy(run, vec![recommendation]), run).unwrap()
    }

    #[test]
    fn unanimous_candidate_is_recommended() {
        let first = parsed("run-a", rec(32, 4096, "candidate_preferred", 0.19));
        let second = parsed("run-b", rec(32, 4096, "candidate_preferred", 0.18));
        let consensus = build_consensus(&[first, second], 2).unwrap();
        assert_eq!(consensus["candidate_recommendations"], 1);
        assert_eq!(
            consensus["recommendations"][0]["decision"],
            "candidate_preferred"
        );
        assert_eq!(
            consensus["recommendations"][0]["recommended_block_size"],
            512
        );
    }

    #[test]
    fn disagreement_is_inconclusive() {
        let first = parsed("run-a", rec(128, 2048, "candidate_preferred", 0.05));
        let second = parsed("run-b", rec(128, 2048, "inconclusive", -0.01));
        let consensus = build_consensus(&[first, second], 2).unwrap();
        assert_eq!(consensus["inconclusive"], 1);
        assert_eq!(consensus["recommendations"][0]["decision"], "inconclusive");
        assert!(consensus["recommendations"][0]["recommended_block_size"].is_null());
    }

    #[test]
    fn duplicate_run_context_fails_closed() {
        let first = parsed("same-run", rec(1, 4096, "candidate_preferred", 0.20));
        let second = parsed("same-run", rec(1, 4096, "candidate_preferred", 0.21));
        assert!(build_consensus(&[first, second], 2).is_err());
    }

    #[test]
    fn mismatched_thresholds_fail_closed() {
        let first = parsed("run-a", rec(1, 4096, "candidate_preferred", 0.20));
        let mut value = policy(
            "run-b",
            vec![rec(1, 4096, "candidate_preferred", 0.20)],
        );
        value["thresholds"]["min_relative_improvement"] = json!(0.04);
        let second = parse_policy(&value, "run-b").unwrap();
        assert!(build_consensus(&[first, second], 2).is_err());
    }

    #[test]
    fn forged_decision_fails_closed() {
        let value = policy(
            "run-a",
            vec![rec(32, 2048, "candidate_preferred", 0.008)],
        );
        assert!(parse_policy(&value, "run-a").is_err());
    }

    #[test]
    fn mismatched_shape_set_fails_closed() {
        let first = parsed("run-a", rec(1, 4096, "candidate_preferred", 0.20));
        let second = parsed("run-b", rec(1, 8192, "candidate_preferred", 0.20));
        assert!(build_consensus(&[first, second], 2).is_err());
    }
}
