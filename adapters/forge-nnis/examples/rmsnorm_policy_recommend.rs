use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;

const SOURCE_SCHEMA_VERSION: u64 = 1;
const SOURCE_CAMPAIGN_KIND: &str = "forge_nnis_canonical_rmsnorm_shape_matrix_v1";
const POLICY_SCHEMA_VERSION: u64 = 1;
const POLICY_KIND: &str = "forge_nnis_rmsnorm_shape_policy_recommendation_v1";
const ORACLE_ID: &str = "forge-nnis/canonical-rmsnorm-f64-host-oracle-v1";
const DEFAULT_MIN_RELATIVE_IMPROVEMENT: f64 = 0.03;
const DEFAULT_MIN_ROUND_WIN_FRACTION: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    CandidatePreferred,
    BaselinePreferred,
    Inconclusive,
}

impl Decision {
    fn as_str(self) -> &'static str {
        match self {
            Self::CandidatePreferred => "candidate_preferred",
            Self::BaselinePreferred => "baseline_preferred",
            Self::Inconclusive => "inconclusive",
        }
    }
}

#[derive(Debug)]
struct PolicyError(String);

impl Display for PolicyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for PolicyError {}

type PolicyResult<T> = Result<T, PolicyError>;

#[derive(Clone, Copy, Debug)]
struct Thresholds {
    min_relative_improvement: f64,
    min_round_win_fraction: f64,
}

impl Thresholds {
    fn validate(self) -> PolicyResult<Self> {
        if !self.min_relative_improvement.is_finite()
            || self.min_relative_improvement <= 0.0
            || self.min_relative_improvement >= 1.0
        {
            return Err(PolicyError(
                "minimum relative improvement must be finite and in (0, 1)".to_string(),
            ));
        }
        if !self.min_round_win_fraction.is_finite()
            || self.min_round_win_fraction <= 0.5
            || self.min_round_win_fraction > 1.0
        {
            return Err(PolicyError(
                "minimum round win fraction must be finite and in (0.5, 1]".to_string(),
            ));
        }
        Ok(self)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: rmsnorm_policy_recommend <shape-matrix-evidence.json>")?;
    if std::env::args().nth(2).is_some() {
        return Err("usage: rmsnorm_policy_recommend <shape-matrix-evidence.json>".into());
    }
    let thresholds = Thresholds {
        min_relative_improvement: env_f64(
            "FORGE_NNIS_RMSNORM_POLICY_MIN_RELATIVE_IMPROVEMENT",
            DEFAULT_MIN_RELATIVE_IMPROVEMENT,
        )?,
        min_round_win_fraction: env_f64(
            "FORGE_NNIS_RMSNORM_POLICY_MIN_ROUND_WIN_FRACTION",
            DEFAULT_MIN_ROUND_WIN_FRACTION,
        )?,
    }
    .validate()?;
    let source: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let recommendation = recommend_policy(&source, thresholds)?;
    println!("{}", serde_json::to_string_pretty(&recommendation)?);
    Ok(())
}

fn recommend_policy(source: &Value, thresholds: Thresholds) -> PolicyResult<Value> {
    thresholds.validate()?;
    let root = object(source, "root")?;
    require_u64(root, "schema_version", SOURCE_SCHEMA_VERSION)?;
    require_str(root, "campaign_kind", SOURCE_CAMPAIGN_KIND)?;
    let run_context_id = nonempty_string(root, "run_context_id")?;
    let baseline_block_size = positive_u64(root, "baseline_block_size")?;
    let candidate_block_size = positive_u64(root, "candidate_block_size")?;
    if baseline_block_size == candidate_block_size {
        return Err(PolicyError(
            "baseline and candidate block sizes must differ".to_string(),
        ));
    }
    let rounds = usize_from_u64(
        positive_u64(root, "rounds_per_shape")?,
        "rounds_per_shape",
    )?;
    if rounds < 2 {
        return Err(PolicyError(
            "rounds_per_shape must be at least 2".to_string(),
        ));
    }
    let declared_shape_count =
        usize_from_u64(positive_u64(root, "shape_count")?, "shape_count")?;
    let results = array_field(root, "results")?;
    if results.len() != declared_shape_count {
        return Err(PolicyError(format!(
            "shape_count {declared_shape_count} does not match results length {}",
            results.len()
        )));
    }

    let declared_candidate_shape_wins = usize_from_u64(
        nonnegative_u64(root, "candidate_shape_wins")?,
        "candidate_shape_wins",
    )?;
    let declared_baseline_shape_wins = usize_from_u64(
        nonnegative_u64(root, "baseline_shape_wins")?,
        "baseline_shape_wins",
    )?;
    let declared_shape_ties =
        usize_from_u64(nonnegative_u64(root, "shape_ties")?, "shape_ties")?;
    if declared_candidate_shape_wins + declared_baseline_shape_wins + declared_shape_ties
        != declared_shape_count
    {
        return Err(PolicyError(
            "declared shape win/tie counts do not sum to shape_count".to_string(),
        ));
    }

    let mut seen_shapes = BTreeSet::new();
    let mut observed_candidate_shape_wins = 0usize;
    let mut observed_baseline_shape_wins = 0usize;
    let mut observed_shape_ties = 0usize;
    let mut candidate_recommendations = 0usize;
    let mut baseline_recommendations = 0usize;
    let mut inconclusive = 0usize;
    let mut recommendations = Vec::with_capacity(results.len());

    for (index, result) in results.iter().enumerate() {
        let result = object(result, &format!("results[{index}]"))?;
        let rows = positive_u64(result, "rows")?;
        let cols = positive_u64(result, "cols")?;
        if !seen_shapes.insert((rows, cols)) {
            return Err(PolicyError(format!(
                "duplicate shape {rows}x{cols} in results"
            )));
        }
        validate_verification(result, "baseline_verification")?;
        validate_verification(result, "candidate_verification")?;

        let candidate_round_wins = usize_from_u64(
            nonnegative_u64(result, "candidate_round_wins")?,
            "candidate_round_wins",
        )?;
        let baseline_round_wins = usize_from_u64(
            nonnegative_u64(result, "baseline_round_wins")?,
            "baseline_round_wins",
        )?;
        let round_ties =
            usize_from_u64(nonnegative_u64(result, "round_ties")?, "round_ties")?;
        if candidate_round_wins + baseline_round_wins + round_ties != rounds {
            return Err(PolicyError(format!(
                "shape {rows}x{cols} round wins/ties do not sum to rounds_per_shape"
            )));
        }

        let baseline_median_ms = positive_f64(result, "median_baseline_ms")?;
        let candidate_median_ms = positive_f64(result, "median_candidate_ms")?;
        let paired_improvement = finite_f64(result, "median_paired_relative_improvement")?;
        let speedup_ratio = positive_f64(result, "aggregate_microbenchmark_speedup_ratio")?;
        let derived_improvement = (baseline_median_ms - candidate_median_ms) / baseline_median_ms;
        let derived_ratio = baseline_median_ms / candidate_median_ms;
        require_close(
            paired_improvement,
            derived_improvement,
            5.0e-3,
            &format!("shape {rows}x{cols} relative improvement"),
        )?;
        require_close(
            speedup_ratio,
            derived_ratio,
            5.0e-3,
            &format!("shape {rows}x{cols} speedup ratio"),
        )?;

        match candidate_median_ms.total_cmp(&baseline_median_ms) {
            std::cmp::Ordering::Less => observed_candidate_shape_wins += 1,
            std::cmp::Ordering::Greater => observed_baseline_shape_wins += 1,
            std::cmp::Ordering::Equal => observed_shape_ties += 1,
        }

        let candidate_win_fraction = candidate_round_wins as f64 / rounds as f64;
        let baseline_win_fraction = baseline_round_wins as f64 / rounds as f64;
        let decision = if paired_improvement >= thresholds.min_relative_improvement
            && candidate_win_fraction >= thresholds.min_round_win_fraction
        {
            candidate_recommendations += 1;
            Decision::CandidatePreferred
        } else if paired_improvement <= -thresholds.min_relative_improvement
            && baseline_win_fraction >= thresholds.min_round_win_fraction
        {
            baseline_recommendations += 1;
            Decision::BaselinePreferred
        } else {
            inconclusive += 1;
            Decision::Inconclusive
        };
        let recommended_block_size = match decision {
            Decision::CandidatePreferred => Some(candidate_block_size),
            Decision::BaselinePreferred => Some(baseline_block_size),
            Decision::Inconclusive => None,
        };

        recommendations.push(json!({
            "rows": rows,
            "cols": cols,
            "decision": decision.as_str(),
            "recommended_block_size": recommended_block_size,
            "baseline_block_size": baseline_block_size,
            "candidate_block_size": candidate_block_size,
            "candidate_round_wins": candidate_round_wins,
            "baseline_round_wins": baseline_round_wins,
            "round_ties": round_ties,
            "candidate_round_win_fraction": candidate_win_fraction,
            "baseline_round_win_fraction": baseline_win_fraction,
            "median_baseline_ms": baseline_median_ms,
            "median_candidate_ms": candidate_median_ms,
            "median_paired_relative_improvement": paired_improvement,
            "aggregate_microbenchmark_speedup_ratio": speedup_ratio,
        }));
    }

    if observed_candidate_shape_wins != declared_candidate_shape_wins
        || observed_baseline_shape_wins != declared_baseline_shape_wins
        || observed_shape_ties != declared_shape_ties
    {
        return Err(PolicyError(
            "declared top-level shape win/tie counts disagree with per-shape medians".to_string(),
        ));
    }

    Ok(json!({
        "schema_version": POLICY_SCHEMA_VERSION,
        "recommendation_kind": POLICY_KIND,
        "source_campaign": {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "campaign_kind": SOURCE_CAMPAIGN_KIND,
            "run_context_id": run_context_id,
            "shape_count": declared_shape_count,
            "rounds_per_shape": rounds,
            "baseline_block_size": baseline_block_size,
            "candidate_block_size": candidate_block_size,
        },
        "thresholds": {
            "min_relative_improvement": thresholds.min_relative_improvement,
            "min_round_win_fraction": thresholds.min_round_win_fraction,
        },
        "candidate_recommendations": candidate_recommendations,
        "baseline_recommendations": baseline_recommendations,
        "inconclusive": inconclusive,
        "recommendations": recommendations,
        "inconclusive_semantics": "no performance winner is asserted; destination runtime should retain its existing qualified behavior unless separately promoted",
        "claim_boundary": "Forge recommendation derived from one validated shape-matrix campaign only; this manifest does not modify NNIS, does not establish cross-device portability, and does not authorize production promotion",
    }))
}

fn validate_verification(result: &serde_json::Map<String, Value>, field: &str) -> PolicyResult<()> {
    let verification = object_field(result, field)?;
    if verification.get("passed").and_then(Value::as_bool) != Some(true) {
        return Err(PolicyError(format!("{field}.passed must be true")));
    }
    require_str(verification, "oracle_id", ORACLE_ID)?;
    let max_abs_error = finite_f64(verification, "max_abs_error")?;
    let max_rel_error = finite_f64(verification, "max_rel_error")?;
    if max_abs_error < 0.0 || max_rel_error < 0.0 {
        return Err(PolicyError(format!(
            "{field} error maxima must be non-negative"
        )));
    }
    Ok(())
}

fn object<'a>(value: &'a Value, label: &str) -> PolicyResult<&'a serde_json::Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| PolicyError(format!("{label} must be a JSON object")))
}

fn object_field<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> PolicyResult<&'a serde_json::Map<String, Value>> {
    object
        .get(field)
        .ok_or_else(|| PolicyError(format!("missing {field}")))?
        .as_object()
        .ok_or_else(|| PolicyError(format!("{field} must be a JSON object")))
}

fn array_field<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> PolicyResult<&'a Vec<Value>> {
    object
        .get(field)
        .ok_or_else(|| PolicyError(format!("missing {field}")))?
        .as_array()
        .ok_or_else(|| PolicyError(format!("{field} must be a JSON array")))
}

fn require_u64(
    object: &serde_json::Map<String, Value>,
    field: &str,
    expected: u64,
) -> PolicyResult<()> {
    let actual = nonnegative_u64(object, field)?;
    if actual != expected {
        return Err(PolicyError(format!(
            "{field} must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn require_str(
    object: &serde_json::Map<String, Value>,
    field: &str,
    expected: &str,
) -> PolicyResult<()> {
    let actual = object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| PolicyError(format!("{field} must be a string")))?;
    if actual != expected {
        return Err(PolicyError(format!(
            "{field} must be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn nonempty_string(object: &serde_json::Map<String, Value>, field: &str) -> PolicyResult<String> {
    let value = object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| PolicyError(format!("{field} must be a string")))?;
    if value.trim().is_empty() {
        return Err(PolicyError(format!("{field} must not be empty")));
    }
    Ok(value.to_string())
}

fn nonnegative_u64(object: &serde_json::Map<String, Value>, field: &str) -> PolicyResult<u64> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| PolicyError(format!("{field} must be a non-negative integer")))
}

fn positive_u64(object: &serde_json::Map<String, Value>, field: &str) -> PolicyResult<u64> {
    let value = nonnegative_u64(object, field)?;
    if value == 0 {
        return Err(PolicyError(format!("{field} must be positive")));
    }
    Ok(value)
}

fn usize_from_u64(value: u64, field: &str) -> PolicyResult<usize> {
    usize::try_from(value).map_err(|_| PolicyError(format!("{field} does not fit usize")))
}

fn finite_f64(object: &serde_json::Map<String, Value>, field: &str) -> PolicyResult<f64> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| PolicyError(format!("{field} must be numeric")))?;
    if !value.is_finite() {
        return Err(PolicyError(format!("{field} must be finite")));
    }
    Ok(value)
}

fn positive_f64(object: &serde_json::Map<String, Value>, field: &str) -> PolicyResult<f64> {
    let value = finite_f64(object, field)?;
    if value <= 0.0 {
        return Err(PolicyError(format!("{field} must be positive")));
    }
    Ok(value)
}

fn require_close(actual: f64, expected: f64, tolerance: f64, label: &str) -> PolicyResult<()> {
    if (actual - expected).abs() > tolerance {
        return Err(PolicyError(format!(
            "{label} is inconsistent: recorded {actual}, derived {expected}"
        )));
    }
    Ok(())
}

fn env_f64(name: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse::<f64>())
        .transpose()?
        .unwrap_or(default))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verification(passed: bool) -> Value {
        json!({
            "passed": passed,
            "oracle_id": ORACLE_ID,
            "max_abs_error": 1.0e-7,
            "max_rel_error": 1.0e-7,
        })
    }

    fn result(
        rows: u64,
        cols: u64,
        baseline_ms: f64,
        candidate_ms: f64,
        candidate_wins: u64,
        baseline_wins: u64,
        ties: u64,
    ) -> Value {
        let improvement = (baseline_ms - candidate_ms) / baseline_ms;
        json!({
            "rows": rows,
            "cols": cols,
            "candidate_round_wins": candidate_wins,
            "baseline_round_wins": baseline_wins,
            "round_ties": ties,
            "median_baseline_ms": baseline_ms,
            "median_candidate_ms": candidate_ms,
            "median_paired_relative_improvement": improvement,
            "aggregate_microbenchmark_speedup_ratio": baseline_ms / candidate_ms,
            "baseline_verification": verification(true),
            "candidate_verification": verification(true),
        })
    }

    fn source(
        results: Vec<Value>,
        candidate_shape_wins: u64,
        baseline_shape_wins: u64,
        ties: u64,
    ) -> Value {
        json!({
            "schema_version": SOURCE_SCHEMA_VERSION,
            "campaign_kind": SOURCE_CAMPAIGN_KIND,
            "run_context_id": "test-run",
            "baseline_block_size": 256,
            "candidate_block_size": 512,
            "shape_count": results.len(),
            "rounds_per_shape": 4,
            "candidate_shape_wins": candidate_shape_wins,
            "baseline_shape_wins": baseline_shape_wins,
            "shape_ties": ties,
            "results": results,
        })
    }

    fn defaults() -> Thresholds {
        Thresholds {
            min_relative_improvement: DEFAULT_MIN_RELATIVE_IMPROVEMENT,
            min_round_win_fraction: DEFAULT_MIN_ROUND_WIN_FRACTION,
        }
    }

    #[test]
    fn recommends_candidate_only_with_material_unanimous_win() {
        let evidence = source(
            vec![result(32, 4096, 0.010, 0.008, 4, 0, 0)],
            1,
            0,
            0,
        );
        let recommendation = recommend_policy(&evidence, defaults()).unwrap();
        assert_eq!(recommendation["candidate_recommendations"], 1);
        assert_eq!(recommendation["baseline_recommendations"], 0);
        assert_eq!(recommendation["inconclusive"], 0);
        assert_eq!(
            recommendation["recommendations"][0]["decision"],
            "candidate_preferred"
        );
        assert_eq!(
            recommendation["recommendations"][0]["recommended_block_size"],
            512
        );
    }

    #[test]
    fn small_margin_remains_inconclusive_even_with_all_round_wins() {
        let evidence = source(
            vec![result(32, 2048, 0.008224, 0.008160, 4, 0, 0)],
            1,
            0,
            0,
        );
        let recommendation = recommend_policy(&evidence, defaults()).unwrap();
        assert_eq!(recommendation["candidate_recommendations"], 0);
        assert_eq!(recommendation["inconclusive"], 1);
        assert_eq!(
            recommendation["recommendations"][0]["decision"],
            "inconclusive"
        );
        assert!(recommendation["recommendations"][0]["recommended_block_size"].is_null());
    }

    #[test]
    fn material_unanimous_regression_can_prefer_baseline() {
        let evidence = source(
            vec![result(128, 2048, 0.010, 0.011, 0, 4, 0)],
            0,
            1,
            0,
        );
        let recommendation = recommend_policy(&evidence, defaults()).unwrap();
        assert_eq!(recommendation["baseline_recommendations"], 1);
        assert_eq!(
            recommendation["recommendations"][0]["decision"],
            "baseline_preferred"
        );
        assert_eq!(
            recommendation["recommendations"][0]["recommended_block_size"],
            256
        );
    }

    #[test]
    fn failed_verification_is_rejected_before_recommendation() {
        let mut row = result(1, 4096, 0.010, 0.008, 4, 0, 0);
        row["candidate_verification"] = verification(false);
        let evidence = source(vec![row], 1, 0, 0);
        assert!(recommend_policy(&evidence, defaults()).is_err());
    }

    #[test]
    fn inconsistent_derived_metrics_fail_closed() {
        let mut row = result(1, 4096, 0.010, 0.008, 4, 0, 0);
        row["aggregate_microbenchmark_speedup_ratio"] = json!(9.0);
        let evidence = source(vec![row], 1, 0, 0);
        assert!(recommend_policy(&evidence, defaults()).is_err());
    }

    #[test]
    fn top_level_counts_must_match_per_shape_medians() {
        let evidence = source(vec![result(1, 4096, 0.010, 0.008, 4, 0, 0)], 0, 1, 0);
        assert!(recommend_policy(&evidence, defaults()).is_err());
    }
}
