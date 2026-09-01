use serde_json::{json, Map, Value};
use std::cmp::Ordering;
use std::error::Error;
use std::fs;

const CAMPAIGN_KIND: &str = "forge_nnis_smollm2_weighted_rmsnorm_launch_requalification_v1";
const RECOMMENDATION_KIND: &str = "forge_nnis_smollm2_weighted_rmsnorm_policy_recommendation_v1";
const DEFAULT_MIN_RELATIVE_IMPROVEMENT: f64 = 0.03;
const DEFAULT_MIN_NON_TIE_WIN_FRACTION: f64 = 1.0;

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

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("weighted_rmsnorm_policy_recommend requires one campaign JSON path")?;
    let min_relative_improvement = env_f64(
        "FORGE_NNIS_WEIGHTED_RMSNORM_POLICY_MIN_RELATIVE_IMPROVEMENT",
        DEFAULT_MIN_RELATIVE_IMPROVEMENT,
    )?;
    let min_non_tie_win_fraction = env_f64(
        "FORGE_NNIS_WEIGHTED_RMSNORM_POLICY_MIN_NON_TIE_WIN_FRACTION",
        DEFAULT_MIN_NON_TIE_WIN_FRACTION,
    )?;
    validate_thresholds(min_relative_improvement, min_non_tie_win_fraction)?;

    let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let recommendation = reduce_campaign(
        &value,
        min_relative_improvement,
        min_non_tie_win_fraction,
    )?;
    println!("{}", serde_json::to_string_pretty(&recommendation)?);
    Ok(())
}

fn reduce_campaign(
    value: &Value,
    min_relative_improvement: f64,
    min_non_tie_win_fraction: f64,
) -> Result<Value, Box<dyn Error>> {
    validate_thresholds(min_relative_improvement, min_non_tie_win_fraction)?;
    let root = object(value, "campaign")?;
    require_u64(root, "schema_version", 1)?;
    require_str(root, "campaign_kind", CAMPAIGN_KIND)?;
    let run_context_id = nonempty_str(root, "run_context_id")?;

    let model_scope = object_field(root, "model_scope")?;
    require_str(model_scope, "model", "HuggingFaceTB/SmolLM2-135M")?;
    require_u64(model_scope, "hidden_size", 576)?;
    require_u64(model_scope, "runtime_rows", 1)?;
    require_u64(model_scope, "num_hidden_layers", 30)?;
    require_u64(model_scope, "weighted_rmsnorm_launches_per_token", 61)?;
    let rms_norm_eps = finite_f64(model_scope, "rms_norm_eps")?;
    if (rms_norm_eps - 1.0e-5).abs() > 1.0e-10 {
        return Err("unexpected SmolLM2 RMSNorm epsilon".into());
    }

    let baseline = object_field(root, "baseline")?;
    require_str(
        baseline,
        "implementation",
        "nnis-model/F32DecoderKernels::weighted_rms_norm",
    )?;
    let baseline_block_size = positive_u64(baseline, "block_size")?;
    if baseline_block_size != 256 {
        return Err("weighted RMSNorm baseline must use block 256".into());
    }

    let candidate = object_field(root, "candidate")?;
    require_str(
        candidate,
        "implementation",
        "nnis-model/F32WeightedRmsNormCandidate",
    )?;
    let candidate_block_size = positive_u64(candidate, "block_size")?;
    if candidate_block_size != 512 {
        return Err("weighted RMSNorm candidate must use block 512".into());
    }

    require_verification_passed(root, "baseline_verification")?;
    require_verification_passed(root, "candidate_verification")?;

    let rounds = usize::try_from(positive_u64(root, "rounds")?)?;
    if rounds < 2 {
        return Err("weighted RMSNorm policy requires at least two paired rounds".into());
    }
    let observations = array_field(root, "observations")?;
    if observations.len() != rounds {
        return Err("observation count does not match rounds".into());
    }

    let mut candidate_wins = 0usize;
    let mut baseline_wins = 0usize;
    let mut ties = 0usize;
    let mut baseline_medians = Vec::with_capacity(rounds);
    let mut candidate_medians = Vec::with_capacity(rounds);
    let mut paired_improvements = Vec::with_capacity(rounds);

    for (index, observation) in observations.iter().enumerate() {
        let observation = object(observation, "observation")?;
        require_u64(observation, "round", index as u64)?;
        let expected_order = if index.is_multiple_of(2) {
            "baseline_then_candidate"
        } else {
            "candidate_then_baseline"
        };
        require_str(observation, "order", expected_order)?;

        let baseline_median = positive_f64(observation, "baseline_median_ms")?;
        let candidate_median = positive_f64(observation, "candidate_median_ms")?;
        let reported_improvement = finite_f64(observation, "relative_improvement")?;
        let derived_improvement = (baseline_median - candidate_median) / baseline_median;
        if (reported_improvement - derived_improvement).abs() > 1.0e-12 {
            return Err("observation relative improvement is inconsistent with medians".into());
        }

        match candidate_median.total_cmp(&baseline_median) {
            Ordering::Less => candidate_wins += 1,
            Ordering::Greater => baseline_wins += 1,
            Ordering::Equal => ties += 1,
        }
        baseline_medians.push(baseline_median);
        candidate_medians.push(candidate_median);
        paired_improvements.push(derived_improvement);
    }

    require_u64(root, "candidate_round_wins", candidate_wins as u64)?;
    require_u64(root, "baseline_round_wins", baseline_wins as u64)?;
    require_u64(root, "round_ties", ties as u64)?;

    let median_baseline = median(&baseline_medians)?;
    let median_candidate = median(&candidate_medians)?;
    let median_improvement = median(&paired_improvements)?;
    require_approx(root, "median_baseline_ms", median_baseline, 1.0e-12)?;
    require_approx(root, "median_candidate_ms", median_candidate, 1.0e-12)?;
    require_approx(
        root,
        "median_paired_relative_improvement",
        median_improvement,
        1.0e-12,
    )?;
    require_approx(
        root,
        "aggregate_microbenchmark_speedup_ratio",
        median_baseline / median_candidate,
        1.0e-12,
    )?;

    let non_ties = candidate_wins + baseline_wins;
    let candidate_win_fraction = if non_ties == 0 {
        0.0
    } else {
        candidate_wins as f64 / non_ties as f64
    };
    let baseline_win_fraction = if non_ties == 0 {
        0.0
    } else {
        baseline_wins as f64 / non_ties as f64
    };

    let decision = if median_improvement >= min_relative_improvement
        && candidate_win_fraction >= min_non_tie_win_fraction
    {
        Decision::CandidatePreferred
    } else if median_improvement <= -min_relative_improvement
        && baseline_win_fraction >= min_non_tie_win_fraction
    {
        Decision::BaselinePreferred
    } else {
        Decision::Inconclusive
    };

    let recommended_block_size = match decision {
        Decision::CandidatePreferred => Some(candidate_block_size),
        Decision::BaselinePreferred => Some(baseline_block_size),
        Decision::Inconclusive => None,
    };

    Ok(json!({
        "schema_version": 1,
        "recommendation_kind": RECOMMENDATION_KIND,
        "source_campaign": {
            "schema_version": 1,
            "campaign_kind": CAMPAIGN_KIND,
            "run_context_id": run_context_id,
            "model": "HuggingFaceTB/SmolLM2-135M",
            "rows": 1,
            "cols": 576,
            "baseline_block_size": baseline_block_size,
            "candidate_block_size": candidate_block_size,
            "rounds": rounds,
        },
        "thresholds": {
            "min_relative_improvement": min_relative_improvement,
            "min_non_tie_win_fraction": min_non_tie_win_fraction,
        },
        "decision": decision.as_str(),
        "recommended_block_size": recommended_block_size,
        "evidence_summary": {
            "candidate_round_wins": candidate_wins,
            "baseline_round_wins": baseline_wins,
            "round_ties": ties,
            "candidate_non_tie_win_fraction": candidate_win_fraction,
            "baseline_non_tie_win_fraction": baseline_win_fraction,
            "median_baseline_ms": median_baseline,
            "median_candidate_ms": median_candidate,
            "median_paired_relative_improvement": median_improvement,
            "aggregate_microbenchmark_speedup_ratio": median_baseline / median_candidate,
        },
        "claim_boundary": "policy decision covers the isolated NNIS SmolLM2 weighted-RMSNorm 1x576 launch comparison only; inconclusive does not authorize runtime promotion and no decision implies end-to-end model speedup",
    }))
}

fn validate_thresholds(
    min_relative_improvement: f64,
    min_non_tie_win_fraction: f64,
) -> Result<(), Box<dyn Error>> {
    if !min_relative_improvement.is_finite()
        || min_relative_improvement <= 0.0
        || min_relative_improvement >= 1.0
    {
        return Err("minimum relative improvement must be finite and in (0, 1)".into());
    }
    if !min_non_tie_win_fraction.is_finite()
        || min_non_tie_win_fraction <= 0.5
        || min_non_tie_win_fraction > 1.0
    {
        return Err("minimum non-tie win fraction must be finite and in (0.5, 1]".into());
    }
    Ok(())
}

fn require_verification_passed(
    root: &Map<String, Value>,
    field: &str,
) -> Result<(), Box<dyn Error>> {
    let verification = object_field(root, field)?;
    if verification.get("passed").and_then(Value::as_bool) != Some(true) {
        return Err(format!("{field} must pass").into());
    }
    let max_abs_error = finite_f64(verification, "max_abs_error")?;
    let max_rel_error = finite_f64(verification, "max_rel_error")?;
    if max_abs_error < 0.0 || max_rel_error < 0.0 {
        return Err(format!("{field} errors must be non-negative").into());
    }
    Ok(())
}

fn median(values: &[f64]) -> Result<f64, Box<dyn Error>> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("median requires non-empty finite values".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let midpoint = sorted.len() / 2;
    Ok(if sorted.len().is_multiple_of(2) {
        (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
    } else {
        sorted[midpoint]
    })
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, Box<dyn Error>> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object").into())
}

fn object_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a Map<String, Value>, Box<dyn Error>> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{field} must be an object").into())
}

fn array_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a Vec<Value>, Box<dyn Error>> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{field} must be an array").into())
}

fn nonempty_str<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a str, Box<dyn Error>> {
    let value = object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))?;
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty").into());
    }
    Ok(value)
}

fn require_str(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
) -> Result<(), Box<dyn Error>> {
    let value = nonempty_str(object, field)?;
    if value != expected {
        return Err(format!("{field} must be {expected:?}, got {value:?}").into());
    }
    Ok(())
}

fn require_u64(
    object: &Map<String, Value>,
    field: &str,
    expected: u64,
) -> Result<(), Box<dyn Error>> {
    let value = object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{field} must be a non-negative integer"))?;
    if value != expected {
        return Err(format!("{field} must be {expected}, got {value}").into());
    }
    Ok(())
}

fn positive_u64(object: &Map<String, Value>, field: &str) -> Result<u64, Box<dyn Error>> {
    let value = object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{field} must be a positive integer"))?;
    if value == 0 {
        return Err(format!("{field} must be positive").into());
    }
    Ok(value)
}

fn finite_f64(object: &Map<String, Value>, field: &str) -> Result<f64, Box<dyn Error>> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{field} must be numeric"))?;
    if !value.is_finite() {
        return Err(format!("{field} must be finite").into());
    }
    Ok(value)
}

fn positive_f64(object: &Map<String, Value>, field: &str) -> Result<f64, Box<dyn Error>> {
    let value = finite_f64(object, field)?;
    if value <= 0.0 {
        return Err(format!("{field} must be positive").into());
    }
    Ok(value)
}

fn require_approx(
    object: &Map<String, Value>,
    field: &str,
    expected: f64,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let value = finite_f64(object, field)?;
    if (value - expected).abs() > tolerance {
        return Err(format!("{field} is inconsistent with raw observations").into());
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

    fn verification() -> Value {
        json!({
            "passed": true,
            "max_abs_error": 1.0e-7,
            "max_rel_error": 1.0e-7,
        })
    }

    fn campaign(candidate_medians: &[f64], baseline_medians: &[f64]) -> Value {
        assert_eq!(candidate_medians.len(), baseline_medians.len());
        let observations = baseline_medians
            .iter()
            .zip(candidate_medians)
            .enumerate()
            .map(|(round, (&baseline, &candidate))| {
                json!({
                    "round": round,
                    "order": if round.is_multiple_of(2) {
                        "baseline_then_candidate"
                    } else {
                        "candidate_then_baseline"
                    },
                    "baseline_median_ms": baseline,
                    "candidate_median_ms": candidate,
                    "relative_improvement": (baseline - candidate) / baseline,
                })
            })
            .collect::<Vec<_>>();
        let candidate_wins = candidate_medians
            .iter()
            .zip(baseline_medians)
            .filter(|(candidate, baseline)| candidate < baseline)
            .count();
        let baseline_wins = candidate_medians
            .iter()
            .zip(baseline_medians)
            .filter(|(candidate, baseline)| candidate > baseline)
            .count();
        let ties = candidate_medians.len() - candidate_wins - baseline_wins;
        let improvements = baseline_medians
            .iter()
            .zip(candidate_medians)
            .map(|(&baseline, &candidate)| (baseline - candidate) / baseline)
            .collect::<Vec<_>>();
        json!({
            "schema_version": 1,
            "campaign_kind": CAMPAIGN_KIND,
            "run_context_id": "test-run",
            "model_scope": {
                "model": "HuggingFaceTB/SmolLM2-135M",
                "hidden_size": 576,
                "runtime_rows": 1,
                "num_hidden_layers": 30,
                "weighted_rmsnorm_launches_per_token": 61,
                "rms_norm_eps": 1.0e-5,
            },
            "baseline": {
                "implementation": "nnis-model/F32DecoderKernels::weighted_rms_norm",
                "block_size": 256,
            },
            "candidate": {
                "implementation": "nnis-model/F32WeightedRmsNormCandidate",
                "block_size": 512,
            },
            "baseline_verification": verification(),
            "candidate_verification": verification(),
            "rounds": observations.len(),
            "candidate_round_wins": candidate_wins,
            "baseline_round_wins": baseline_wins,
            "round_ties": ties,
            "median_baseline_ms": median(baseline_medians).unwrap(),
            "median_candidate_ms": median(candidate_medians).unwrap(),
            "median_paired_relative_improvement": median(&improvements).unwrap(),
            "aggregate_microbenchmark_speedup_ratio": median(baseline_medians).unwrap()
                / median(candidate_medians).unwrap(),
            "observations": observations,
        })
    }

    #[test]
    fn small_effect_is_inconclusive_even_with_unanimous_non_tie_wins() {
        let value = campaign(&[0.995, 0.995, 1.0, 0.995], &[1.0; 4]);
        let result = reduce_campaign(&value, 0.03, 1.0).unwrap();
        assert_eq!(result["decision"], "inconclusive");
        assert!(result["recommended_block_size"].is_null());
    }

    #[test]
    fn strong_unanimous_candidate_effect_is_preferred() {
        let value = campaign(&[0.95; 4], &[1.0; 4]);
        let result = reduce_campaign(&value, 0.03, 1.0).unwrap();
        assert_eq!(result["decision"], "candidate_preferred");
        assert_eq!(result["recommended_block_size"], 512);
    }

    #[test]
    fn failed_verification_fails_closed() {
        let mut value = campaign(&[0.95; 4], &[1.0; 4]);
        value["candidate_verification"]["passed"] = json!(false);
        assert!(reduce_campaign(&value, 0.03, 1.0).is_err());
    }

    #[test]
    fn tampered_top_level_metric_fails_closed() {
        let mut value = campaign(&[0.95; 4], &[1.0; 4]);
        value["median_paired_relative_improvement"] = json!(0.99);
        assert!(reduce_campaign(&value, 0.03, 1.0).is_err());
    }
}
