use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;

const SOURCE_SCHEMA_VERSION: u64 = 1;
const SOURCE_CAMPAIGN_KIND: &str = "forge_nnis_canonical_rmsnorm_shape_matrix_v1";
const CONSENSUS_SCHEMA_VERSION: u64 = 1;
const CONSENSUS_KIND: &str = "forge_nnis_rmsnorm_environment_consensus_v1";
const CASE_NAME: &str = "nnis_f32_rmsnorm_fused";
const CASE_DTYPE: &str = "f32";
const DEFAULT_MIN_RUNS: usize = 2;

#[derive(Debug)]
struct EnvironmentError(String);

impl Display for EnvironmentError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for EnvironmentError {}

type EnvironmentResult<T> = Result<T, EnvironmentError>;

#[derive(Clone, Debug)]
struct CampaignSpec {
    identity: Value,
    shapes: BTreeSet<(u64, u64)>,
    shape_count: usize,
    rounds: usize,
    warmups: u64,
    iterations: u64,
    baseline_block_size: u64,
    candidate_block_size: u64,
}

#[derive(Clone, Debug)]
struct ParsedRun {
    run_context_id: String,
    campaign_identity: Value,
    environment_identity: Value,
    metadata_record_count: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let paths = std::env::args().skip(1).collect::<Vec<_>>();
    let min_runs = env_usize(
        "FORGE_NNIS_RMSNORM_ENVIRONMENT_CONSENSUS_MIN_RUNS",
        DEFAULT_MIN_RUNS,
    )?;
    if min_runs < 2 {
        return Err("environment consensus requires at least two runs".into());
    }
    if paths.len() < min_runs {
        return Err(format!(
            "rmsnorm_environment_consensus requires at least {min_runs} raw shape-matrix files"
        )
        .into());
    }

    let mut runs = Vec::with_capacity(paths.len());
    for path in paths {
        let source: Value = serde_json::from_str(&fs::read_to_string(&path)?)?;
        runs.push(parse_run(&source, &path)?);
    }

    let consensus = build_consensus(&runs, min_runs)?;
    println!("{}", serde_json::to_string_pretty(&consensus)?);
    Ok(())
}

fn parse_run(source: &Value, label: &str) -> EnvironmentResult<ParsedRun> {
    let root = object(source, label)?;
    require_u64(root, "schema_version", SOURCE_SCHEMA_VERSION)?;
    require_str(root, "campaign_kind", SOURCE_CAMPAIGN_KIND)?;
    let run_context_id = nonempty_string(root, "run_context_id")?;
    let campaign = parse_campaign_spec(root, label)?;
    let results = array_field(root, "results")?;
    if results.len() != campaign.shape_count {
        return Err(EnvironmentError(format!(
            "{label}: shape_count {} does not match results length {}",
            campaign.shape_count,
            results.len()
        )));
    }

    let mut expected_environment: Option<Value> = None;
    let mut metadata_record_count = 0usize;
    let mut seen_shapes = BTreeSet::new();
    for (result_index, result) in results.iter().enumerate() {
        let result_label = format!("{label}.results[{result_index}]");
        let result = object(result, &result_label)?;
        let rows = positive_u64(result, "rows")?;
        let cols = positive_u64(result, "cols")?;
        if !campaign.shapes.contains(&(rows, cols)) {
            return Err(EnvironmentError(format!(
                "{result_label}: shape {rows}x{cols} is not declared by the campaign"
            )));
        }
        if !seen_shapes.insert((rows, cols)) {
            return Err(EnvironmentError(format!(
                "{result_label}: duplicate shape {rows}x{cols}"
            )));
        }

        let observations = array_field(result, "observations")?;
        if observations.len() != campaign.rounds {
            return Err(EnvironmentError(format!(
                "{result_label}: observations length {} does not match rounds_per_shape {}",
                observations.len(),
                campaign.rounds
            )));
        }
        for (observation_index, observation) in observations.iter().enumerate() {
            let observation_label = format!("{result_label}.observations[{observation_index}]");
            let observation = object(observation, &observation_label)?;
            for (report_field, block_size) in [
                ("baseline_report", campaign.baseline_block_size),
                ("candidate_report", campaign.candidate_block_size),
            ] {
                let report = object_field(observation, report_field)?;
                validate_report(
                    report,
                    rows,
                    cols,
                    block_size,
                    campaign.warmups,
                    campaign.iterations,
                    &observation_label,
                )?;
                let metadata = object_field(report, "metadata")?;
                let identity = environment_identity(metadata, &run_context_id)?;
                match &expected_environment {
                    Some(expected) if expected != &identity => {
                        return Err(EnvironmentError(format!(
                            "{label}: stable environment identity changed within one campaign"
                        )));
                    }
                    Some(_) => {}
                    None => expected_environment = Some(identity),
                }
                metadata_record_count = metadata_record_count.checked_add(1).ok_or_else(|| {
                    EnvironmentError("metadata record count overflows".to_string())
                })?;
            }
        }
    }
    if seen_shapes != campaign.shapes {
        return Err(EnvironmentError(format!(
            "{label}: result shape set does not match declared campaign shapes"
        )));
    }

    let environment_identity = expected_environment.ok_or_else(|| {
        EnvironmentError(format!(
            "{label}: no benchmark metadata was found in the campaign"
        ))
    })?;
    Ok(ParsedRun {
        run_context_id,
        campaign_identity: campaign.identity,
        environment_identity,
        metadata_record_count,
    })
}

fn parse_campaign_spec(root: &Map<String, Value>, label: &str) -> EnvironmentResult<CampaignSpec> {
    let baseline_block_size = positive_u64(root, "baseline_block_size")?;
    let candidate_block_size = positive_u64(root, "candidate_block_size")?;
    if baseline_block_size == candidate_block_size {
        return Err(EnvironmentError(format!(
            "{label}: baseline and candidate block sizes must differ"
        )));
    }
    let shape_count = usize_from_u64(positive_u64(root, "shape_count")?, "shape_count")?;
    let shape_values = array_field(root, "shapes")?;
    if shape_values.len() != shape_count {
        return Err(EnvironmentError(format!(
            "{label}: shape_count {shape_count} does not match shapes length {}",
            shape_values.len()
        )));
    }
    let mut shapes = BTreeSet::new();
    let mut canonical_shapes = Vec::with_capacity(shape_count);
    for (index, shape) in shape_values.iter().enumerate() {
        let shape = object(shape, &format!("{label}.shapes[{index}]"))?;
        let rows = positive_u64(shape, "rows")?;
        let cols = positive_u64(shape, "cols")?;
        if !shapes.insert((rows, cols)) {
            return Err(EnvironmentError(format!(
                "{label}: duplicate declared shape {rows}x{cols}"
            )));
        }
        canonical_shapes.push(json!({"rows": rows, "cols": cols}));
    }

    let rounds = usize_from_u64(positive_u64(root, "rounds_per_shape")?, "rounds_per_shape")?;
    if rounds < 2 {
        return Err(EnvironmentError(format!(
            "{label}: rounds_per_shape must be at least 2"
        )));
    }
    let warmups = nonnegative_u64(root, "warmup_iterations_per_observation")?;
    let iterations = positive_u64(root, "measured_iterations_per_observation")?;
    let epsilon = finite_f64(root, "epsilon")?;
    let gamma = finite_f64(root, "gamma")?;
    let atol = finite_f64(root, "atol")?;
    let rtol = finite_f64(root, "rtol")?;
    if epsilon <= 0.0 {
        return Err(EnvironmentError(format!(
            "{label}: epsilon must be positive"
        )));
    }
    if atol < 0.0 || rtol < 0.0 {
        return Err(EnvironmentError(format!(
            "{label}: tolerances must be non-negative"
        )));
    }

    Ok(CampaignSpec {
        identity: json!({
            "schema_version": SOURCE_SCHEMA_VERSION,
            "campaign_kind": SOURCE_CAMPAIGN_KIND,
            "baseline_block_size": baseline_block_size,
            "candidate_block_size": candidate_block_size,
            "shape_count": shape_count,
            "shapes": canonical_shapes,
            "rounds_per_shape": rounds,
            "warmup_iterations_per_observation": warmups,
            "measured_iterations_per_observation": iterations,
            "epsilon": epsilon,
            "gamma": gamma,
            "atol": atol,
            "rtol": rtol,
        }),
        shapes,
        shape_count,
        rounds,
        warmups,
        iterations,
        baseline_block_size,
        candidate_block_size,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_report(
    report: &Map<String, Value>,
    rows: u64,
    cols: u64,
    block_size: u64,
    warmups: u64,
    iterations: u64,
    label: &str,
) -> EnvironmentResult<()> {
    let case = object_field(report, "case")?;
    require_str(case, "name", CASE_NAME)?;
    require_str(case, "dtype", CASE_DTYPE)?;
    let dimensions = object_field(case, "dimensions")?;
    require_u64(dimensions, "rows", rows)?;
    require_u64(dimensions, "cols", cols)?;
    require_u64(dimensions, "block_size", block_size)?;

    let config = object_field(report, "config")?;
    require_u64(config, "warmup_iterations", warmups)?;
    require_u64(config, "iterations", iterations)?;
    if !report.contains_key("metadata") {
        return Err(EnvironmentError(format!(
            "{label}: report metadata is missing"
        )));
    }
    Ok(())
}

fn environment_identity(
    metadata: &Map<String, Value>,
    expected_run_context_id: &str,
) -> EnvironmentResult<Value> {
    if bool_field(metadata, "git_dirty")? {
        return Err(EnvironmentError(
            "benchmark metadata must come from a clean git tree".to_string(),
        ));
    }
    let _timestamp = nonnegative_u64(metadata, "unix_timestamp_seconds")?;
    let fingerprint = object_field(metadata, "environment_fingerprint")?;
    require_u64(fingerprint, "schema_version", 1)?;
    let metadata_run_context_id = nonempty_string(fingerprint, "run_context_id")?;
    if metadata_run_context_id != expected_run_context_id {
        return Err(EnvironmentError(format!(
            "benchmark metadata run_context_id {metadata_run_context_id:?} does not match campaign {expected_run_context_id:?}"
        )));
    }

    Ok(json!({
        "compute_capability_major": nonnegative_u64(metadata, "compute_capability_major")?,
        "compute_capability_minor": nonnegative_u64(metadata, "compute_capability_minor")?,
        "driver_version": nonempty_string(metadata, "driver_version")?,
        "git_commit": nonempty_string(metadata, "git_commit")?,
        "git_dirty": false,
        "gpu_name": nonempty_string(metadata, "gpu_name")?,
        "gpu_ordinal": nonnegative_u64(metadata, "gpu_ordinal")?,
        "gpu_uuid": nonempty_string(metadata, "gpu_uuid")?,
        "host_arch": nonempty_string(metadata, "host_arch")?,
        "host_os": nonempty_string(metadata, "host_os")?,
        "multiprocessor_count": positive_u64(metadata, "multiprocessor_count")?,
        "nnis_version": nonempty_string(metadata, "nnis_version")?,
        "nvrtc_version": nonempty_string(metadata, "nvrtc_version")?,
        "environment_fingerprint": {
            "schema_version": 1,
            "cuda_visible_devices": nullable_string(fingerprint, "cuda_visible_devices")?,
            "environment_label": nullable_string(fingerprint, "environment_label")?,
            "host_kernel_release": nonempty_string(fingerprint, "host_kernel_release")?,
            "jetson_clock_state": nonempty_string(fingerprint, "jetson_clock_state")?,
            "jetson_power_mode": nonempty_string(fingerprint, "jetson_power_mode")?,
            "platform_model": nonempty_string(fingerprint, "platform_model")?,
        },
    }))
}

fn build_consensus(runs: &[ParsedRun], min_runs: usize) -> EnvironmentResult<Value> {
    if min_runs < 2 || runs.len() < min_runs {
        return Err(EnvironmentError(format!(
            "at least {min_runs} independent runs are required"
        )));
    }
    let reference = runs
        .first()
        .ok_or_else(|| EnvironmentError("no runs supplied".to_string()))?;
    let mut run_context_ids = BTreeSet::new();
    let mut total_metadata_records = 0usize;
    for run in runs {
        if !run_context_ids.insert(run.run_context_id.clone()) {
            return Err(EnvironmentError(format!(
                "duplicate run_context_id {:?}",
                run.run_context_id
            )));
        }
        if run.campaign_identity != reference.campaign_identity {
            return Err(EnvironmentError(format!(
                "run {:?} uses different RMSNorm experiment semantics",
                run.run_context_id
            )));
        }
        if run.environment_identity != reference.environment_identity {
            return Err(EnvironmentError(format!(
                "run {:?} is not environment-compatible with the reference run",
                run.run_context_id
            )));
        }
        total_metadata_records = total_metadata_records
            .checked_add(run.metadata_record_count)
            .ok_or_else(|| EnvironmentError("metadata record count overflows".to_string()))?;
    }

    Ok(json!({
        "schema_version": CONSENSUS_SCHEMA_VERSION,
        "consensus_kind": CONSENSUS_KIND,
        "source_campaign_kind": SOURCE_CAMPAIGN_KIND,
        "run_count": runs.len(),
        "minimum_required_runs": min_runs,
        "run_context_ids": run_context_ids.into_iter().collect::<Vec<_>>(),
        "validated_metadata_records": total_metadata_records,
        "campaign_identity": reference.campaign_identity.clone(),
        "environment_identity": reference.environment_identity.clone(),
        "compatible": true,
        "volatile_fields_excluded_from_environment_identity": [
            "metadata.unix_timestamp_seconds",
            "metadata.environment_fingerprint.run_context_id"
        ],
        "claim_boundary": "this gate establishes stable hardware/software environment and raw experiment-semantic compatibility across independent RMSNorm microbenchmark campaigns only; it does not establish a kernel speedup, end-to-end NNIS speedup, or runtime-promotion authority",
    }))
}

fn object<'a>(value: &'a Value, label: &str) -> EnvironmentResult<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| EnvironmentError(format!("{label} must be a JSON object")))
}

fn object_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> EnvironmentResult<&'a Map<String, Value>> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| EnvironmentError(format!("{field} must be a JSON object")))
}

fn array_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> EnvironmentResult<&'a Vec<Value>> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| EnvironmentError(format!("{field} must be a JSON array")))
}

fn require_u64(object: &Map<String, Value>, field: &str, expected: u64) -> EnvironmentResult<()> {
    let actual = nonnegative_u64(object, field)?;
    if actual != expected {
        return Err(EnvironmentError(format!(
            "{field} must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn require_str(object: &Map<String, Value>, field: &str, expected: &str) -> EnvironmentResult<()> {
    let actual = string_field(object, field)?;
    if actual != expected {
        return Err(EnvironmentError(format!(
            "{field} must be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn string_field<'a>(object: &'a Map<String, Value>, field: &str) -> EnvironmentResult<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| EnvironmentError(format!("{field} must be a string")))
}

fn nonempty_string(object: &Map<String, Value>, field: &str) -> EnvironmentResult<String> {
    let value = string_field(object, field)?;
    if value.trim().is_empty() {
        return Err(EnvironmentError(format!("{field} must not be empty")));
    }
    Ok(value.to_string())
}

fn bool_field(object: &Map<String, Value>, field: &str) -> EnvironmentResult<bool> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| EnvironmentError(format!("{field} must be a boolean")))
}

fn nonnegative_u64(object: &Map<String, Value>, field: &str) -> EnvironmentResult<u64> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| EnvironmentError(format!("{field} must be a non-negative integer")))
}

fn positive_u64(object: &Map<String, Value>, field: &str) -> EnvironmentResult<u64> {
    let value = nonnegative_u64(object, field)?;
    if value == 0 {
        return Err(EnvironmentError(format!("{field} must be positive")));
    }
    Ok(value)
}

fn finite_f64(object: &Map<String, Value>, field: &str) -> EnvironmentResult<f64> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| EnvironmentError(format!("{field} must be numeric")))?;
    if !value.is_finite() {
        return Err(EnvironmentError(format!("{field} must be finite")));
    }
    Ok(value)
}

fn nullable_string(object: &Map<String, Value>, field: &str) -> EnvironmentResult<Value> {
    match object.get(field) {
        Some(Value::Null) => Ok(Value::Null),
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Value::String(value.clone())),
        Some(Value::String(_)) => Err(EnvironmentError(format!("{field} must not be empty"))),
        Some(_) => Err(EnvironmentError(format!(
            "{field} must be a string or null"
        ))),
        None => Err(EnvironmentError(format!("missing field {field}"))),
    }
}

fn usize_from_u64(value: u64, field: &str) -> EnvironmentResult<usize> {
    usize::try_from(value).map_err(|_| EnvironmentError(format!("{field} does not fit usize")))
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

    fn metadata(run: &str, timestamp: u64, driver: &str, dirty: bool) -> Value {
        json!({
            "compute_capability_major": 11,
            "compute_capability_minor": 0,
            "driver_version": driver,
            "git_commit": "a4635547f3e46abd6652f8d319093dabc0baee6f",
            "git_dirty": dirty,
            "gpu_name": "NVIDIA Thor",
            "gpu_ordinal": 0,
            "gpu_uuid": "test-gpu-uuid",
            "host_arch": "aarch64",
            "host_os": "linux",
            "multiprocessor_count": 20,
            "nnis_version": "0.1.0",
            "nvrtc_version": "13.0",
            "unix_timestamp_seconds": timestamp,
            "environment_fingerprint": {
                "schema_version": 1,
                "cuda_visible_devices": null,
                "environment_label": null,
                "host_kernel_release": "6.8.12-tegra",
                "jetson_clock_state": "fixed-clocks",
                "jetson_power_mode": "MAXN",
                "platform_model": "NVIDIA Jetson AGX Thor Developer Kit",
                "run_context_id": run,
            }
        })
    }

    fn report(run: &str, timestamp: u64, driver: &str, dirty: bool, block: u64) -> Value {
        json!({
            "case": {
                "name": CASE_NAME,
                "dtype": CASE_DTYPE,
                "dimensions": {"rows": 1, "cols": 4096, "block_size": block}
            },
            "config": {"warmup_iterations": 20, "iterations": 100},
            "metadata": metadata(run, timestamp, driver, dirty)
        })
    }

    fn matrix(run: &str, timestamp: u64, driver: &str, dirty: bool) -> Value {
        json!({
            "schema_version": 1,
            "campaign_kind": SOURCE_CAMPAIGN_KIND,
            "run_context_id": run,
            "baseline_block_size": 256,
            "candidate_block_size": 512,
            "shape_count": 1,
            "shapes": [{"rows": 1, "cols": 4096}],
            "rounds_per_shape": 2,
            "warmup_iterations_per_observation": 20,
            "measured_iterations_per_observation": 100,
            "epsilon": 1.0e-6,
            "gamma": 1.0,
            "atol": 5.0e-5,
            "rtol": 5.0e-5,
            "results": [{
                "rows": 1,
                "cols": 4096,
                "observations": [
                    {
                        "baseline_report": report(run, timestamp, driver, dirty, 256),
                        "candidate_report": report(run, timestamp + 1, driver, dirty, 512)
                    },
                    {
                        "baseline_report": report(run, timestamp + 2, driver, dirty, 256),
                        "candidate_report": report(run, timestamp + 3, driver, dirty, 512)
                    }
                ]
            }]
        })
    }

    #[test]
    fn independent_runs_with_only_volatile_differences_are_compatible() {
        let first = parse_run(&matrix("run-a", 100, "13.0", false), "first").unwrap();
        let second = parse_run(&matrix("run-b", 200, "13.0", false), "second").unwrap();
        let consensus = build_consensus(&[first, second], 2).unwrap();
        assert_eq!(consensus["compatible"], true);
        assert_eq!(consensus["run_count"], 2);
        assert_eq!(consensus["validated_metadata_records"], 8);
    }

    #[test]
    fn driver_change_fails_closed() {
        let first = parse_run(&matrix("run-a", 100, "13.0", false), "first").unwrap();
        let second = parse_run(&matrix("run-b", 200, "13.1", false), "second").unwrap();
        assert!(build_consensus(&[first, second], 2).is_err());
    }

    #[test]
    fn experiment_change_fails_closed() {
        let first = parse_run(&matrix("run-a", 100, "13.0", false), "first").unwrap();
        let mut changed = matrix("run-b", 200, "13.0", false);
        changed["epsilon"] = json!(2.0e-6);
        let second = parse_run(&changed, "second").unwrap();
        assert!(build_consensus(&[first, second], 2).is_err());
    }

    #[test]
    fn duplicate_run_context_fails_closed() {
        let first = parse_run(&matrix("same", 100, "13.0", false), "first").unwrap();
        let second = parse_run(&matrix("same", 200, "13.0", false), "second").unwrap();
        assert!(build_consensus(&[first, second], 2).is_err());
    }

    #[test]
    fn dirty_git_metadata_fails_closed() {
        assert!(parse_run(&matrix("run-a", 100, "13.0", true), "dirty").is_err());
    }

    #[test]
    fn metadata_run_context_must_match_campaign() {
        let mut value = matrix("run-a", 100, "13.0", false);
        value["results"][0]["observations"][0]["baseline_report"]["metadata"]
            ["environment_fingerprint"]["run_context_id"] = json!("other-run");
        assert!(parse_run(&value, "mismatch").is_err());
    }
}
