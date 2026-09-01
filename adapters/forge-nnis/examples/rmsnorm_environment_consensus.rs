use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;

const SOURCE_SCHEMA_VERSION: u64 = 1;
const SOURCE_CAMPAIGN_KIND: &str = "forge_nnis_canonical_rmsnorm_shape_matrix_v1";
const CONSENSUS_SCHEMA_VERSION: u64 = 1;
const CONSENSUS_KIND: &str = "forge_nnis_rmsnorm_environment_consensus_v1";
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
struct ParsedRun {
    run_context_id: String,
    environment_identity: Value,
    observation_count: usize,
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
    let rounds = positive_u64(root, "rounds_per_shape")?;
    if rounds < 2 {
        return Err(EnvironmentError(format!(
            "{label}: rounds_per_shape must be at least 2"
        )));
    }
    let shape_count = usize_from_u64(positive_u64(root, "shape_count")?, "shape_count")?;
    let results = array_field(root, "results")?;
    if results.len() != shape_count {
        return Err(EnvironmentError(format!(
            "{label}: shape_count {shape_count} does not match results length {}",
            results.len()
        )));
    }

    let mut expected_identity: Option<Value> = None;
    let mut observation_count = 0usize;
    for (result_index, result) in results.iter().enumerate() {
        let result = object(result, &format!("{label}.results[{result_index}]"))?;
        let observations = array_field(result, "observations")?;
        if observations.len() != rounds as usize {
            return Err(EnvironmentError(format!(
                "{label}.results[{result_index}]: observations length {} does not match rounds_per_shape {rounds}",
                observations.len()
            )));
        }
        for (observation_index, observation) in observations.iter().enumerate() {
            let observation = object(
                observation,
                &format!("{label}.results[{result_index}].observations[{observation_index}]"),
            )?;
            for report_field in ["baseline_report", "candidate_report"] {
                let report = object_field(observation, report_field)?;
                let metadata = object_field(report, "metadata")?;
                let identity = environment_identity(metadata, &run_context_id)?;
                match &expected_identity {
                    Some(expected) if expected != &identity => {
                        return Err(EnvironmentError(format!(
                            "{label}: stable environment identity changed within one campaign"
                        )));
                    }
                    Some(_) => {}
                    None => expected_identity = Some(identity),
                }
                observation_count += 1;
            }
        }
    }

    let environment_identity = expected_identity.ok_or_else(|| {
        EnvironmentError(format!(
            "{label}: no benchmark metadata was found in the campaign"
        ))
    })?;
    Ok(ParsedRun {
        run_context_id,
        environment_identity,
        observation_count,
    })
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
        if run.environment_identity != reference.environment_identity {
            return Err(EnvironmentError(format!(
                "run {:?} is not environment-compatible with the reference run",
                run.run_context_id
            )));
        }
        total_metadata_records = total_metadata_records
            .checked_add(run.observation_count)
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
        "environment_identity": reference.environment_identity,
        "compatible": true,
        "volatile_fields_excluded_from_identity": [
            "metadata.unix_timestamp_seconds",
            "metadata.environment_fingerprint.run_context_id"
        ],
        "claim_boundary": "this gate establishes stable hardware/software environment compatibility across independent RMSNorm microbenchmark campaigns only; it does not establish a kernel speedup, end-to-end NNIS speedup, or runtime-promotion authority",
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

fn array_field<'a>(object: &'a Map<String, Value>, field: &str) -> EnvironmentResult<&'a Vec<Value>> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| EnvironmentError(format!("{field} must be a JSON array")))
}

fn require_u64(
    object: &Map<String, Value>,
    field: &str,
    expected: u64,
) -> EnvironmentResult<()> {
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

fn nullable_string(object: &Map<String, Value>, field: &str) -> EnvironmentResult<Value> {
    match object.get(field) {
        Some(Value::Null) => Ok(Value::Null),
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Value::String(value.clone())),
        Some(Value::String(_)) => Err(EnvironmentError(format!("{field} must not be empty"))),
        Some(_) => Err(EnvironmentError(format!("{field} must be a string or null"))),
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

    fn matrix(run: &str, timestamp: u64, driver: &str, dirty: bool) -> Value {
        let report = |offset| {
            json!({
                "metadata": metadata(run, timestamp + offset, driver, dirty)
            })
        };
        json!({
            "schema_version": 1,
            "campaign_kind": SOURCE_CAMPAIGN_KIND,
            "run_context_id": run,
            "shape_count": 1,
            "rounds_per_shape": 2,
            "results": [{
                "observations": [
                    {"baseline_report": report(0), "candidate_report": report(1)},
                    {"baseline_report": report(2), "candidate_report": report(3)}
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
