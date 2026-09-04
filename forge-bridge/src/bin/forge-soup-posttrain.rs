//! Process entry point for the versioned Forge SOUP post-training campaign.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use forge_bridge::soup_campaign::{run_soup_campaign, SoupCampaignSpecV1};
use forge_bridge::soup_posttrain::{ProcessSoupEvaluator, DEFAULT_MAX_RESPONSE_BYTES};

const MAX_CAMPAIGN_BYTES: u64 = 1024 * 1024;

struct Args {
    campaign: PathBuf,
    evaluator: PathBuf,
    evaluator_args: Vec<String>,
    output: PathBuf,
    isolation_available: bool,
    max_response_bytes: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("forge-soup-posttrain: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(std::env::args().skip(1))?;
    require_regular_non_symlink(&args.campaign, "campaign")?;
    require_regular_non_symlink(&args.evaluator, "evaluator")?;
    if !args.evaluator.is_absolute() {
        return Err("--evaluator must be an absolute path".to_string());
    }

    let campaign_bytes = read_bounded(&args.campaign, MAX_CAMPAIGN_BYTES)?;
    let spec: SoupCampaignSpecV1 = serde_json::from_slice(&campaign_bytes)
        .map_err(|error| format!("invalid campaign JSON: {error}"))?;

    let evaluator = ProcessSoupEvaluator::new(args.evaluator, args.evaluator_args)
        .map_err(|error| format!("invalid evaluator binding: {error}"))?
        .with_max_response_bytes(args.max_response_bytes)
        .map_err(|error| format!("invalid evaluator response limit: {error}"))?;

    let report = run_soup_campaign(spec, evaluator, args.isolation_available)?;
    let payload = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("serialize campaign report: {error}"))?;
    write_atomic_regular(&args.output, &payload)?;
    Ok(())
}

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut campaign = None;
    let mut evaluator = None;
    let mut evaluator_args = Vec::new();
    let mut output = None;
    let mut isolation_available = false;
    let mut max_response_bytes = DEFAULT_MAX_RESPONSE_BYTES;
    let mut values = args.into_iter();

    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--campaign" => campaign = Some(next_value(&mut values, "--campaign")?.into()),
            "--evaluator" => evaluator = Some(next_value(&mut values, "--evaluator")?.into()),
            "--evaluator-arg" => evaluator_args.push(next_value(&mut values, "--evaluator-arg")?),
            "--output" => output = Some(next_value(&mut values, "--output")?.into()),
            "--isolation-available" => isolation_available = true,
            "--max-response-bytes" => {
                let raw = next_value(&mut values, "--max-response-bytes")?;
                max_response_bytes = raw
                    .parse::<u64>()
                    .map_err(|_| "--max-response-bytes must be an integer".to_string())?;
            }
            "--help" | "-h" => return Err(usage()),
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }

    Ok(Args {
        campaign: campaign.ok_or_else(|| format!("missing --campaign\n{}", usage()))?,
        evaluator: evaluator.ok_or_else(|| format!("missing --evaluator\n{}", usage()))?,
        evaluator_args,
        output: output.ok_or_else(|| format!("missing --output\n{}", usage()))?,
        isolation_available,
        max_response_bytes,
    })
}

fn next_value<I>(values: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    values
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn usage() -> String {
    "usage: forge-soup-posttrain --campaign <campaign.json> --evaluator <absolute-program> \
     [--evaluator-arg <arg>]... --output <report.json> [--isolation-available] \
     [--max-response-bytes <bytes>]"
        .to_string()
}

fn require_regular_non_symlink(path: &Path, role: &str) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("stat {role} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!("{role} path must not be a symlink"));
    }
    if !metadata.is_file() {
        return Err(format!("{role} path must be a regular file"));
    }
    Ok(())
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("stat campaign {}: {error}", path.display()))?;
    if metadata.len() > limit {
        return Err(format!("campaign exceeds {limit} bytes"));
    }
    let bytes = fs::read(path)
        .map_err(|error| format!("read campaign {}: {error}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(format!("campaign exceeds {limit} bytes"));
    }
    Ok(bytes)
}

fn write_atomic_regular(path: &Path, payload: &[u8]) -> Result<(), String> {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err("output path, when present, must be a regular non-symlink file".to_string());
        }
    }

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|error| format!("create output directory {}: {error}", parent.display()))?;

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "output path must have a UTF-8 file name".to_string())?;
    let temp = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    let _ = fs::remove_file(&temp);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|error| format!("create temporary report {}: {error}", temp.display()))?;
    if let Err(error) = file.write_all(payload).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temp);
        return Err(format!("write temporary report {}: {error}", temp.display()));
    }
    drop(file);
    if let Err(error) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(format!("publish report {}: {error}", path.display()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_require_explicit_process_bindings() {
        let error = parse_args(Vec::<String>::new()).err().expect("missing args");
        assert!(error.contains("--campaign"));

        let parsed = parse_args([
            "--campaign".to_string(),
            "campaign.json".to_string(),
            "--evaluator".to_string(),
            "/opt/evaluator".to_string(),
            "--evaluator-arg".to_string(),
            "fixed".to_string(),
            "--output".to_string(),
            "report.json".to_string(),
            "--isolation-available".to_string(),
        ])
        .expect("valid args");
        assert!(parsed.isolation_available);
        assert_eq!(parsed.evaluator_args, ["fixed"]);
    }

    #[test]
    fn unknown_arguments_fail_closed() {
        let error = parse_args(["--mystery".to_string()])
            .err()
            .expect("unknown argument");
        assert!(error.contains("unknown argument"));
    }
}
