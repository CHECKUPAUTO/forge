//! Execution-driven Forge domain for SOUP post-training recipe search.
//!
//! Forge owns candidate generation and Pareto selection. SOUP (normally behind
//! a Hub-qualified evaluator) owns training/evaluation semantics. This module
//! never invents task, memory, or timing scores: verification and objectives
//! must come from executed evaluator evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{ExternalDomainManifestV1, ObjectiveDirection};
use forge_core::{fnv1a, Candidate, CandidateId, Domain, ForgeError, Result, Score, Trial};
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const EVALUATOR_SCHEMA_VERSION: u16 = 1;
pub const DEFAULT_MAX_RESPONSE_BYTES: u64 = 1024 * 1024;

static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SoupRecipeCandidate {
    pub values: BTreeMap<String, String>,
}

impl Candidate for SoupRecipeCandidate {
    fn id(&self) -> CandidateId {
        fnv1a(&self.repr())
    }

    fn repr(&self) -> String {
        serde_json::to_string(&self.values).unwrap_or_else(|_| "{}".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct SoupSearchSpace {
    pub dimensions: BTreeMap<String, Vec<String>>,
    pub baseline: SoupRecipeCandidate,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SoupEvaluatorRequest {
    pub schema_version: u16,
    pub phase: String,
    pub domain_id: String,
    pub candidate_id: CandidateId,
    pub candidate: SoupRecipeCandidate,
    pub generation: u64,
    pub trial_seed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SoupVerificationEvidence {
    pub schema_version: u16,
    pub candidate_id: CandidateId,
    pub trial_seed: u64,
    pub passed: bool,
    pub evidence_id: String,
    pub environment_fingerprint: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SoupMeasurementEvidence {
    pub schema_version: u16,
    pub candidate_id: CandidateId,
    pub trial_seed: u64,
    pub evidence_id: String,
    pub environment_fingerprint: String,
    pub metrics: BTreeMap<String, f64>,
}

pub trait SoupEvaluator: Send + Sync {
    fn verify(
        &self,
        request: &SoupEvaluatorRequest,
    ) -> std::result::Result<SoupVerificationEvidence, String>;

    fn measure(
        &self,
        request: &SoupEvaluatorRequest,
    ) -> std::result::Result<SoupMeasurementEvidence, String>;
}

/// Structured process evaluator. The child receives one JSON request on stdin
/// and must emit one JSON response on stdout. No shell is involved.
///
/// Output is redirected to scratch files rather than pipes so a noisy child
/// cannot deadlock on a full pipe. The files are size-bounded before decoding.
/// Wall-clock limits and hostile-code isolation remain responsibilities of the
/// outer campaign/worker boundary.
#[derive(Clone, Debug)]
pub struct ProcessSoupEvaluator {
    program: PathBuf,
    args: Vec<String>,
    max_response_bytes: u64,
}

impl ProcessSoupEvaluator {
    pub fn new(
        program: impl Into<PathBuf>,
        args: Vec<String>,
    ) -> std::result::Result<Self, String> {
        let program = program.into();
        if !program.is_absolute() {
            return Err("SOUP evaluator program path must be absolute".to_string());
        }
        if args.iter().any(|arg| arg.contains('\0')) {
            return Err("SOUP evaluator arguments must not contain NUL bytes".to_string());
        }
        Ok(Self {
            program,
            args,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        })
    }

    pub fn with_max_response_bytes(mut self, bytes: u64) -> std::result::Result<Self, String> {
        if bytes == 0 || bytes > 16 * 1024 * 1024 {
            return Err("SOUP evaluator response limit must be in 1..=16777216 bytes".to_string());
        }
        self.max_response_bytes = bytes;
        Ok(self)
    }

    fn execute<T: for<'de> Deserialize<'de>>(
        &self,
        request: &SoupEvaluatorRequest,
    ) -> std::result::Result<T, String> {
        let mut input = ScratchFile::new("stdin")?;
        serde_json::to_writer(&mut input.file, request)
            .map_err(|error| format!("serialize evaluator request: {error}"))?;
        input
            .file
            .write_all(b"\n")
            .map_err(|error| format!("write evaluator request: {error}"))?;
        input
            .file
            .rewind()
            .map_err(|error| format!("rewind evaluator request: {error}"))?;

        let mut stdout = ScratchFile::new("stdout")?;
        let mut stderr = ScratchFile::new("stderr")?;
        let status = Command::new(&self.program)
            .args(&self.args)
            .stdin(Stdio::from(input.reopen()?))
            .stdout(Stdio::from(stdout.reopen()?))
            .stderr(Stdio::from(stderr.reopen()?))
            .status()
            .map_err(|error| format!("start SOUP evaluator {:?}: {error}", self.program))?;

        let stdout_bytes = stdout.read_bounded(self.max_response_bytes)?;
        let stderr_bytes = stderr.read_bounded(self.max_response_bytes)?;
        if !status.success() {
            return Err(format!(
                "SOUP evaluator exited with {status}: {}",
                String::from_utf8_lossy(&stderr_bytes)
            ));
        }
        serde_json::from_slice(&stdout_bytes)
            .map_err(|error| format!("invalid SOUP evaluator JSON: {error}"))
    }
}

impl SoupEvaluator for ProcessSoupEvaluator {
    fn verify(
        &self,
        request: &SoupEvaluatorRequest,
    ) -> std::result::Result<SoupVerificationEvidence, String> {
        self.execute(request)
    }

    fn measure(
        &self,
        request: &SoupEvaluatorRequest,
    ) -> std::result::Result<SoupMeasurementEvidence, String> {
        self.execute(request)
    }
}

struct ScratchFile {
    path: PathBuf,
    file: File,
}

impl ScratchFile {
    fn new(role: &str) -> std::result::Result<Self, String> {
        for _ in 0..64 {
            let nonce = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "forge-soup-{}-{nonce}-{role}.tmp",
                std::process::id()
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => return Ok(Self { path, file }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("create evaluator scratch file: {error}")),
            }
        }
        Err("could not allocate unique evaluator scratch file".to_string())
    }

    fn reopen(&self) -> std::result::Result<File, String> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|error| format!("reopen evaluator scratch file: {error}"))
    }

    fn read_bounded(&mut self, limit: u64) -> std::result::Result<Vec<u8>, String> {
        let size = self
            .file
            .metadata()
            .map_err(|error| format!("stat evaluator output: {error}"))?
            .len();
        if size > limit {
            return Err(format!("evaluator output exceeded {limit} bytes"));
        }
        self.file
            .rewind()
            .map_err(|error| format!("rewind evaluator output: {error}"))?;
        let mut bytes = Vec::new();
        Read::by_ref(&mut self.file)
            .take(limit.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read evaluator output: {error}"))?;
        if bytes.len() as u64 > limit {
            return Err(format!("evaluator output exceeded {limit} bytes"));
        }
        Ok(bytes)
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

struct EvidenceHeader<'a> {
    schema_version: u16,
    candidate_id: CandidateId,
    trial_seed: u64,
    evidence_id: &'a str,
    environment_fingerprint: &'a str,
}

pub struct SoupPostTrainDomain<E: SoupEvaluator> {
    manifest: ExternalDomainManifestV1,
    search: SoupSearchSpace,
    evaluator: E,
    domain_name: String,
}

impl<E: SoupEvaluator> SoupPostTrainDomain<E> {
    pub fn new(
        manifest: ExternalDomainManifestV1,
        search: SoupSearchSpace,
        evaluator: E,
        isolation_available: bool,
    ) -> Result<Self> {
        manifest.validate().map_err(|error| {
            ForgeError::InvalidCandidate(format!("SOUP domain manifest: {error}"))
        })?;
        if manifest.environment.isolation_required && !isolation_available {
            return Err(ForgeError::InvalidCandidate(
                "SOUP domain requires external isolation but none was declared available"
                    .to_string(),
            ));
        }
        validate_search_space(&manifest, &search)?;
        let domain_name = format!("soup_posttrain/{}", manifest.domain_id);
        Ok(Self {
            manifest,
            search,
            evaluator,
            domain_name,
        })
    }

    fn validate_candidate(&self, candidate: &SoupRecipeCandidate) -> Result<()> {
        let expected: BTreeSet<&str> = self.search.dimensions.keys().map(String::as_str).collect();
        let actual: BTreeSet<&str> = candidate.values.keys().map(String::as_str).collect();
        if expected != actual {
            return Err(ForgeError::InvalidCandidate(
                "SOUP candidate dimensions do not match the declared search space".to_string(),
            ));
        }
        for (name, value) in &candidate.values {
            let allowed = self.search.dimensions.get(name).ok_or_else(|| {
                ForgeError::InvalidCandidate(format!("unknown SOUP candidate dimension {name:?}"))
            })?;
            if !allowed.contains(value) {
                return Err(ForgeError::InvalidCandidate(format!(
                    "SOUP candidate value {value:?} is not allowed for {name:?}"
                )));
            }
        }
        Ok(())
    }

    fn request(
        &self,
        candidate: &SoupRecipeCandidate,
        trial: &Trial,
        phase: &str,
    ) -> SoupEvaluatorRequest {
        SoupEvaluatorRequest {
            schema_version: EVALUATOR_SCHEMA_VERSION,
            phase: phase.to_string(),
            domain_id: self.manifest.domain_id.clone(),
            candidate_id: candidate.id(),
            candidate: candidate.clone(),
            generation: trial.generation,
            trial_seed: trial.seed,
        }
    }

    fn validate_evidence(
        &self,
        candidate: &SoupRecipeCandidate,
        trial: &Trial,
        header: EvidenceHeader<'_>,
    ) -> Result<()> {
        if header.schema_version != EVALUATOR_SCHEMA_VERSION {
            return Err(ForgeError::Evaluation(format!(
                "unsupported SOUP evaluator schema {}",
                header.schema_version
            )));
        }
        if header.candidate_id != candidate.id() || header.trial_seed != trial.seed {
            return Err(ForgeError::Evaluation(
                "SOUP evaluator evidence identity does not match candidate/trial".to_string(),
            ));
        }
        if header.evidence_id.trim().is_empty() {
            return Err(ForgeError::Evaluation(
                "SOUP evaluator evidence_id must be non-empty".to_string(),
            ));
        }
        if self.manifest.environment.fingerprint_required
            && header.environment_fingerprint.trim().is_empty()
        {
            return Err(ForgeError::Evaluation(
                "SOUP evaluator omitted required environment fingerprint".to_string(),
            ));
        }
        Ok(())
    }

    fn measured_objectives(&self, evidence: &SoupMeasurementEvidence) -> Result<Vec<f64>> {
        let expected: BTreeSet<&str> = self
            .manifest
            .objectives
            .iter()
            .map(|objective| objective.name.as_str())
            .collect();
        let actual: BTreeSet<&str> = evidence.metrics.keys().map(String::as_str).collect();
        if expected != actual {
            return Err(ForgeError::Evaluation(
                "SOUP measurement objective set does not match the domain manifest".to_string(),
            ));
        }

        self.manifest
            .objectives
            .iter()
            .map(|objective| {
                let value = *evidence.metrics.get(&objective.name).ok_or_else(|| {
                    ForgeError::Evaluation(format!("missing SOUP objective {:?}", objective.name))
                })?;
                if !value.is_finite() {
                    return Err(ForgeError::Evaluation(format!(
                        "SOUP objective {:?} is not finite",
                        objective.name
                    )));
                }
                Ok(match objective.direction {
                    ObjectiveDirection::Minimize => value,
                    ObjectiveDirection::Maximize => -value,
                })
            })
            .collect()
    }
}

impl<E: SoupEvaluator> Domain for SoupPostTrainDomain<E> {
    type Cand = SoupRecipeCandidate;

    fn name(&self) -> &str {
        &self.domain_name
    }

    fn seed(&self, rng: &mut StdRng) -> Self::Cand {
        let values = self
            .search
            .dimensions
            .iter()
            .map(|(name, allowed)| {
                let index = rng.gen_range(0..allowed.len());
                (name.clone(), allowed[index].clone())
            })
            .collect();
        SoupRecipeCandidate { values }
    }

    fn mutate(&self, rng: &mut StdRng, parents: &[&Self::Cand]) -> Result<Self::Cand> {
        let mut candidate = if parents.is_empty() {
            self.seed(rng)
        } else {
            parents[rng.gen_range(0..parents.len())].clone()
        };
        let dimensions: Vec<&String> = self.search.dimensions.keys().collect();
        let name = dimensions[rng.gen_range(0..dimensions.len())];
        let allowed = &self.search.dimensions[name];
        candidate.values.insert(
            name.clone(),
            allowed[rng.gen_range(0..allowed.len())].clone(),
        );
        Ok(candidate)
    }

    fn verify(&self, candidate: &Self::Cand, trial: &Trial) -> Result<bool> {
        self.validate_candidate(candidate)?;
        let evidence = self
            .evaluator
            .verify(&self.request(candidate, trial, "verify"))
            .map_err(|error| ForgeError::Evaluation(format!("SOUP verify: {error}")))?;
        self.validate_evidence(
            candidate,
            trial,
            EvidenceHeader {
                schema_version: evidence.schema_version,
                candidate_id: evidence.candidate_id,
                trial_seed: evidence.trial_seed,
                evidence_id: &evidence.evidence_id,
                environment_fingerprint: &evidence.environment_fingerprint,
            },
        )?;
        Ok(evidence.passed)
    }

    fn measure(&self, candidate: &Self::Cand, trial: &Trial) -> Result<Vec<f64>> {
        self.validate_candidate(candidate)?;
        let evidence = self
            .evaluator
            .measure(&self.request(candidate, trial, "measure"))
            .map_err(|error| ForgeError::Evaluation(format!("SOUP measure: {error}")))?;
        self.validate_evidence(
            candidate,
            trial,
            EvidenceHeader {
                schema_version: evidence.schema_version,
                candidate_id: evidence.candidate_id,
                trial_seed: evidence.trial_seed,
                evidence_id: &evidence.evidence_id,
                environment_fingerprint: &evidence.environment_fingerprint,
            },
        )?;
        self.measured_objectives(&evidence)
    }

    fn objective_names(&self) -> Vec<String> {
        self.manifest
            .objectives
            .iter()
            .map(|objective| match objective.direction {
                ObjectiveDirection::Minimize => format!("minimize:{}", objective.name),
                ObjectiveDirection::Maximize => format!("maximize:{}", objective.name),
            })
            .collect()
    }

    fn baseline(&self, trial: &Trial) -> Result<Score> {
        if !self.verify(&self.search.baseline, trial)? {
            return Ok(Score::invalid());
        }
        Ok(Score::valid(self.measure(&self.search.baseline, trial)?))
    }
}

fn validate_search_space(
    manifest: &ExternalDomainManifestV1,
    search: &SoupSearchSpace,
) -> Result<()> {
    let manifest_dimensions: BTreeSet<&str> = manifest
        .allowed_candidate_dimensions
        .iter()
        .map(String::as_str)
        .collect();
    let search_dimensions: BTreeSet<&str> = search.dimensions.keys().map(String::as_str).collect();
    if manifest_dimensions != search_dimensions {
        return Err(ForgeError::InvalidCandidate(
            "SOUP search dimensions must exactly match external-domain allowed dimensions"
                .to_string(),
        ));
    }
    for (name, values) in &search.dimensions {
        if values.is_empty() {
            return Err(ForgeError::InvalidCandidate(format!(
                "SOUP search dimension {name:?} has no values"
            )));
        }
        let unique: BTreeSet<&str> = values.iter().map(String::as_str).collect();
        if unique.len() != values.len() || unique.iter().any(|value| value.is_empty()) {
            return Err(ForgeError::InvalidCandidate(format!(
                "SOUP search dimension {name:?} contains empty or duplicate values"
            )));
        }
    }

    let baseline_dimensions: BTreeSet<&str> =
        search.baseline.values.keys().map(String::as_str).collect();
    if search_dimensions != baseline_dimensions {
        return Err(ForgeError::InvalidCandidate(
            "SOUP baseline dimensions do not match the search space".to_string(),
        ));
    }
    for (name, value) in &search.baseline.values {
        if !search.dimensions[name].contains(value) {
            return Err(ForgeError::InvalidCandidate(format!(
                "SOUP baseline value {value:?} is not allowed for {name:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DataBoundaryV1, EnvironmentPolicyV1, ObjectiveSpecV1, UpstreamContractRefV1,
        VerificationBindingV1, EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION,
    };
    use rand::SeedableRng;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct FakeEvaluator {
        calls: Arc<Mutex<Vec<String>>>,
    }

    impl SoupEvaluator for FakeEvaluator {
        fn verify(
            &self,
            request: &SoupEvaluatorRequest,
        ) -> std::result::Result<SoupVerificationEvidence, String> {
            self.calls.lock().unwrap().push(request.phase.clone());
            Ok(SoupVerificationEvidence {
                schema_version: EVALUATOR_SCHEMA_VERSION,
                candidate_id: request.candidate_id,
                trial_seed: request.trial_seed,
                passed: true,
                evidence_id: "verify-evidence".to_string(),
                environment_fingerprint: "gpu=test;driver=test".to_string(),
            })
        }

        fn measure(
            &self,
            request: &SoupEvaluatorRequest,
        ) -> std::result::Result<SoupMeasurementEvidence, String> {
            self.calls.lock().unwrap().push(request.phase.clone());
            Ok(SoupMeasurementEvidence {
                schema_version: EVALUATOR_SCHEMA_VERSION,
                candidate_id: request.candidate_id,
                trial_seed: request.trial_seed,
                evidence_id: "measure-evidence".to_string(),
                environment_fingerprint: "gpu=test;driver=test".to_string(),
                metrics: BTreeMap::from([
                    ("task_score".to_string(), 0.75),
                    ("peak_vram_bytes".to_string(), 1024.0),
                    ("wall_ms".to_string(), 50.0),
                ]),
            })
        }
    }

    fn manifest() -> ExternalDomainManifestV1 {
        ExternalDomainManifestV1 {
            schema_version: EXTERNAL_DOMAIN_MANIFEST_SCHEMA_VERSION,
            domain_id: "soup/posttrain-v1".to_string(),
            upstream: UpstreamContractRefV1 {
                repository: "MakazhanAlpamys/Soup".to_string(),
                commit_id: "05b646523727925990530667e7012ede50bd30b2".to_string(),
                contract_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
            },
            allowed_candidate_dimensions: vec![
                "recipe.learning_rate".to_string(),
                "recipe.lora_rank".to_string(),
            ],
            data_boundary: DataBoundaryV1 {
                generation_sources: vec!["train-split".to_string()],
                verification_sources: vec!["validation-split".to_string()],
                final_holdout_sources: vec!["final-holdout".to_string()],
            },
            verification: VerificationBindingV1 {
                adapter_id: "scirust-hub/soup-eval-v1".to_string(),
                adapter_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                    .to_string(),
            },
            objectives: vec![
                ObjectiveSpecV1 {
                    name: "task_score".to_string(),
                    direction: ObjectiveDirection::Maximize,
                },
                ObjectiveSpecV1 {
                    name: "peak_vram_bytes".to_string(),
                    direction: ObjectiveDirection::Minimize,
                },
                ObjectiveSpecV1 {
                    name: "wall_ms".to_string(),
                    direction: ObjectiveDirection::Minimize,
                },
            ],
            environment: EnvironmentPolicyV1 {
                fingerprint_required: true,
                isolation_required: false,
            },
        }
    }

    fn search_space() -> SoupSearchSpace {
        SoupSearchSpace {
            dimensions: BTreeMap::from([
                (
                    "recipe.learning_rate".to_string(),
                    vec!["1e-5".to_string(), "2e-5".to_string()],
                ),
                (
                    "recipe.lora_rank".to_string(),
                    vec!["8".to_string(), "16".to_string()],
                ),
            ]),
            baseline: SoupRecipeCandidate {
                values: BTreeMap::from([
                    ("recipe.learning_rate".to_string(), "2e-5".to_string()),
                    ("recipe.lora_rank".to_string(), "16".to_string()),
                ]),
            },
        }
    }

    #[test]
    fn executed_metrics_are_the_only_objective_source() {
        let evaluator = FakeEvaluator::default();
        let calls = evaluator.calls.clone();
        let domain =
            SoupPostTrainDomain::new(manifest(), search_space(), evaluator, false).unwrap();
        let trial = Trial {
            generation: 2,
            seed: 7,
        };
        let candidate = search_space().baseline;
        assert!(domain.verify(&candidate, &trial).unwrap());
        assert_eq!(
            domain.measure(&candidate, &trial).unwrap(),
            vec![-0.75, 1024.0, 50.0]
        );
        assert_eq!(*calls.lock().unwrap(), vec!["verify", "measure"]);
    }

    #[test]
    fn baseline_is_verified_before_measurement() {
        let evaluator = FakeEvaluator::default();
        let calls = evaluator.calls.clone();
        let domain =
            SoupPostTrainDomain::new(manifest(), search_space(), evaluator, false).unwrap();
        assert!(
            domain
                .baseline(&Trial {
                    generation: 0,
                    seed: 11,
                })
                .unwrap()
                .valid
        );
        assert_eq!(*calls.lock().unwrap(), vec!["verify", "measure"]);
    }

    #[test]
    fn manifest_search_space_and_isolation_fail_closed() {
        let mut search = search_space();
        search.dimensions.remove("recipe.lora_rank");
        assert!(
            SoupPostTrainDomain::new(manifest(), search, FakeEvaluator::default(), false).is_err()
        );

        let mut contract = manifest();
        contract.environment.isolation_required = true;
        assert!(SoupPostTrainDomain::new(
            contract,
            search_space(),
            FakeEvaluator::default(),
            false,
        )
        .is_err());
    }

    #[test]
    fn seed_and_mutation_stay_inside_declared_values() {
        let domain =
            SoupPostTrainDomain::new(manifest(), search_space(), FakeEvaluator::default(), false)
                .unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let mut candidate = domain.seed(&mut rng);
        for _ in 0..100 {
            domain.validate_candidate(&candidate).unwrap();
            candidate = domain.mutate(&mut rng, &[&candidate]).unwrap();
        }
    }

    #[test]
    fn mismatched_evidence_identity_is_rejected() {
        #[derive(Clone)]
        struct BadEvaluator;
        impl SoupEvaluator for BadEvaluator {
            fn verify(
                &self,
                request: &SoupEvaluatorRequest,
            ) -> std::result::Result<SoupVerificationEvidence, String> {
                Ok(SoupVerificationEvidence {
                    schema_version: EVALUATOR_SCHEMA_VERSION,
                    candidate_id: request.candidate_id.wrapping_add(1),
                    trial_seed: request.trial_seed,
                    passed: true,
                    evidence_id: "bad".to_string(),
                    environment_fingerprint: "env".to_string(),
                })
            }

            fn measure(
                &self,
                _request: &SoupEvaluatorRequest,
            ) -> std::result::Result<SoupMeasurementEvidence, String> {
                unreachable!()
            }
        }

        let domain =
            SoupPostTrainDomain::new(manifest(), search_space(), BadEvaluator, false).unwrap();
        let error = domain
            .verify(
                &search_space().baseline,
                &Trial {
                    generation: 0,
                    seed: 1,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("identity"));
    }

    #[test]
    fn process_evaluator_requires_absolute_program() {
        assert!(ProcessSoupEvaluator::new("relative-evaluator", Vec::new()).is_err());
        let absolute = if cfg!(windows) {
            PathBuf::from("C:\\forge-soup-evaluator.exe")
        } else {
            PathBuf::from("/usr/local/bin/forge-soup-evaluator")
        };
        assert!(ProcessSoupEvaluator::new(absolute, Vec::new()).is_ok());
    }
}
