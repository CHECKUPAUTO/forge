//! Execution-driven Forge domain for SOUP post-training recipe search.
//!
//! Forge owns candidate generation and Pareto selection. SOUP (normally behind
//! a Hub-qualified evaluator) owns training/evaluation semantics. This adapter
//! never invents a task score, memory value, or timing value: verification and
//! objectives must come from executed evaluator evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use forge_bridge::{ExternalDomainManifestV1, ObjectiveDirection};
use forge_core::{fnv1a, Candidate, CandidateId, Domain, ForgeError, Result, Score, Trial};
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

pub const EVALUATOR_SCHEMA_VERSION: u16 = 1;
pub const DEFAULT_MAX_RESPONSE_BYTES: u64 = 1024 * 1024;

/// One concrete SOUP recipe candidate. Keys are constrained by the external
/// domain manifest and values by [`SoupSearchSpace`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SoupRecipeCandidate {
    pub values: BTreeMap<String, String>,
}

impl Candidate for SoupRecipeCandidate {
    fn id(&self) -> CandidateId {
        fnv1a(&self.repr())
    }

    fn repr(&self) -> String {
        match serde_json::to_string(&self.values) {
            Ok(value) => value,
            Err(_) => String::from("{}"),
        }
    }
}

/// Finite, explicitly declared search space. Forge never guesses SOUP config
/// field names; callers must bind every mutable dimension to the upstream
/// contract through [`ExternalDomainManifestV1::allowed_candidate_dimensions`].
#[derive(Clone, Debug)]
pub struct SoupSearchSpace {
    pub dimensions: BTreeMap<String, Vec<String>>,
    pub baseline: SoupRecipeCandidate,
}

/// One evaluator request. `phase` is either `verify` or `measure`.
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

/// Independent correctness/quality gate result. A successful process is not
/// sufficient: `passed` must be true before Forge requests measurements.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SoupVerificationEvidence {
    pub schema_version: u16,
    pub candidate_id: CandidateId,
    pub trial_seed: u64,
    pub passed: bool,
    pub evidence_id: String,
    pub environment_fingerprint: String,
}

/// Executed measurements keyed by the exact objective names in the external
/// manifest. Forge rejects missing, extra, or non-finite metrics.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SoupMeasurementEvidence {
    pub schema_version: u16,
    pub candidate_id: CandidateId,
    pub trial_seed: u64,
    pub evidence_id: String,
    pub environment_fingerprint: String,
    pub metrics: BTreeMap<String, f64>,
}

/// Evaluation seam. Implementations may call SciRust Hub/SOUP directly or a
/// trusted worker service, but executed evidence remains authoritative.
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

/// Structured-process evaluator. The child executable receives one request JSON
/// on stdin and must emit one response JSON on stdout. No shell is involved.
///
/// The executable path must be absolute. stdout/stderr are redirected to files
/// so the child cannot deadlock on full pipes; response size is checked before
/// reading. Hostile-code isolation and wall-clock limits remain the outer
/// campaign/worker responsibility.
#[derive(Clone, Debug)]
pub struct ProcessSoupEvaluator {
    program: PathBuf,
    args: Vec<String>,
    max_response_bytes: u64,
}

impl ProcessSoupEvaluator {
    pub fn new(program: impl Into<PathBuf>, args: Vec<String>) -> std::result::Result<Self, String> {
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
        let mut input = NamedTempFile::new().map_err(|error| format!("temp input: {error}"))?;
        serde_json::to_writer(&mut input, request)
            .map_err(|error| format!("serialize evaluator request: {error}"))?;
        input
            .write_all(b"\n")
            .map_err(|error| format!("write evaluator request: {error}"))?;
        input
            .as_file_mut()
            .rewind()
            .map_err(|error| format!("rewind evaluator request: {error}"))?;

        let mut stdout = NamedTempFile::new().map_err(|error| format!("temp stdout: {error}"))?;
        let mut stderr = NamedTempFile::new().map_err(|error| format!("temp stderr: {error}"))?;

        let status = Command::new(&self.program)
            .args(&self.args)
            .stdin(Stdio::from(
                input
                    .reopen()
                    .map_err(|error| format!("reopen evaluator input: {error}"))?,
            ))
            .stdout(Stdio::from(
                stdout
                    .reopen()
                    .map_err(|error| format!("reopen evaluator stdout: {error}"))?,
            ))
            .stderr(Stdio::from(
                stderr
                    .reopen()
                    .map_err(|error| format!("reopen evaluator stderr: {error}"))?,
            ))
            .status()
            .map_err(|error| format!("start SOUP evaluator {:?}: {error}", self.program))?;

        let stdout_size = stdout
            .as_file()
            .metadata()
            .map_err(|error| format!("stat evaluator stdout: {error}"))?
            .len();
        let stderr_size = stderr
            .as_file()
            .metadata()
            .map_err(|error| format!("stat evaluator stderr: {error}"))?
            .len();
        if stdout_size > self.max_response_bytes {
            return Err(format!(
                "SOUP evaluator stdout exceeded {} bytes",
                self.max_response_bytes
            ));
        }
        if stderr_size > self.max_response_bytes {
            return Err(format!(
                "SOUP evaluator stderr exceeded {} bytes",
                self.max_response_bytes
            ));
        }

        let stdout_bytes = read_temp(&mut stdout, self.max_response_bytes)?;
        let stderr_bytes = read_temp(&mut stderr, self.max_response_bytes)?;
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

fn read_temp(file: &mut NamedTempFile, limit: u64) -> std::result::Result<Vec<u8>, String> {
    file.as_file_mut()
        .rewind()
        .map_err(|error| format!("rewind evaluator output: {error}"))?;
    let mut bytes = Vec::new();
    file.as_file_mut()
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read evaluator output: {error}"))?;
    if bytes.len() as u64 > limit {
        return Err(format!("evaluator output exceeded {limit} bytes"));
    }
    Ok(bytes)
}

/// Forge search domain over a finite SOUP recipe space.
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
        manifest
            .validate()
            .map_err(|error| ForgeError::InvalidCandidate(format!("SOUP domain manifest: {error}")))?;
        if manifest.environment.isolation_required && !isolation_available {
            return Err(ForgeError::InvalidCandidate(
                "SOUP domain requires external isolation but none was declared available".to_string(),
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

    fn request(&self, candidate: &SoupRecipeCandidate, trial: &Trial, phase: &str) -> SoupEvaluatorRequest {
        SoupEvaluatorRequest {
            schema_version: EVALUATOR_SCHEMA_VERSION,
            phase: phase.to_string(),
            domain_id: self.manifest.domain_id.clone(),
            candidate_id: candidate.id(),
            candidate: candidate.clone(),
            generation: trial.generation as u64,
            trial_seed: trial.seed,
        }
    }

    fn validate_common_evidence(
        &self,
        candidate: &SoupRecipeCandidate,
        trial: &Trial,
        schema_version: u16,
        candidate_id: CandidateId,
        trial_seed: u64,
        evidence_id: &str,
        environment_fingerprint: &str,
    ) -> Result<()> {
        if schema_version != EVALUATOR_SCHEMA_VERSION {
            return Err(ForgeError::Evaluation(format!(
                "unsupported SOUP evaluator schema {schema_version}"
            )));
        }
        if candidate_id != candidate.id() || trial_seed != trial.seed {
            return Err(ForgeError::Evaluation(
                "SOUP evaluator evidence identity does not match candidate/trial".to_string(),
            ));
        }
        if evidence_id.trim().is_empty() {
            return Err(ForgeError::Evaluation(
                "SOUP evaluator evidence_id must be non-empty".to_string(),
            ));
        }
        if self.manifest.environment.fingerprint_required
            && environment_fingerprint.trim().is_empty()
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

        let mut values = Vec::with_capacity(self.manifest.objectives.len());
        for objective in &self.manifest.objectives {
            let value = *evidence.metrics.get(&objective.name).ok_or_else(|| {
                ForgeError::Evaluation(format!("missing SOUP objective {:?}", objective.name))
            })?;
            if !value.is_finite() {
                return Err(ForgeError::Evaluation(format!(
                    "SOUP objective {:?} is not finite",
                    objective.name
                )));
            }
            values.push(match objective.direction {
                ObjectiveDirection::Minimize => value,
                ObjectiveDirection::Maximize => -value,
            });
        }
        Ok(values)
    }
}

impl<E: SoupEvaluator> Domain for SoupPostTrainDomain<E> {
    type Cand = SoupRecipeCandidate;

    fn name(&self) -> &str {
        &self.domain_name
    }

    fn seed(&self, rng: &mut StdRng) -> Self::Cand {
        let mut values = BTreeMap::new();
        for (name, allowed) in &self.search.dimensions {
            let index = rng.gen_range(0..allowed.len());
            values.insert(name.clone(), allowed[index].clone());
        }
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
        let value = allowed[rng.gen_range(0..allowed.len())].clone();
        candidate.values.insert(name.clone(), value);
        Ok(candidate)
    }

    fn verify(&self, candidate: &Self::Cand, trial: &Trial) -> Result<bool> {
        self.validate_candidate(candidate)?;
        let request = self.request(candidate, trial, "verify");
        let evidence = self
            .evaluator
            .verify(&request)
            .map_err(|error| ForgeError::Evaluation(format!("SOUP verify: {error}")))?;
        self.validate_common_evidence(
            candidate,
            trial,
            evidence.schema_version,
            evidence.candidate_id,
            evidence.trial_seed,
            &evidence.evidence_id,
            &evidence.environment_fingerprint,
        )?;
        Ok(evidence.passed)
    }

    fn measure(&self, candidate: &Self::Cand, trial: &Trial) -> Result<Vec<f64>> {
        self.validate_candidate(candidate)?;
        let request = self.request(candidate, trial, "measure");
        let evidence = self
            .evaluator
            .measure(&request)
            .map_err(|error| ForgeError::Evaluation(format!("SOUP measure: {error}")))?;
        self.validate_common_evidence(
            candidate,
            trial,
            evidence.schema_version,
            evidence.candidate_id,
            evidence.trial_seed,
            &evidence.evidence_id,
            &evidence.environment_fingerprint,
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
    let search_dimensions: BTreeSet<&str> =
        search.dimensions.keys().map(String::as_str).collect();
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
    let expected: BTreeSet<&str> = search.dimensions.keys().map(String::as_str).collect();
    let baseline: BTreeSet<&str> = search.baseline.values.keys().map(String::as_str).collect();
    if expected != baseline {
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
    use forge_bridge::{
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
                adapter_sha256:
                    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
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
        let domain = SoupPostTrainDomain::new(manifest(), search_space(), evaluator, false).unwrap();
        let trial = Trial {
            generation: 2,
            seed: 7,
        };
        let candidate = search_space().baseline;
        assert!(domain.verify(&candidate, &trial).unwrap());
        let objectives = domain.measure(&candidate, &trial).unwrap();
        assert_eq!(objectives, vec![-0.75, 1024.0, 50.0]);
        assert_eq!(
            domain.objective_names(),
            vec![
                "maximize:task_score",
                "minimize:peak_vram_bytes",
                "minimize:wall_ms"
            ]
        );
        assert_eq!(*calls.lock().unwrap(), vec!["verify", "measure"]);
    }

    #[test]
    fn baseline_is_verified_before_measurement() {
        let evaluator = FakeEvaluator::default();
        let calls = evaluator.calls.clone();
        let domain = SoupPostTrainDomain::new(manifest(), search_space(), evaluator, false).unwrap();
        let score = domain
            .baseline(&Trial {
                generation: 0,
                seed: 11,
            })
            .unwrap();
        assert!(score.valid);
        assert_eq!(*calls.lock().unwrap(), vec!["verify", "measure"]);
    }

    #[test]
    fn manifest_and_search_space_must_match_exactly() {
        let mut search = search_space();
        search.dimensions.remove("recipe.lora_rank");
        assert!(SoupPostTrainDomain::new(manifest(), search, FakeEvaluator::default(), false).is_err());
    }

    #[test]
    fn required_isolation_fails_closed() {
        let mut contract = manifest();
        contract.environment.isolation_required = true;
        assert!(
            SoupPostTrainDomain::new(contract, search_space(), FakeEvaluator::default(), false)
                .is_err()
        );
    }

    #[test]
    fn seed_and_mutation_never_leave_declared_values() {
        let domain = SoupPostTrainDomain::new(
            manifest(),
            search_space(),
            FakeEvaluator::default(),
            false,
        )
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

        let domain = SoupPostTrainDomain::new(manifest(), search_space(), BadEvaluator, false).unwrap();
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
            Path::new("C:\\forge-soup-evaluator.exe")
        } else {
            Path::new("/usr/local/bin/forge-soup-evaluator")
        };
        assert!(ProcessSoupEvaluator::new(absolute, Vec::new()).is_ok());
    }
}
