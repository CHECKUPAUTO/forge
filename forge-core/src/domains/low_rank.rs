//! Domaine de compression Tensor Train (Low-Rank / MPS) pour SciRust.
//!
//! Ce domaine génère, compile et évalue des décompositions de tenseurs
//! d'ordre N en rangs faibles. Chaque candidat est du code Rust brut
//! (`TensorCode`) compilé dans un répertoire isolé avec limites rlimit,
//! benchmarké via Criterion, et validé statistiquement.
//!
//! ## Contrat du candidat
//! ```rust,ignore
//! /// Compresse le tenseur aplati en un vecteur de scalaires (le format compressé).
//! /// Sa LONGUEUR est le nombre de paramètres stockés — mesuré par le harnais.
//! pub fn compress(flat_tensor: &[f64], shape: &[usize]) -> Vec<f64>;
//!
//! /// Reconstruit le tenseur À PARTIR DU SEUL format compressé.
//! pub fn reconstruct(compressed: &[f64], shape: &[usize], rebuilt: &mut [f64]);
//! ```
//!
//! ## Pourquoi deux fonctions
//! Le candidat ne *déclare* pas son nombre de paramètres (il pourrait mentir) :
//! le harnais le **mesure** comme `compressed.len()`, c.-à-d. la taille exacte
//! des données qui transitent vers `reconstruct`. Pour obtenir peu de
//! paramètres il faut réellement stocker peu *et* reconstruire fidèlement.
//!
//! ## Entrée : tenseur structuré tiré de la graine
//! Le harnais fabrique un tenseur de rang faible à partir de `trial.seed`
//! (compressible, donc l'objectif a un sens), mais non régénérable sans la
//! graine — sinon `reconstruct` recréerait l'original de zéro (même faille
//! que des entrées constantes).
//!
//! ## Pipeline d'évaluation
//! 1. `verify` — lint anti-état-global, puis compile et exécute sur un
//!    tenseur 4×4×4×4 tiré de la graine. Valide l'erreur L2 relative < tolérance.
//! 2. `measure` — exécution sur 8×8×8×8 pour [erreur_L2, latence_ns,
//!    paramètres_stockés], benchmark Criterion + filtrage thermique 4%.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use rand::rngs::StdRng;

use crate::candidate::CandidateId;
use crate::criterion_parser::parse_and_validate_metrics;
use crate::domain::{Domain, Score};
use crate::error::ForgeError;
use crate::isolation::run_with_secure_limits;
use crate::trial::Trial;

/// Constructions interdites dans le code candidat : elles permettent de faire
/// passer le tenseur de `compress` à `reconstruct` par un canal caché (état
/// global), contournant la mesure de `compressed.len()`. Un compresseur
/// numérique honnête n'en a aucun besoin.
const BANNED_GLOBAL_STATE: &[&str] = &[
    "thread_local",
    "lazy_static",
    "static mut",
    "OnceCell",
    "OnceLock",
    "AtomicPtr",
];

/// Heuristique de rejet des candidats à état global. N'est pas un bac à sable :
/// le blindage complet est l'isolation de `compress`/`reconstruct` en process
/// séparés. Mais cela bloque l'exploit réaliste à coût nul.
fn uses_global_state(source: &str) -> bool {
    BANNED_GLOBAL_STATE.iter().any(|needle| source.contains(needle))
}

/// Harnais `main.rs`. Code de confiance (le candidat ne fournit que la lib) :
/// il possède la graine, fabrique le tenseur, mesure `compressed.len()` et
/// calcule l'erreur. `__CRATE__`/`__N__`/`__DIMS__`/`__TOTAL__`/`__RANK__`/
/// `__SEED__`/`__TOL__` sont remplacés à la génération.
const VERIFY_MAIN_RS: &str = r#"use __CRATE__::{compress, reconstruct};

fn sm(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
fn rf(s: &mut u64) -> f64 { (sm(s) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0 }

// Tenseur CP de rang faible, derive de la graine : compressible mais non
// regenerable sans la graine (les cores sont caches au candidat).
fn gen_original(seed: u64, n: usize, dims: usize, rank: usize, total: usize) -> Vec<f64> {
    let mut s = seed ^ 0xA5A5A5A5A5A5A5A5u64;
    let mut cores: Vec<Vec<Vec<f64>>> = Vec::new();
    for _ in 0..rank {
        let mut modes = Vec::new();
        for _ in 0..dims {
            let v: Vec<f64> = (0..n).map(|_| rf(&mut s)).collect();
            modes.push(v);
        }
        cores.push(modes);
    }
    let mut out = vec![0.0f64; total];
    for idx in 0..total {
        let mut rem = idx;
        let mut digit = [0usize; 8];
        for m in (0..dims).rev() { digit[m] = rem % n; rem /= n; }
        let mut acc = 0.0;
        for t in 0..rank {
            let mut p = 1.0;
            for m in 0..dims { p *= cores[t][m][digit[m]]; }
            acc += p;
        }
        out[idx] = acc;
    }
    out
}

fn main() {
    let n: usize = __N__;
    let dims: usize = __DIMS__;
    let total: usize = __TOTAL__;
    let rank: usize = __RANK__;
    let seed: u64 = __SEED__;
    let shape: Vec<usize> = vec![n; dims];

    let original = gen_original(seed, n, dims, rank, total);

    // Le candidat compresse ; le harnais MESURE la taille stockee.
    let compressed = compress(&original, &shape);
    let params = compressed.len();

    // Le candidat reconstruit a partir du SEUL format compresse.
    let mut rebuilt = vec![0.0f64; total];
    reconstruct(&compressed, &shape, &mut rebuilt);

    let mut l2 = 0.0f64;
    let mut nrm = 0.0f64;
    for i in 0..total {
        let d = original[i] - rebuilt[i];
        l2 += d * d;
        nrm += original[i] * original[i];
    }
    let rel = l2.sqrt() / nrm.sqrt().max(1e-12);

    // Toujours emettre le nombre de parametres (mesure, pas declaration).
    println!("PARAMS={}", params);
    if rel > __TOL__ {
        eprintln!("L2_ERROR={:.6e}", rel);
        std::process::exit(101);
    }
    println!("L2_ERROR={:.12e}", rel);
    std::process::exit(0);
}
"#;

/// Benchmark `tt_bench.rs` : chronomètre la RECONSTRUCTION seule. La
/// compression est faite une fois hors mesure — sinon l'allocation de
/// `compress` à chaque itération gonflerait la variance.
const BENCH_RS: &str = r#"use criterion::{criterion_group, criterion_main, Criterion, black_box};
use __CRATE__::{compress, reconstruct};

fn bench_tensor_train(c: &mut Criterion) {
    let n: usize = __N__;
    let dims: usize = __DIMS__;
    let total: usize = __TOTAL__;
    let shape: Vec<usize> = vec![n; dims];
    let original: Vec<f64> = (0..total).map(|i| (i as f64).cos()).collect();
    // Compression effectuee une fois, hors mesure ; on chronometre la reconstruction.
    let compressed = compress(&original, &shape);
    let mut rebuilt = vec![0.0f64; total];
    c.bench_function("tt_target", |b_run| b_run.iter(|| {
        reconstruct(black_box(&compressed), black_box(&shape), black_box(&mut rebuilt));
    }));
}
criterion_group!(benches, bench_tensor_train);
criterion_main!(benches);
"#;

// ---------------------------------------------------------------------------
// Candidat : code source Rust
// ---------------------------------------------------------------------------

/// Représente le code d'un algorithme de compression Tensor Train
/// généré par le LLM ou le micro-mutateur.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorCode {
    pub raw_source: String,
    pub id: CandidateId,
}

impl crate::candidate::Candidate for TensorCode {
    fn id(&self) -> CandidateId {
        self.id
    }
    fn repr(&self) -> String {
        self.raw_source.clone()
    }
}

// ---------------------------------------------------------------------------
// Domaine Tensor Train
// ---------------------------------------------------------------------------

/// Domaine de compression Tensor Train avec compilation, exécution
/// réelle et benchmarking Criterion.
#[derive(Clone)]
pub struct TensorTrainDomain {
    pub workspace_root: PathBuf,
    pub max_mem: u64,
    pub max_disk: u64,
    pub compile_timeout: Duration,
    pub exec_timeout: Duration,
    pub bench_timeout: Duration,
    /// Rang du tenseur de test fabriqué par le harnais (difficulté de compression).
    pub tt_rank: usize,
    /// Tolérance d'erreur L2 **relative** pour la porte de correction.
    pub tolerance: f64,
}

impl TensorTrainDomain {
    pub fn new(workspace: &str) -> Self {
        TensorTrainDomain {
            workspace_root: PathBuf::from(workspace),
            max_mem: 4 * 1024 * 1024 * 1024, // 4 GiB
            max_disk: 50 * 1024 * 1024,      // 50 MiB
            compile_timeout: Duration::from_secs(45),
            exec_timeout: Duration::from_secs(10),
            bench_timeout: Duration::from_secs(60),
            tt_rank: 3,
            tolerance: 1e-3,
        }
    }

    /// Prépare un environnement Cargo isolé pour évaluer un candidat.
    /// `seed` rend le tenseur de test déterministe par génération.
    fn setup_candidate_env(
        &self,
        cand_id: CandidateId,
        source: &str,
        tensor_size: usize, // dimension par mode (ex: 4 pour 4×4×4×4)
        num_dims: usize,    // nombre de modes (ex: 4)
        seed: u64,
    ) -> crate::error::Result<PathBuf> {
        let cand_dir = self.workspace_root.join(format!("cand_{}", cand_id));
        let src_dir = cand_dir.join("src");
        let benches_dir = cand_dir.join("benches");
        fs::create_dir_all(&src_dir).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        fs::create_dir_all(&benches_dir).map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        let total_elements = tensor_size.pow(num_dims as u32);
        let crate_name = format!("cand_{}", cand_id);

        // 1. Cargo.toml
        let toml_path = cand_dir.join("Cargo.toml");
        let mut toml = fs::File::create(&toml_path).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        writeln!(
            toml,
            "[package]\n\
             name = \"cand_{}\"\n\
             version = \"0.1.0\"\n\
             edition = \"2021\"\n\n\
             [dependencies]\n\
             rand = \"0.8\"\n\n\
             [dev-dependencies]\n\
             criterion = \"0.5\"\n\n\
             [[bench]]\n\
             name = \"tt_bench\"\n\
             harness = false\n",
            cand_id
        )
        .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 2. src/lib.rs — le code du candidat
        let mut lib = fs::File::create(src_dir.join("lib.rs")).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        lib.write_all(source.as_bytes()).map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 3. src/main.rs — harnais : tenseur tiré de la graine + mesure de params + porte L2
        let main_src = VERIFY_MAIN_RS
            .replace("__CRATE__", &crate_name)
            .replace("__N__", &tensor_size.to_string())
            .replace("__DIMS__", &num_dims.to_string())
            .replace("__TOTAL__", &total_elements.to_string())
            .replace("__RANK__", &self.tt_rank.to_string())
            .replace("__SEED__", &seed.to_string())
            .replace("__TOL__", &format!("{:e}", self.tolerance));
        let mut main = fs::File::create(src_dir.join("main.rs")).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        main.write_all(main_src.as_bytes()).map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 4. benches/tt_bench.rs — benchmark Criterion
        let bench_src = BENCH_RS
            .replace("__CRATE__", &crate_name)
            .replace("__N__", &tensor_size.to_string())
            .replace("__DIMS__", &num_dims.to_string())
            .replace("__TOTAL__", &total_elements.to_string());
        let mut bench = fs::File::create(benches_dir.join("tt_bench.rs")).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        bench.write_all(bench_src.as_bytes()).map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        Ok(cand_dir)
    }

    fn clean_env(&self, cand_id: CandidateId) {
        let cand_dir = self.workspace_root.join(format!("cand_{}", cand_id));
        let _ = fs::remove_dir_all(cand_dir);
    }

    /// Extrait l'erreur L2 relative depuis le stdout du harnais.
    fn extract_l2_error(stdout: &str) -> Option<f64> {
        for line in stdout.lines() {
            if let Some(val_str) = line.strip_prefix("L2_ERROR=") {
                return val_str.trim().parse::<f64>().ok();
            }
        }
        None
    }

    /// Extrait le nombre de paramètres MESURÉ par le harnais (`compressed.len()`).
    fn extract_params(stdout: &str) -> Option<f64> {
        for line in stdout.lines() {
            if let Some(val_str) = line.strip_prefix("PARAMS=") {
                return val_str.trim().parse::<f64>().ok();
            }
        }
        None
    }
}

impl Domain for TensorTrainDomain {
    type Cand = TensorCode;

    fn name(&self) -> &str {
        "low_rank_compression"
    }

    fn seed(&self, _rng: &mut StdRng) -> Self::Cand {
        let baseline_src = r#"// Algorithme de reference : identite (aucune compression).
// Contrat : compress(flat, shape) -> Vec<f64> ; reconstruct(compressed, shape, rebuilt)
// L'identite stocke TOUT le tenseur : params = flat.len(), erreur = 0.
pub fn compress(flat_tensor: &[f64], _shape: &[usize]) -> Vec<f64> {
    flat_tensor.to_vec()
}

pub fn reconstruct(compressed: &[f64], _shape: &[usize], rebuilt: &mut [f64]) {
    for (i, &v) in compressed.iter().enumerate() {
        rebuilt[i] = v;
    }
}
"#;
        TensorCode {
            raw_source: baseline_src.to_string(),
            id: crate::fnv1a(baseline_src),
        }
    }

    fn mutate(&self, _rng: &mut StdRng, _parents: &[&Self::Cand]) -> crate::error::Result<Self::Cand> {
        Err(ForgeError::Evaluation(
            "Mutation brute non implémentée au niveau domaine — utiliser Engine + LlmMutator".into(),
        ))
    }

    /// PORTE DE CORRECTION : lint anti-état-global, puis compile et exécute sur
    /// un tenseur 4×4×4×4 tiré de `trial.seed`, vérifie l'erreur L2 relative.
    fn verify(&self, cand: &Self::Cand, trial: &Trial) -> crate::error::Result<bool> {
        // Lint : un candidat à état global peut contourner la mesure de params.
        if uses_global_state(&cand.raw_source) {
            return Ok(false);
        }

        let dim_size = 4usize;
        let num_dims = 4usize;

        let env_path = self.setup_candidate_env(cand.id, &cand.raw_source, dim_size, num_dims, trial.seed)?;

        // Étape 1 : Compilation supervisée
        let mut compile_cmd = Command::new("cargo");
        compile_cmd.arg("build").arg("--release").current_dir(&env_path);

        if run_with_secure_limits(compile_cmd, self.compile_timeout, self.max_mem, self.max_disk).is_err() {
            self.clean_env(cand.id);
            return Ok(false);
        }

        // Étape 2 : Exécution du harnais (exit 101 si l'erreur dépasse la tolérance)
        let mut run_cmd = Command::new("cargo");
        run_cmd.arg("run").arg("--release").current_dir(&env_path);

        let run_res = run_with_secure_limits(run_cmd, self.exec_timeout, self.max_mem, self.max_disk);
        self.clean_env(cand.id);

        match run_res {
            Ok(_stdout) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// ÉVALUATION DES OBJECTIFS : [erreur_L2_relative, latence_ns, paramètres_stockés]
    fn measure(&self, cand: &Self::Cand, trial: &Trial) -> crate::error::Result<Vec<f64>> {
        if uses_global_state(&cand.raw_source) {
            return Err(ForgeError::Evaluation(
                "Candidat rejeté : état global interdit (contournement de la mesure de paramètres)".into(),
            ));
        }

        let dim_size = 8usize;
        let num_dims = 4usize;

        let env_path = self.setup_candidate_env(cand.id, &cand.raw_source, dim_size, num_dims, trial.seed)?;

        // Compilation
        let mut compile_cmd = Command::new("cargo");
        compile_cmd.arg("build").arg("--release").current_dir(&env_path);

        if run_with_secure_limits(compile_cmd, self.compile_timeout, self.max_mem, self.max_disk).is_err() {
            self.clean_env(cand.id);
            return Err(ForgeError::Evaluation("Échec de compilation pour la mesure".into()));
        }

        // ── Objectifs 1 & 3 : erreur L2 relative + paramètres stockés (mesurés au même run) ──
        let mut run_cmd = Command::new("cargo");
        run_cmd.arg("run").arg("--release").current_dir(&env_path);

        let (l2_error, param_count) = match run_with_secure_limits(run_cmd, self.exec_timeout, self.max_mem, self.max_disk) {
            Ok(stdout) => {
                let l2 = Self::extract_l2_error(&stdout).unwrap_or(f64::INFINITY);
                let params = Self::extract_params(&stdout).unwrap_or(f64::INFINITY);
                (l2, params)
            }
            Err(_) => {
                self.clean_env(cand.id);
                return Err(ForgeError::Evaluation("Échec d'exécution durant la mesure L2".into()));
            }
        };

        // ── Objectif 2 : Latence via Criterion ──
        let mut bench_cmd = Command::new("cargo");
        bench_cmd.arg("bench").arg("--bench").arg("tt_bench").current_dir(&env_path);

        let latency_ns = match run_with_secure_limits(bench_cmd, self.bench_timeout, self.max_mem, self.max_disk) {
            Ok(_) => match parse_and_validate_metrics(&env_path, "tt_target", 0.04) {
                Ok(objs) => objs[0],
                Err(_) => {
                    self.clean_env(cand.id);
                    return Err(ForgeError::Evaluation(
                        "Mesure Criterion instable ou absente — bruit thermique probable".into(),
                    ));
                }
            },
            Err(_) => {
                self.clean_env(cand.id);
                return Err(ForgeError::Evaluation("Échec du benchmark Criterion".into()));
            }
        };

        self.clean_env(cand.id);

        // Les 3 objectifs à minimiser
        Ok(vec![l2_error, latency_ns, param_count])
    }

    fn objective_names(&self) -> Vec<String> {
        vec![
            "reconstruction_error_L2".into(),
            "latency_ns".into(),
            "parameters_count".into(),
        ]
    }

    fn baseline(&self, _trial: &Trial) -> crate::error::Result<Score> {
        // Baseline : identité (stocke tout → params = 8^4 = 4096, erreur ~0).
        Ok(Score::valid(vec![0.0, 5000.0, 4096.0]))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_seed_is_valid() {
        let domain = TensorTrainDomain::new("/tmp/forge_tt_test_v4");
        let mut rng = StdRng::seed_from_u64(42);
        let cand = domain.seed(&mut rng);
        assert!(cand.raw_source.contains("compress"));
        assert!(cand.raw_source.contains("reconstruct"));
        assert!(cand.id > 0);
    }

    #[test]
    fn test_domain_name() {
        let domain = TensorTrainDomain::new("/tmp/irrelevant");
        assert_eq!(domain.name(), "low_rank_compression");
    }

    #[test]
    fn test_objective_names() {
        let domain = TensorTrainDomain::new("/tmp/irrelevant");
        let names = domain.objective_names();
        assert_eq!(names.len(), 3);
        assert_eq!(names[0], "reconstruction_error_L2");
        assert_eq!(names[1], "latency_ns");
        assert_eq!(names[2], "parameters_count");
    }

    #[test]
    fn test_extract_l2_error_from_stdout() {
        let stdout = "PARAMS=17\nL2_ERROR=1.234567890123e-05\n";
        let err = TensorTrainDomain::extract_l2_error(stdout);
        assert!(err.is_some());
        assert!((err.unwrap() - 1.234567890123e-05).abs() < 1e-15);
    }

    #[test]
    fn test_extract_l2_error_missing() {
        let stdout = "No error here\n";
        assert!(TensorTrainDomain::extract_l2_error(stdout).is_none());
    }

    #[test]
    fn test_extract_params_from_stdout() {
        let stdout = "PARAMS=33\nL2_ERROR=1.0e-09\n";
        let p = TensorTrainDomain::extract_params(stdout);
        assert!(p.is_some());
        assert_eq!(p.unwrap(), 33.0);
    }

    #[test]
    fn test_lint_rejects_global_state() {
        assert!(uses_global_state("thread_local! { static X: u32 = 0; }"));
        assert!(uses_global_state("static mut COUNTER: i64 = 0;"));
        assert!(!uses_global_state("pub fn compress(f: &[f64], _s: &[usize]) -> Vec<f64> { f.to_vec() }"));
    }
}
