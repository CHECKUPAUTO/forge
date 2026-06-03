//! Domaine de compression Tensor Train (Low-Rank / MPS) pour SciRust.
//!
//! Ce domaine génère, compile et évalue des décompositions de tenseurs
//! d'ordre N en utilisant une factorisation en rangs faibles (Tensor Train).
//! Chaque candidat est du code Rust brut (`TensorCode`) compilé dans un
//! répertoire isolé avec limites rlimit, benchmarké via Criterion, et
//! validé statistiquement par `criterion_parser::parse_and_validate_metrics`.
//!
//! ## Signature du candidat
//! ```rust,ignore
//! pub fn deconstruct_and_reconstruct(flat_tensor: &[f64], shape: &[usize], rebuilt: &mut [f64]);
//! ```
//!
//! ## Pipeline d'évaluation
//! 1. `verify` — compile et exécute sur un tenseur 4×4×4×4 connu.
//!    Valide que l'erreur de reconstruction L2 < 1e-4.
//! 2. `measure` — benchmark Criterion sur 8×8×8×8, parsing + filtrage
//!    thermique à 4%. Retourne [erreur_L2, latence_ns, taille_paramètres].

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
    /// Seuil de tolérance pour la vérification mathématique (L2).
    pub tolerance: f64,
}

impl TensorTrainDomain {
    pub fn new(workspace: &str) -> Self {
        TensorTrainDomain {
            workspace_root: PathBuf::from(workspace),
            max_mem: 4 * 1024 * 1024 * 1024, // 4 GiB
            max_disk: 50 * 1024 * 1024,       // 50 MiB
            compile_timeout: Duration::from_secs(45),
            exec_timeout: Duration::from_secs(10),
            bench_timeout: Duration::from_secs(60),
            tolerance: 1e-4,
        }
    }

    /// Prépare un environnement Cargo isolé pour évaluer un candidat.
    /// Crée un projet avec `lib.rs` (code candidat), `main.rs` (harnais
    /// de vérification), et `benches/tt_bench.rs` (benchmark Criterion).
    fn setup_candidate_env(
        &self,
        cand_id: CandidateId,
        source: &str,

        tensor_size: usize,   // dimension par mode (ex: 4 pour 4×4×4×4)
        num_dims: usize,      // nombre de modes (ex: 4)
    ) -> crate::error::Result<PathBuf> {
        let cand_dir = self.workspace_root.join(format!("cand_{}", cand_id));
        let src_dir = cand_dir.join("src");
        let benches_dir = cand_dir.join("benches");
        fs::create_dir_all(&src_dir)
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        fs::create_dir_all(&benches_dir)
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 1. Cargo.toml
        let toml_path = cand_dir.join("Cargo.toml");
        let mut toml = fs::File::create(&toml_path)
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;
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
        let mut lib = fs::File::create(src_dir.join("lib.rs"))
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        lib.write_all(source.as_bytes())
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 3. src/main.rs — harnais de vérification mathématique
        let total_elements = tensor_size.pow(num_dims as u32);
        let mut main = fs::File::create(src_dir.join("main.rs"))
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        writeln!(
            main,
            "use cand_{cand_id}::deconstruct_and_reconstruct;\n\n\
             fn main() {{\n\
                 let shape: Vec<usize> = vec!{shape:?};\n\
                 let total: usize = {total_elements};\n\
                 let original: Vec<f64> = (0..total).map(|i| (i as f64).sin()).collect();\n\
                 let mut rebuilt = vec![0.0f64; total];\n\
                 deconstruct_and_reconstruct(&original, &shape, &mut rebuilt);\n\
                 let mut l2_err = 0.0f64;\n\
                 for i in 0..total {{\n\
                     let diff = original[i] - rebuilt[i];\n\
                     l2_err += diff * diff;\n\
                 }}\n\
                 l2_err = l2_err.sqrt();\n\
                 if l2_err > {tolerance:e} {{\n\
                     eprintln!(\"L2_ERROR={{:.6e}}\", l2_err);\n\
                     std::process::exit(101);\n\
                 }}\n\
                 // Affiche l'erreur pour capture par le harnais\n\
                 println!(\"L2_ERROR={{:.12e}}\", l2_err);\n\
                 std::process::exit(0);\n\
             }}",
            cand_id = cand_id,
            shape = vec![tensor_size; num_dims],
            total_elements = total_elements,
            tolerance = self.tolerance,
        )
        .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 4. benches/tt_bench.rs — benchmark Criterion
        let mut bench = fs::File::create(benches_dir.join("tt_bench.rs"))
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        writeln!(
            bench,
            "use criterion::{{criterion_group, criterion_main, Criterion, black_box}};\n\
             use cand_{cand_id}::deconstruct_and_reconstruct;\n\n\
             fn bench_tensor_train(c: &mut Criterion) {{\n\
                 let shape: Vec<usize> = vec!{shape:?};\n\
                 let total: usize = {total_elements};\n\
                 let original: Vec<f64> = (0..total).map(|i| (i as f64).cos()).collect();\n\
                 let mut rebuilt = vec![0.0f64; total];\n\
                 c.bench_function(\"tt_target\", |b_run| b_run.iter(|| {{\n\
                     deconstruct_and_reconstruct(\n\
                         black_box(&original),\n\
                         black_box(&shape),\n\
                         black_box(&mut rebuilt),\n\
                     );\n\
                 }}));\n\
             }}\n\
             criterion_group!(benches, bench_tensor_train);\n\
             criterion_main!(benches);\n",
            cand_id = cand_id,
            shape = vec![tensor_size; num_dims],
            total_elements = total_elements,
        )
        .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        Ok(cand_dir)
    }

    fn clean_env(&self, cand_id: CandidateId) {
        let cand_dir = self.workspace_root.join(format!("cand_{}", cand_id));
        let _ = fs::remove_dir_all(cand_dir);
    }

    /// Extrait l'erreur L2 à partir de la sortie stdout du harnais de vérification.
    fn extract_l2_error(stdout: &str) -> Option<f64> {
        for line in stdout.lines() {
            if line.starts_with("L2_ERROR=") {
                if let Some(val_str) = line.strip_prefix("L2_ERROR=") {
                    return val_str.trim().parse::<f64>().ok();
                }
            }
        }
        None
    }

    /// Estime la taille des paramètres (en nombre d'éléments) à partir de
    /// la taille du code source (proxy pour la complexité du format de
    /// stockage compressé).
    fn estimate_parameter_size(source: &str) -> f64 {
        // Heuristique : la taille du code source compressé est corrélée
        // à la taille des structures de données internes du format.
        // Nombre de lignes significatives (hors commentaires et lignes vides).
        let significant_lines = source
            .lines()
            .filter(|l| {
                let trimmed = l.trim();
                !trimmed.is_empty() && !trimmed.starts_with("//") && !trimmed.starts_with("/*")
            })
            .count();
        significant_lines as f64
    }
}

impl Domain for TensorTrainDomain {
    type Cand = TensorCode;

    fn name(&self) -> &str {
        "low_rank_compression"
    }

    fn seed(&self, _rng: &mut StdRng) -> Self::Cand {
        let baseline_src = r#"// Algorithme de référence : identité brute (pas de compression).
// Signature : deconstruct_and_reconstruct(flat, shape, rebuilt)
// Cette baseline ne compresse pas du tout — elle recopie.
pub fn deconstruct_and_reconstruct(flat_tensor: &[f64], _shape: &[usize], rebuilt: &mut [f64]) {
    for (i, &v) in flat_tensor.iter().enumerate() {
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

    /// PORTE DE CORRECTION : compile et exécute sur un tenseur 4×4×4×4
    /// connu, vérifie que l'erreur de reconstruction L2 est < tolerance.
    fn verify(&self, cand: &Self::Cand, trial: &Trial) -> crate::error::Result<bool> {
        let _rng = trial.rng();
        let dim_size = 4usize;
        let num_dims = 4usize;

        let env_path = self.setup_candidate_env(
            cand.id,
            &cand.raw_source,
            dim_size,
            num_dims,
        )?;

        // Étape 1 : Compilation supervisée
        let mut compile_cmd = Command::new("cargo");
        compile_cmd
            .arg("build")
            .arg("--release")
            .current_dir(&env_path);

        if run_with_secure_limits(
            compile_cmd,
            self.compile_timeout,
            self.max_mem,
            self.max_disk,
        )
        .is_err()
        {
            self.clean_env(cand.id);
            return Ok(false);
        }

        // Étape 2 : Exécution du harnais de vérification mathématique
        let mut run_cmd = Command::new("cargo");
        run_cmd
            .arg("run")
            .arg("--release")
            .current_dir(&env_path);

        let run_res = run_with_secure_limits(
            run_cmd,
            self.exec_timeout,
            self.max_mem,
            self.max_disk,
        );
        self.clean_env(cand.id);

        match run_res {
            Ok(_stdout) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// ÉVALUATION DES OBJECTIFS : [erreur_L2, latence_ns, taille_paramètres]
    ///
    /// 1. Exécute le harnais de vérification sur 8×8×8×8 pour extraire
    ///    l'erreur de reconstruction L2 réelle.
    /// 2. Lance le benchmark Criterion (`cargo bench`) et parse les
    ///    métriques via `criterion_parser::parse_and_validate_metrics`
    ///    avec un seuil de variance maximale de 4%.
    /// 3. Estime la taille des paramètres à partir du code source.
    fn measure(&self, cand: &Self::Cand, _trial: &Trial) -> crate::error::Result<Vec<f64>> {
        let dim_size = 8usize;
        let num_dims = 4usize;

        let env_path = self.setup_candidate_env(
            cand.id,
            &cand.raw_source,
            dim_size,
            num_dims,
        )?;

        // ── Objectif 1 : Erreur de reconstruction L2 ──
        // Compilation (si pas déjà faite — on recompile pour la taille 8)
        let mut compile_cmd = Command::new("cargo");
        compile_cmd
            .arg("build")
            .arg("--release")
            .current_dir(&env_path);

        if run_with_secure_limits(
            compile_cmd,
            self.compile_timeout,
            self.max_mem,
            self.max_disk,
        )
        .is_err()
        {
            self.clean_env(cand.id);
            return Err(ForgeError::Evaluation(
                "Échec de compilation pour la mesure".into(),
            ));
        }

        // Exécution du binaire pour extraire l'erreur L2
        let mut run_cmd = Command::new("cargo");
        run_cmd
            .arg("run")
            .arg("--release")
            .current_dir(&env_path);

        let l2_error = match run_with_secure_limits(
            run_cmd,
            self.exec_timeout,
            self.max_mem,
            self.max_disk,
        ) {
            Ok(stdout) => {
                Self::extract_l2_error(&stdout).unwrap_or(f64::INFINITY)
            }
            Err(_) => {
                self.clean_env(cand.id);
                return Err(ForgeError::Evaluation(
                    "Échec d'exécution durant la mesure L2".into(),
                ));
            }
        };

        // ── Objectif 2 : Latence via Criterion ──
        let mut bench_cmd = Command::new("cargo");
        bench_cmd
            .arg("bench")
            .arg("--bench")
            .arg("tt_bench")
            .current_dir(&env_path);

        let bench_res = run_with_secure_limits(
            bench_cmd,
            self.bench_timeout,
            self.max_mem,
            self.max_disk,
        );

        let latency_ns = match bench_res {
            Ok(_) => {
                // Parse et valide les métriques Criterion avec seuil de 4%
                match parse_and_validate_metrics(&env_path, "tt_target", 0.04) {
                    Ok(objs) => objs[0], // mean_latency_ns
                    Err(_) => {
                        self.clean_env(cand.id);
                        return Err(ForgeError::Evaluation(
                            "Mesure Criterion instable ou absente — \
                             bruit thermique probable".into(),
                        ));
                    }
                }
            }
            Err(_) => {
                self.clean_env(cand.id);
                return Err(ForgeError::Evaluation(
                    "Échec du benchmark Criterion".into(),
                ));
            }
        };

        self.clean_env(cand.id);

        // ── Objectif 3 : Taille des paramètres ──
        let param_size = Self::estimate_parameter_size(&cand.raw_source);

        // Les 3 objectifs à minimiser
        Ok(vec![l2_error, latency_ns, param_size])
    }

    fn objective_names(&self) -> Vec<String> {
        vec![
            "reconstruction_error_L2".into(),
            "latency_ns".into(),
            "parameters_count".into(),
        ]
    }

    fn baseline(&self, _trial: &Trial) -> crate::error::Result<Score> {
        // Baseline : identité (pas de compression, erreur ~0, latence fixe)
        Ok(Score::valid(vec![0.0, 5000.0, 10.0]))
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
        assert!(cand.raw_source.contains("deconstruct_and_reconstruct"));
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
        let stdout = "Some preamble\nL2_ERROR=1.234567890123e-05\nSome epilogue\n";
        let err = TensorTrainDomain::extract_l2_error(stdout);
        assert!(err.is_some());
        assert!((err.unwrap() - 1.234567890123e-05).abs() < 1e-15);
    }

    #[test]
    fn test_extract_l2_error_missing() {
        let stdout = "No error here\n";
        let err = TensorTrainDomain::extract_l2_error(stdout);
        assert!(err.is_none());
    }

    #[test]
    fn test_estimate_parameter_size() {
        let source = "// Comment\npub fn foo() {\n    let x = 1;\n    let y = 2;\n}\n";
        let size = TensorTrainDomain::estimate_parameter_size(source);
        // 3 significant lines: "pub fn foo() {", "let x = 1;", "let y = 2;", "}" => 4
        assert!(size >= 3.0);
    }

    #[test]
    fn test_estimate_parameter_size_empty() {
        let size = TensorTrainDomain::estimate_parameter_size("// only comments\n\n");
        assert_eq!(size, 0.0);
    }
}
