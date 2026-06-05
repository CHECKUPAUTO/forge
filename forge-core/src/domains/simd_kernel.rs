//! Domaine d'optimisation de micro-kernels mathématiques réels (GEMM)
//! ciblant l'auto-vectorisation agressive par LLVM.
//!
//! ## Candidat
//! `compute_kernel(c: &mut [f64], a: &[f64], b: &[f64], n: usize)`
//! — multiplication matricielle `C = A × B` pour matrices carrées `n × n`.
//!
//! ## Pipeline
//! 1. Compilation avec `RUSTFLAGS="-C target-cpu=native"`
//! 2. Vérification mathématique : matrices `A`, `B` ALÉATOIRES tirées de la
//!    graine d'essai (`trial.seed`), comparées élément par élément à la
//!    référence naïve calculée dans le harnais (tolérance `1e-7`). Les entrées
//!    tournent à chaque génération : un kernel faux-mais-rapide (ex. `c[i]=n`)
//!    ne peut pas passer.
//! 3. Benchmark Criterion `cargo bench --bench gemm_hot`
//! 4. Parsing `target/criterion/gemm_target/new/estimates.json`

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::candidate::{Candidate, CandidateId};
use crate::criterion_parser::parse_and_validate_metrics;
use crate::domain::{Domain, Score};
use crate::error::{ForgeError, Result};
use crate::isolation::run_with_timeout;
use crate::trial::Trial;

/// Harnais `main.rs` de vérification. Le candidat ne fournit que
/// `compute_kernel` (dans `lib.rs`) ; ce harnais — qu'il ne contrôle pas —
/// possède les entrées, la référence et la comparaison. `__SIZE__`/`__SEED__`
/// sont remplacés à la génération.
const VERIFY_MAIN: &str = r#"use gemm_bench::compute_kernel;

fn rng_next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
fn rng_f64(s: &mut u64) -> f64 { (rng_next(s) >> 11) as f64 / (1u64 << 53) as f64 }

fn reference(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n { acc += a[i * n + k] * b[k * n + j]; }
            c[i * n + j] = acc;
        }
    }
}

fn main() {
    let n: usize = __SIZE__;
    let mut s: u64 = __SEED__u64 ^ 0xD1B5_4A32_D192_ED03;
    let a: Vec<f64> = (0..n * n).map(|_| rng_f64(&mut s) * 2.0 - 1.0).collect();
    let b: Vec<f64> = (0..n * n).map(|_| rng_f64(&mut s) * 2.0 - 1.0).collect();
    let mut c_ref = vec![0.0f64; n * n];
    reference(&mut c_ref, &a, &b, n);
    let mut c = vec![0.0f64; n * n];
    for x in c.iter_mut() { *x = 1.0e30; } // pre-remplissage : un kernel correct DOIT ecraser/zeroer, pas accumuler
    compute_kernel(&mut c, &a, &b, n);
    let mut max_diff = 0.0f64;
    for i in 0..n * n {
        let d = (c_ref[i] - c[i]).abs();
        if d > max_diff { max_diff = d; }
    }
    if max_diff > 1e-7 {
        eprintln!("MAXDIFF={:e}", max_diff);
        std::process::exit(101);
    }
    std::process::exit(0);
}
"#;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SimdKernelCode {
    pub source: String,
    pub id: CandidateId,
}

impl Candidate for SimdKernelCode {
    fn id(&self) -> CandidateId {
        self.id
    }
    fn repr(&self) -> String {
        self.source.clone()
    }
}

pub struct SimdKernelDomain {
    pub workspace_root: PathBuf,
    pub compile_timeout: Duration,
    pub exec_timeout: Duration,
    #[cfg(feature = "llm")]
    pub llm: Option<crate::mutation::llm_mutator::LlmMutator>,
}

impl SimdKernelDomain {
    pub fn new(root: &str) -> Self {
        Self {
            workspace_root: PathBuf::from(root),
            compile_timeout: Duration::from_secs(30),
            exec_timeout: Duration::from_secs(10),
            #[cfg(feature = "llm")]
            llm: None,
        }
    }

    /// Active la mutation guidee par LLM (Ollama) : few-shot (GEMM correct ET plus rapide) + gabarit vectorisable.
    #[cfg(feature = "llm")]
    pub fn with_llm(mut self, endpoint: &str, model: &str) -> Self {
        const OBJ: &str = r#"OBJECTIF: VITESSE. compute_kernel(c: &mut [f64], a: &[f64], b: &[f64], n: usize) calcule le produit matriciel C = A x B pour des matrices carrees n x n en row-major. Le score est `latency_ns` (latence mesuree par Criterion), plus petit = meilleur. Le kernel DOIT rester correct : il est compare element par element a la reference naive sur des entrees aleatoires (tolerance 1e-7), donc un kernel rapide mais FAUX est rejete. Garde EXACTEMENT cette signature publique.

DEPENDANCES (AUCUNE autre): uniquement la bibliotheque standard `std`. Pas de crate externe (ni rayon, ni nalgebra, ni ndarray), sinon le code ne compile pas. Compilation avec `-C target-cpu=native -C opt-level=3` : l'auto-vectorisation LLVM est active, structure les boucles pour qu'elle opere (boucle interne contigue en memoire, accumulation dans un buffer contigu, eviter les dependances portees ; bloquer par tuiles ou derouler si utile).

EXEMPLE correct et PLUS RAPIDE que la naive (ordre de boucle i-k-j : la boucle interne sur j est contigue en memoire pour b et c, donc bien vectorisee). Inspire-toi en -- tu peux bloquer par tuiles, derouler, ou pre-zeroer autrement -- mais garde EXACTEMENT cette signature:

#[inline(never)]
pub fn compute_kernel(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    for x in c.iter_mut() { *x = 0.0; }
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
}"#;
        const OBJ_WEAK: &str = r#"OBJECTIF: VITESSE. compute_kernel(c: &mut [f64], a: &[f64], b: &[f64], n: usize) calcule le produit matriciel C = A x B (matrices carrees n x n, stockage row-major). Score = latency_ns mesure par Criterion, plus petit = meilleur. Le kernel DOIT rester correct : compare element par element a la reference naive sur entrees aleatoires (tolerance 1e-7), un kernel rapide mais FAUX est rejete. Garde EXACTEMENT cette signature publique. IMPORTANT: a l'entree c peut contenir des valeurs ARBITRAIRES (pas forcement zero) ; le kernel doit produire EXACTEMENT C = A*B, donc ecrase/initialise tout c (ne PAS accumuler dans son contenu initial).

DEPENDANCES: uniquement la bibliotheque standard `std` (aucune crate externe, sinon ne compile pas). Compilation avec -C target-cpu=native -C opt-level=3.

Voici l'implementation NAIVE actuelle (le baseline a battre). Rends-la plus rapide en exploitant la localite memoire (cache) et l'auto-vectorisation LLVM, sans changer le resultat numerique ni la signature :

#[inline(never)]
pub fn compute_kernel(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n { acc += a[i * n + k] * b[k * n + j]; }
            c[i * n + j] = acc;
        }
    }
}"#;
        const OBJ_MID: &str = r#"OBJECTIF: VITESSE. compute_kernel(c: &mut [f64], a: &[f64], b: &[f64], n: usize) calcule le produit matriciel C = A x B (matrices carrees n x n, row-major). Score = latency_ns (Criterion), plus petit = meilleur. Correction OBLIGATOIRE : compare element par element a la reference naive sur entrees aleatoires (tolerance 1e-7), un kernel FAUX est rejete. A l'entree, c peut contenir des valeurs ARBITRAIRES (pas forcement zero) : le kernel doit produire EXACTEMENT C=A*B, donc ecraser/initialiser tout c. Garde EXACTEMENT cette signature publique.
DEPENDANCES: uniquement `std` (aucune crate externe, sinon ne compile pas). Compilation -C target-cpu=native -C opt-level=3 : l'auto-vectorisation LLVM est active.
Implementation NAIVE actuelle (baseline a battre) :
#[inline(never)]
pub fn compute_kernel(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n { acc += a[i * n + k] * b[k * n + j]; }
            c[i * n + j] = acc;
        }
    }
}
DIAGNOSTIC du probleme (sans te donner la solution) : dans la boucle interne sur k, l'acces b[k * n + j] avance de n elements a chaque iteration (une colonne de b en stockage row-major) -> sauts en memoire, mauvaise localite de cache, et le compilateur ne peut PAS auto-vectoriser cette boucle (accumulateur scalaire + acces non contigu a b). a[i * n + k] lui est contigu ; c'est l'acces a b qui tue la performance.
A TOI de reorganiser le calcul pour que la boucle la plus INTERNE parcoure la memoire de façon CONTIGUE (acces a b avec un stride de 1) et devienne vectorisable, tout en produisant le bon resultat dans c (reflechis a comment et quand initialiser c selon l'ordre de boucle que tu choisis). Garde EXACTEMENT la signature."#;
        let fewshot = std::env::var("FORGE_FEWSHOT").unwrap_or_default();
        let objective = match fewshot.as_str() {
            "weak" => OBJ_WEAK,
            "mid" => OBJ_MID,
            _ => OBJ,
        };
        self.llm = Some(
            crate::mutation::llm_mutator::LlmMutator::new(endpoint, model)
                .with_timeout(600)
                .with_objective(objective),
        );
        self
    }

    fn write_environment(
        &self,
        cand_id: CandidateId,
        raw_code: &str,
        size: usize,
        seed: u64,
    ) -> Result<PathBuf> {
        static ENV_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nonce = ENV_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let env_path = self
            .workspace_root
            .join(format!("simd_gemm_{}_{}", cand_id, nonce));
        let src_path = env_path.join("src");
        fs::create_dir_all(&src_path).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        fs::create_dir_all(env_path.join("benches"))
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 1. Cargo.toml
        let mut toml =
            File::create(env_path.join("Cargo.toml")).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        writeln!(
            toml,
            "[package]\nname = \"gemm_bench\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
             [dependencies]\n\n\
             [dev-dependencies]\ncriterion = \"0.5\"\n\n\
             [[bench]]\nname = \"gemm_hot\"\nharness = false\n",
        )
        .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 2. lib.rs — code candidat
        let mut lib =
            File::create(src_path.join("lib.rs")).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        writeln!(lib, "#![allow(dead_code)]\n{}", raw_code)
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 3. main.rs — validation par entrées ALÉATOIRES + comparaison à la référence.
        let main_src = VERIFY_MAIN
            .replace("__SIZE__", &size.to_string())
            .replace("__SEED__", &seed.to_string());
        let mut main =
            File::create(src_path.join("main.rs")).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        main.write_all(main_src.as_bytes())
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 4. benches/gemm_hot.rs — benchmark Criterion (entrées variées, pas constantes).
        let mut bench = File::create(env_path.join("benches").join("gemm_hot.rs"))
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        writeln!(
            bench,
            "use criterion::{{criterion_group, criterion_main, Criterion, black_box}};\n\
             use gemm_bench::compute_kernel;\n\n\
             fn bench_gemm(c: &mut Criterion) {{\n\
                 let n = {size};\n\
                 let a: Vec<f64> = (0..n * n).map(|i| ((i as f64) * 0.1).sin()).collect();\n\
                 let b: Vec<f64> = (0..n * n).map(|i| ((i as f64) * 0.1).cos()).collect();\n\
                 let mut out = vec![0.0f64; n * n];\n\
                 c.bench_function(\"gemm_target\", |b_run| b_run.iter(|| {{\n\
                     compute_kernel(\n\
                         black_box(&mut out),\n\
                         black_box(&a),\n\
                         black_box(&b),\n\
                         black_box(n),\n\
                     );\n\
                 }}));\n\
             }}\n\
             criterion_group!(benches, bench_gemm);\n\
             criterion_main!(benches);\n",
        )
        .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        Ok(env_path)
    }
}

impl Domain for SimdKernelDomain {
    type Cand = SimdKernelCode;

    fn name(&self) -> &str {
        "simd_gemm"
    }

    fn seed(&self, _rng: &mut StdRng) -> Self::Cand {
        let base_gemm = r#"#[inline(never)]
pub fn compute_kernel(c: &mut [f64], a: &[f64], b: &[f64], n: usize) {
    // Implementation naive de reference (Baseline a battre par auto-vectorisation)
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}
"#;
        SimdKernelCode {
            source: base_gemm.to_string(),
            id: crate::fnv1a(base_gemm),
        }
    }

    fn mutate(&self, _rng: &mut StdRng, parents: &[&Self::Cand]) -> Result<Self::Cand> {
        #[cfg(feature = "llm")]
        {
            if let Some(ref llm) = self.llm {
                use crate::candidate::Candidate as _;
                let parent_src = parents.first().map(|p| p.repr()).unwrap_or_default();
                let new_src = match llm.mutate_with_feedback(&parent_src, None) {
                    Ok(src) => src,
                    Err(e) => {
                        if std::env::var("FORGE_VERBOSE").is_ok() {
                            eprintln!("[forge:mutate] echec LLM ({e}) -> repli sur le parent");
                        }
                        parent_src.clone()
                    }
                };
                if std::env::var("FORGE_VERBOSE").is_ok() {
                    eprintln!("[forge:mutate] LLM -> {} octets de source", new_src.len());
                }
                let id = crate::fnv1a(&new_src);
                return Ok(SimdKernelCode { source: new_src, id });
            }
        }
        let _ = parents;
        Err(ForgeError::Evaluation(
            "Mutation LLM indisponible -- compiler avec --features llm".into(),
        ))
    }

    /// Porte de correction : compile, puis exécute le harnais qui compare la
    /// sortie du kernel à la référence naïve sur des entrées ALÉATOIRES tirées
    /// de `trial.seed`. Un kernel incorrect (même rapide) échoue.
    fn verify(&self, cand: &Self::Cand, trial: &Trial) -> Result<bool> {
        let size = 64;
        let env_path = self.write_environment(cand.id, &cand.source, size, trial.seed)?;

        // Étape 1 : Compilation avec optimisations natives
        let mut comp_cmd = Command::new("cargo");
        comp_cmd
            .arg("build")
            .arg("--release")
            .current_dir(&env_path)
            .env("RUSTFLAGS", "-C target-cpu=native -C opt-level=3");

        if run_with_timeout(comp_cmd, self.compile_timeout).is_err() {
            let _ = fs::remove_dir_all(&env_path);
            return Ok(false);
        }

        // Étape 2 : Exécution de la conformité numérique (exit 101 si faux)
        let mut run_cmd = Command::new("cargo");
        run_cmd
            .arg("run")
            .arg("--release")
            .current_dir(&env_path)
            .env("RUSTFLAGS", "-C target-cpu=native -C opt-level=3");

        let run_res = run_with_timeout(run_cmd, self.exec_timeout);
        let _ = fs::remove_dir_all(&env_path);

        match run_res {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Mesure de performance réelle : latence via Criterion.
    fn measure(&self, cand: &Self::Cand, trial: &Trial) -> Result<Vec<f64>> {
        let size = 64;
        let env_path = self.write_environment(cand.id, &cand.source, size, trial.seed)?;

        let mut bench_cmd = Command::new("cargo");
        bench_cmd
            .arg("bench")
            .arg("--bench")
            .arg("gemm_hot")
            .current_dir(&env_path)
            .env("RUSTFLAGS", "-C target-cpu=native -C opt-level=3");

        let bench_res = run_with_timeout(bench_cmd, Duration::from_secs(45));

        if bench_res.is_err() {
            let _ = fs::remove_dir_all(&env_path);
            return Err(ForgeError::Evaluation(
                "Échec d'exécution du benchmark de performance".into(),
            ));
        }

        // Parse et valide les métriques Criterion avec seuil de variance thermique 4%
        let validated = parse_and_validate_metrics(&env_path, "gemm_target", 0.04)?;
        let latency_ns = validated[0];
        let _ = fs::remove_dir_all(&env_path);

        Ok(vec![latency_ns])
    }

    fn objective_names(&self) -> Vec<String> {
        vec!["latency_ns".into()]
    }

    fn baseline(&self, trial: &Trial) -> Result<Score> {
        let base = self.seed(&mut StdRng::seed_from_u64(0));
        let objs = self.measure(&base, trial)?;
        Ok(Score::valid(objs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_is_valid_gemm() {
        let domain = SimdKernelDomain::new("/tmp/forge_simd_v3");
        let mut rng = StdRng::seed_from_u64(42);
        let cand = domain.seed(&mut rng);
        assert!(cand.source.contains("compute_kernel"));
        assert!(cand.source.contains("acc += a[i * n + k] * b[k * n + j]"));
    }

    #[test]
    fn test_domain_name() {
        let domain = SimdKernelDomain::new("/tmp/irrelevant");
        assert_eq!(domain.name(), "simd_gemm");
    }
}
