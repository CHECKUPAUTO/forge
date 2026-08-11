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

use rand::{rngs::StdRng, SeedableRng};

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
const BANNED_CAPABILITIES: &[&str] = &[
    "thread_local",
    "lazy_static",
    "static mut",
    "OnceCell",
    "OnceLock",
    "AtomicPtr",
    "std::fs",
    "File",
    "OpenOptions",
    "std::net",
    "TcpStream",
    "UdpSocket",
    "UnixStream",
    "std::process",
    "Command",
    "std::env",
    "std::path",
    "PathBuf",
    "include_bytes",
    "include_str",
    "unsafe",
    "extern \"C\"",
];

/// Défense en profondeur contre les canaux cachés évidents. Ceci n'est PAS
/// une frontière de sécurité : les candidats non fiables doivent toujours être
/// exécutés dans un sandbox OS externe au worker Forge.
fn uses_forbidden_capability(source: &str) -> bool {
    BANNED_CAPABILITIES
        .iter()
        .any(|needle| source.contains(needle))
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
    #[cfg(feature = "llm")]
    pub llm: Option<crate::mutation::llm_mutator::LlmMutator>,
    #[cfg(all(feature = "llm", feature = "bandit"))]
    pub bandit: std::sync::Arc<std::sync::Mutex<crate::mutation::bandit::MutationBandit>>,
    /// Index of objective to use as primary for bandit reward (index into `objective_names`).
    pub reward_objective_idx: usize,
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
            #[cfg(feature = "llm")]
            llm: None,
            #[cfg(all(feature = "llm", feature = "bandit"))]
            bandit: std::sync::Arc::new(std::sync::Mutex::new(
                crate::mutation::bandit::MutationBandit::new(
                    crate::mutation::llm_mutator::LlmMutator::new("", ""),
                    vec![String::new()], // placeholder — overwritten by with_llm
                ),
            )),
            reward_objective_idx: 2, // compression objective: parameters_count
        }
    }

    /// Prépare un environnement Cargo isolé pour évaluer un candidat.
    /// `seed` rend le tenseur de test déterministe par génération.
    /// Active la mutation guidee par LLM (Ollama) : few-shot selectionne via FORGE_FEWSHOT.
    #[cfg(feature = "llm")]
    pub fn with_llm(mut self, endpoint: &str, model: &str) -> Self {
        /// Full objectif : SVD complete avec exemple de solution compresse.
        const OBJ_FULL: &str = r#"OBJECTIF: COMPRESSION. compress(flat_tensor: &[f64], shape: &[usize]) -> Vec<f64> doit renvoyer le MOINS de valeurs possible -- c'est le score `parameters_count`, plus petit = meilleur. reconstruct(compressed: &[f64], shape: &[usize], rebuilt: &mut [f64]) reconstruit le tenseur dans `rebuilt`. Le tenseur d'entree est de RANG FAIBLE (somme de quelques produits exterieurs), donc son depliage matriciel est de rang faible et une SVD tronquee le compresse fortement sans perte. Stocker tous les elements (params = taille totale) est le PIRE score. Garde l'erreur L2 relative sous 1e-3.

DEPENDANCES DISPONIBLES (AUCUNE autre): std, le crate `rand` 0.8, et le crate `nalgebra` 0.33. Utilise `nalgebra::DMatrix` et `.svd(true, true)` (champs `.u`, `.v_t`, `.singular_values`). N'utilise NI ndarray NI ndarray_stats: ils ne sont PAS disponibles, le code ne compilera pas.

EXEMPLE de solution CORRECTE qui compresse (~4x, reconstruction exacte). Inspire-toi en -- tu peux ameliorer le choix du rang ou le depliage -- mais garde EXACTEMENT ces deux signatures publiques:

use nalgebra::DMatrix;

pub fn compress(flat_tensor: &[f64], shape: &[usize]) -> Vec<f64> {
    let n0 = shape[0];
    let ncols: usize = shape[1..].iter().product();
    let m = DMatrix::from_row_slice(n0, ncols, flat_tensor);
    let svd = m.svd(true, true);
    let s = &svd.singular_values;
    let u = svd.u.as_ref().unwrap();
    let vt = svd.v_t.as_ref().unwrap();
    let smax = s.iter().cloned().fold(0.0_f64, f64::max);
    let tol = 1e-9_f64.max(smax * 1e-7);
    let mut r = 0usize;
    for i in 0..s.len() { if s[i] > tol { r += 1; } }
    if r == 0 { r = 1; }
    let mut out = Vec::with_capacity(1 + r * (n0 + ncols + 1));
    out.push(r as f64);
    for j in 0..r { for i in 0..n0 { out.push(u[(i, j)]); } }
    for j in 0..r { out.push(s[j]); }
    for j in 0..r { for c in 0..ncols { out.push(vt[(j, c)]); } }
    out
}

pub fn reconstruct(compressed: &[f64], shape: &[usize], rebuilt: &mut [f64]) {
    let n0 = shape[0];
    let ncols: usize = shape[1..].iter().product();
    let r = compressed[0] as usize;
    let mut idx = 1usize;
    let u = DMatrix::from_column_slice(n0, r, &compressed[idx..idx + n0 * r]); idx += n0 * r;
    let s: Vec<f64> = compressed[idx..idx + r].to_vec(); idx += r;
    let vt = DMatrix::from_row_slice(r, ncols, &compressed[idx..idx + r * ncols]);
    let mut us = u.clone();
    for j in 0..r { for i in 0..n0 { us[(i, j)] *= s[j]; } }
    let m = us * vt;
    for i in 0..n0 { for c in 0..ncols { rebuilt[i * ncols + c] = m[(i, c)]; } }
}"#;

        /// Weak objectif : identity baseline a battre, aucune piste.
        const OBJ_WEAK: &str = r#"OBJECTIF: COMPRESSION. compress(flat_tensor: &[f64], shape: &[usize]) -> Vec<f64> doit renvoyer le MOINS de valeurs possible -- c'est le score `parameters_count`, plus petit = meilleur. reconstruct(compressed: &[f64], shape: &[usize], rebuilt: &mut [f64]) reconstruit le tenseur dans `rebuilt`. Le tenseur d'entree est compressible (structure sous-jacente), mais tu ne connais pas cette structure. Garde l'erreur L2 relative sous 1e-3.

DEPENDANCES DISPONIBLES (AUCUNE autre): std, le crate `rand` 0.8, et le crate `nalgebra` 0.33. Utilise `nalgebra::DMatrix` et `.svd(true, true)`. N'utilise NI ndarray NI ndarray_stats.

Voici l'implementation actuelle (baseline a battre — identite, params = flat.len()) :

pub fn compress(flat_tensor: &[f64], _shape: &[usize]) -> Vec<f64> {
    flat_tensor.to_vec()
}

pub fn reconstruct(compressed: &[f64], _shape: &[usize], rebuilt: &mut [f64]) {
    for (i, &v) in compressed.iter().enumerate() {
        rebuilt[i] = v;
    }
}

Rends compress plus compresseant tout en gardant reconstruct correct. L2 relative < 1e-3."#;

        /// Mid objectif : diagnostic du probleme de compression SANS donner la solution.
        const OBJ_MID: &str = r#"OBJECTIF: COMPRESSION. compress(flat_tensor: &[f64], shape: &[usize]) -> Vec<f64> doit renvoyer le MOINS de valeurs possible -- c'est le score `parameters_count`, plus petit = meilleur. reconstruct(compressed: &[f64], shape: &[usize], rebuilt: &mut [f64]) reconstruit le tenseur dans `rebuilt`. Le tenseur d'entree a une structure sous-jacente compressible. Garde l'erreur L2 relative sous 1e-3.

DEPENDANCES DISPONIBLES (AUCUNE autre): std, le crate `rand` 0.8, et le crate `nalgebra` 0.33. Utilise `nalgebra::DMatrix` et `.svd(true, true)`. N'utilise NI ndarray NI ndarray_stats.

DIAGNOSTIC du probleme : le tenseur d'entree provient de decompositions de rang faible (tenseurs CP / Tensor Train). Quand on deflace ce tenseur multi-mode en une matrice 2D (depliage matriciel), cette matrice conserve un rang faible equivalent. L'algorithme de deplacement des modes importe pour la forme mais pas pour le rang du resultat : deplier le tenseur en matrice et appliquer une SVD tronquee sur les valeurs singulieres revela les composantes redondantes. Stocker tous les elements (params = taille totale) est le PIRE score.

Inspire-toi de cette idee de depliage + SVD tronquee pour concevoir ton compresseur. Garde EXACTEMENT ces deux signatures publiques:

pub fn compress(flat_tensor: &[f64], shape: &[usize]) -> Vec<f64>;
pub fn reconstruct(compressed: &[f64], shape: &[usize], rebuilt: &mut [f64]);"#;

        #[cfg(feature = "bandit")]
        {
            if matches!(std::env::var("FORGE_MAB").as_deref(), Ok("1")) {
                let objectives = vec![
                    OBJ_FULL.to_string(),
                    OBJ_MID.to_string(),
                    OBJ_WEAK.to_string(),
                ];
                let base = crate::mutation::llm_mutator::LlmMutator::new(endpoint, model)
                    .with_timeout(600);
                self.bandit = std::sync::Arc::new(std::sync::Mutex::new(
                    crate::mutation::bandit::MutationBandit::new(base, objectives),
                ));
            } else {
                let objective = match std::env::var("FORGE_FEWSHOT").unwrap_or_default().as_str() {
                    "weak" => OBJ_WEAK,
                    "mid" => OBJ_MID,
                    _ => OBJ_FULL,
                };
                self.llm = Some(
                    crate::mutation::llm_mutator::LlmMutator::new(endpoint, model)
                        .with_timeout(600)
                        .with_objective(objective),
                );
            }
        }

        #[cfg(not(feature = "bandit"))]
        {
            let objective = match std::env::var("FORGE_FEWSHOT").unwrap_or_default().as_str() {
                "weak" => OBJ_WEAK,
                "mid" => OBJ_MID,
                _ => OBJ_FULL,
            };
            self.llm = Some(
                crate::mutation::llm_mutator::LlmMutator::new(endpoint, model)
                    .with_timeout(600)
                    .with_objective(objective),
            );
        }

        self
    }

    /// Retourne le nombre de bras du bandit (si active).
    #[cfg(feature = "bandit")]
    pub fn bandit_n_arms(&self) -> usize {
        if let Ok(b) = self.bandit.lock() {
            b.n_arms()
        } else {
            0
        }
    }

    /// Renvoie le nom du few-shot actuellement selectionne (pour debug).
    #[cfg(feature = "llm")]
    pub fn current_fewshot(&self) -> &'static str {
        match std::env::var("FORGE_FEWSHOT").unwrap_or_default().as_str() {
            "weak" => "weak",
            "mid" => "mid",
            _ => "full",
        }
    }

    fn setup_candidate_env(
        &self,
        cand_id: CandidateId,
        source: &str,
        tensor_size: usize, // dimension par mode (ex: 4 pour 4×4×4×4)
        num_dims: usize,    // nombre de modes (ex: 4)
        seed: u64,
    ) -> crate::error::Result<PathBuf> {
        static ENV_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nonce = ENV_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let cand_dir = self
            .workspace_root
            .join(format!("cand_{}_{}", cand_id, nonce));
        let src_dir = cand_dir.join("src");
        let benches_dir = cand_dir.join("benches");
        fs::create_dir_all(&src_dir).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        fs::create_dir_all(&benches_dir).map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        let total_elements = tensor_size.pow(num_dims as u32);
        let crate_name = format!("cand_{}", cand_id);

        // 1. Cargo.toml
        let toml_path = cand_dir.join("Cargo.toml");
        let mut toml =
            fs::File::create(&toml_path).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        writeln!(
            toml,
            "[package]\n\
             name = \"cand_{}\"\n\
             version = \"0.1.0\"\n\
             edition = \"2021\"\n\n\
             [dependencies]\n\
             rand = \"0.8\"\n\
             nalgebra = \"0.33\"\n\n\
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

        // 3. src/main.rs — harnais : tenseur tiré de la graine + mesure de params + porte L2
        let main_src = VERIFY_MAIN_RS
            .replace("__CRATE__", &crate_name)
            .replace("__N__", &tensor_size.to_string())
            .replace("__DIMS__", &num_dims.to_string())
            .replace("__TOTAL__", &total_elements.to_string())
            .replace("__RANK__", &self.tt_rank.to_string())
            .replace("__SEED__", &seed.to_string())
            .replace("__TOL__", &format!("{:e}", self.tolerance));
        let mut main = fs::File::create(src_dir.join("main.rs"))
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        main.write_all(main_src.as_bytes())
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 4. benches/tt_bench.rs — benchmark Criterion
        let bench_src = BENCH_RS
            .replace("__CRATE__", &crate_name)
            .replace("__N__", &tensor_size.to_string())
            .replace("__DIMS__", &num_dims.to_string())
            .replace("__TOTAL__", &total_elements.to_string());
        let mut bench = fs::File::create(benches_dir.join("tt_bench.rs"))
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        bench
            .write_all(bench_src.as_bytes())
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        Ok(cand_dir)
    }

    fn clean_env(&self, env_path: &std::path::Path) {
        let _ = fs::remove_dir_all(env_path);
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

    fn mutate(
        &self,
        _rng: &mut StdRng,
        parents: &[&Self::Cand],
    ) -> crate::error::Result<Self::Cand> {
        #[cfg(feature = "llm")]
        {
            // Si bandit active (FORGE_MAB=1), delegate vers le bandit.
            #[cfg(feature = "bandit")]
            if matches!(std::env::var("FORGE_MAB").as_deref(), Ok("1")) {
                if let Ok(mut bandit) = self.bandit.lock() {
                    use crate::candidate::Candidate as _;
                    let parent_src = parents.first().map(|p| p.repr()).unwrap_or_default();
                    let (new_src, arm) = bandit.mutate_and_record_arm(&parent_src, None);

                    if std::env::var("FORGE_VERBOSE").is_ok() {
                        eprintln!(
                            "[forge:mutate] MAB arm={} -> {} octets de source (best_arm={})",
                            arm,
                            new_src.len(),
                            bandit.best_arm()
                        );
                    }
                    // Stocker l'arm selectionné pour le runner (delivre le reward apres evaluation).
                    let id = crate::fnv1a(&new_src);
                    if parents.first().map(|p| p.id) != Some(id) {
                        bandit.track_candidate_arm(id, arm);
                    }
                    return Ok(TensorCode {
                        raw_source: new_src,
                        id,
                    });
                }
            }

            // Chemin standard LLM.
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
                    let _ = std::fs::create_dir_all("/tmp/forge_children");
                    let _ = std::fs::write(
                        format!("/tmp/forge_children/child_{}.rs", crate::fnv1a(&new_src)),
                        &new_src,
                    );
                }
                let id = crate::fnv1a(&new_src);
                return Ok(TensorCode {
                    raw_source: new_src,
                    id,
                });
            }
        }
        let _ = parents;
        Err(ForgeError::Evaluation(
            "Mutation LLM indisponible -- compiler avec --features llm".into(),
        ))
    }

    fn observe_evaluation(
        &self,
        _cand: &Self::Cand,
        _score: &Score,
        _parent_score: Option<&Score>,
    ) {
        #[cfg(feature = "bandit")]
        if matches!(std::env::var("FORGE_MAB").as_deref(), Ok("1")) {
            let Some(parent_score) = _parent_score else {
                return;
            };
            let Some(reward) = crate::mutation::bandit::MutationBandit::minimization_reward(
                parent_score,
                _score,
                self.reward_objective_idx,
            ) else {
                return;
            };
            if let Ok(mut bandit) = self.bandit.lock() {
                let delivered = bandit.deliver_reward_for_candidate(_cand.id, reward);
                if delivered && std::env::var("FORGE_VERBOSE").is_ok() {
                    eprintln!(
                        "[forge:bandit] candidate={} reward={:+.4} best_arm={} means={:?}",
                        _cand.id,
                        reward,
                        bandit.best_arm(),
                        bandit.means()
                    );
                }
            }
        }
    }

    /// PORTE DE CORRECTION : lint anti-état-global, puis compile et exécute sur
    /// un tenseur 4×4×4×4 tiré de `trial.seed`, vérifie l'erreur L2 relative.
    fn verify(&self, cand: &Self::Cand, trial: &Trial) -> crate::error::Result<bool> {
        // Lint : un candidat à état global peut contourner la mesure de params.
        if uses_forbidden_capability(&cand.raw_source) {
            return Ok(false);
        }

        let dim_size = 4usize;
        let num_dims = 4usize;

        let env_path =
            self.setup_candidate_env(cand.id, &cand.raw_source, dim_size, num_dims, trial.seed)?;

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
            self.clean_env(&env_path);
            return Ok(false);
        }

        // Étape 2 : Exécution du harnais (exit 101 si l'erreur dépasse la tolérance)
        let mut run_cmd = Command::new("cargo");
        run_cmd.arg("run").arg("--release").current_dir(&env_path);

        let run_res =
            run_with_secure_limits(run_cmd, self.exec_timeout, self.max_mem, self.max_disk);
        self.clean_env(&env_path);

        match run_res {
            Ok(_stdout) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// ÉVALUATION DES OBJECTIFS : [erreur_L2_relative, latence_ns, paramètres_stockés]
    fn measure(&self, cand: &Self::Cand, trial: &Trial) -> crate::error::Result<Vec<f64>> {
        if uses_forbidden_capability(&cand.raw_source) {
            return Err(ForgeError::Evaluation(
                "Candidat rejeté : état global interdit (contournement de la mesure de paramètres)"
                    .into(),
            ));
        }

        let dim_size = 8usize;
        let num_dims = 4usize;

        let env_path =
            self.setup_candidate_env(cand.id, &cand.raw_source, dim_size, num_dims, trial.seed)?;

        // Compilation
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
            self.clean_env(&env_path);
            return Err(ForgeError::Evaluation(
                "Échec de compilation pour la mesure".into(),
            ));
        }

        // ── Objectifs 1 & 3 : erreur L2 relative + paramètres stockés (mesurés au même run) ──
        let mut run_cmd = Command::new("cargo");
        run_cmd.arg("run").arg("--release").current_dir(&env_path);

        let (l2_error, param_count) =
            match run_with_secure_limits(run_cmd, self.exec_timeout, self.max_mem, self.max_disk) {
                Ok(stdout) => {
                    let l2 = Self::extract_l2_error(&stdout).unwrap_or(f64::INFINITY);
                    let params = Self::extract_params(&stdout).unwrap_or(f64::INFINITY);
                    (l2, params)
                }
                Err(_) => {
                    self.clean_env(&env_path);
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

        let latency_ns = match run_with_secure_limits(
            bench_cmd,
            self.bench_timeout,
            self.max_mem,
            self.max_disk,
        ) {
            Ok(_) => match parse_and_validate_metrics(&env_path, "tt_target", 0.04) {
                Ok(objs) => objs[0],
                Err(_) => {
                    self.clean_env(&env_path);
                    return Err(ForgeError::Evaluation(
                        "Mesure Criterion instable ou absente — bruit thermique probable".into(),
                    ));
                }
            },
            Err(_) => {
                self.clean_env(&env_path);
                return Err(ForgeError::Evaluation(
                    "Échec du benchmark Criterion".into(),
                ));
            }
        };

        self.clean_env(&env_path);

        // Les 3 objectifs à minimiser
        if std::env::var("FORGE_VERBOSE").is_ok() {
            eprintln!(
                "[forge:measure] valide  L2={:.3e}  latency_ns={:.0}  params={:.0}",
                l2_error, latency_ns, param_count
            );
        }
        Ok(vec![l2_error, latency_ns, param_count])
    }

    fn objective_names(&self) -> Vec<String> {
        vec![
            "reconstruction_error_L2".into(),
            "latency_ns".into(),
            "parameters_count".into(),
        ]
    }

    fn baseline(&self, trial: &Trial) -> crate::error::Result<Score> {
        // Mesure la vraie baseline identité sur le même trial et la même machine.
        let base = self.seed(&mut StdRng::seed_from_u64(0));
        Ok(Score::valid(self.measure(&base, trial)?))
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
        assert!(uses_forbidden_capability(
            "thread_local! { static X: u32 = 0; }"
        ));
        assert!(uses_forbidden_capability("static mut COUNTER: i64 = 0;"));
        assert!(!uses_forbidden_capability(
            "pub fn compress(f: &[f64], _s: &[usize]) -> Vec<f64> { f.to_vec() }"
        ));
    }
}
