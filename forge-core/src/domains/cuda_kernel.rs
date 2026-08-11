//! Domaine d'optimisation de kernels CUDA natifs pour calcul GPGPU.
//!
//! Ce domaine génère et évalue du code CUDA brut (`CudaCode`) compilé par
//! `nvcc` et exécuté sur GPU. Le moteur cherche à minimiser la latence
//! d'exécution GPU et le nombre d'instructions PTX (proxy de compacité).
//!
//! ## Candidat
//! `__global__ void compute_kernel(double* c, const double* a, const double* b, int n);`
//! — multiplication matricielle `C = A × B` pour matrices carrées `n × n`.
//!
//! ## Pipeline
//! 1. Compilation `nvcc -O3 --ptxas-options=-v -o cuda_verify kernel.cu main.cu`
//! 2. Vérification mathématique : matrices `A`, `B` ALÉATOIRES tirées de la
//!    graine d'essai (`trial.seed`), comparées élément par élément à une
//!    référence CPU calculée dans le harnais (tolérance `1e-6 · N`). Un kernel
//!    faux-mais-rapide (ex. `c[i]=N`) ne peut pas passer.
//! 3. Mesure de latence via CUDA Events (`cudaEventRecord` + `cudaDeviceSynchronize`)
//! 4. Comptage d'instructions PTX depuis le fichier `.ptx` généré

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::candidate::{Candidate, CandidateId};
use crate::domain::{Domain, Score};
use crate::error::{ForgeError, Result};
use crate::isolation::run_with_timeout;
use crate::trial::Trial;

/// Harnais hôte `main.cu`. Le candidat ne fournit que le kernel
/// `compute_kernel` (dans `kernel.cu`) ; ce harnais — qu'il ne contrôle pas —
/// possède les entrées, la référence CPU et la comparaison. `__N__`/`__SEED__`
/// sont remplacés à la génération.
const VERIFY_MAIN_CU: &str = r#"#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstdint>

// Kernel externe defini dans kernel.cu
extern "C" __global__ void compute_kernel(double* c, const double* a, const double* b, int n);

static uint64_t rng_next(uint64_t* s) {
    *s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = *s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double rng_f64(uint64_t* s) {
    return (double)(rng_next(s) >> 11) / (double)(1ULL << 53);
}

int main() {
    const int N = __N__;
    const size_t bytes = (size_t)N * N * sizeof(double);

    double *h_a = (double*)malloc(bytes);
    double *h_b = (double*)malloc(bytes);
    double *h_c = (double*)malloc(bytes);
    double *h_ref = (double*)malloc(bytes);

    // Entrees ALEATOIRES deterministes, derivees de la graine d'essai.
    uint64_t s = (uint64_t)__SEED__ULL ^ 0xD1B54A32D192ED03ULL;
    for (int i = 0; i < N * N; i++) {
        h_a[i] = rng_f64(&s) * 2.0 - 1.0;
        h_b[i] = rng_f64(&s) * 2.0 - 1.0;
        h_c[i] = 0.0;
    }

    // Reference CPU (GEMM naif) — possedee par le harnais, pas par le candidat.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < N; k++) acc += h_a[i * N + k] * h_b[k * N + j];
            h_ref[i * N + j] = acc;
        }
    }

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, N);
    cudaEventRecord(stop);

    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        fprintf(stderr, "CUDA_KERNEL_ERROR: %s\n", cudaGetErrorString(kernelErr));
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c); free(h_ref);
        return 101;
    }

    cudaDeviceSynchronize();

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    double latency_ns = (double)elapsed_ms * 1e6;

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verification mathematique contre la reference CPU.
    double max_diff = 0.0;
    for (int i = 0; i < N * N; i++) {
        double d = fabs(h_c[i] - h_ref[i]);
        if (d > max_diff) max_diff = d;
    }
    // Tolerance adaptee a l'accumulation O(N) en double precision.
    double tol = 1e-6 * (double)N;
    if (max_diff > tol) {
        fprintf(stderr, "ASSERTION_FAILED: max_diff=%.6e tol=%.6e\n", max_diff, tol);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c); free(h_ref);
        return 101;
    }

    printf("CUDA_LATENCY_NS=%.6f\n", latency_ns);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c); free(h_ref);
    return 0;
}
"#;

// ---------------------------------------------------------------------------
// Candidat CUDA
// ---------------------------------------------------------------------------

/// Code source CUDA brut généré par le LLM ou le micro-mutateur.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CudaCode {
    pub source: String,
    pub id: CandidateId,
}

impl Candidate for CudaCode {
    fn id(&self) -> CandidateId {
        self.id
    }
    fn repr(&self) -> String {
        self.source.clone()
    }
}

// ---------------------------------------------------------------------------
// Domaine CUDA
// ---------------------------------------------------------------------------

/// Domaine d'optimisation de kernels CUDA avec compilation `nvcc` et
/// exécution GPU réelle.
pub struct CudaKernelDomain {
    pub workspace_root: PathBuf,
    pub compile_timeout: Duration,
    pub exec_timeout: Duration,
    /// Dimension des matrices de test (N × N).
    pub matrix_size: usize,
    #[cfg(feature = "llm")]
    pub llm: Option<crate::mutation::llm_mutator::LlmMutator>,
    #[cfg(all(feature = "llm", feature = "bandit"))]
    pub bandit: std::sync::Arc<std::sync::Mutex<crate::mutation::bandit::MutationBandit>>,
    pub reward_objective_idx: usize,
}

impl CudaKernelDomain {
    pub fn new(root: &str) -> Self {
        Self {
            workspace_root: PathBuf::from(root),
            compile_timeout: Duration::from_secs(120),
            exec_timeout: Duration::from_secs(30),
            matrix_size: 256,
            #[cfg(feature = "llm")]
            llm: None,
            #[cfg(all(feature = "llm", feature = "bandit"))]
            bandit: std::sync::Arc::new(std::sync::Mutex::new(
                crate::mutation::bandit::MutationBandit::new(
                    crate::mutation::llm_mutator::LlmMutator::new("", ""),
                    vec![String::new()], // placeholder — overwritten by with_llm
                ),
            )),
            reward_objective_idx: 0,
        }
    }

    #[cfg(feature = "llm")]
    pub fn with_llm(mut self, endpoint: &str, model: &str) -> Self {
        const OBJ: &str = r#"OBJECTIF: VITESSE GPU. Le kernel CUDA `extern "C" __global__ void compute_kernel(double* c, const double* a, const double* b, int n)` calcule C = A x B pour matrices carrees n x n en row-major. Score = latency_ns (CUDA Events), plus petit = meilleur ; un second objectif minimise le nombre d'instructions PTX. Correction OBLIGATOIRE : compare element par element a une reference CPU naive sur entrees aleatoires (tolerance 1e-6 * N), un kernel rapide mais FAUX est rejete. Garde EXACTEMENT cette signature et le extern "C".
CONTEXTE D'EXECUTION (non controle) : lancement en blocs de 16x16 threads, grille = ceil(n/16) x ceil(n/16). Le tampon de sortie c n'est PAS initialise -> tu DOIS ecrire chaque c[row*n+col] (ecrasement, jamais +=). Compilation nvcc -O3 -arch=native.
EXEMPLE correct et PLUS RAPIDE que le naif : GEMM TUILE en memoire partagee (tuile 16x16 = taille de bloc), acces coalesces et reutilisation en shared memory. Inspire-toi en (tu peux augmenter le travail par thread, derouler, etc.) mais garde EXACTEMENT la signature :
extern "C" __global__ void compute_kernel(double* c, const double* a, const double* b, int n) {
    const int TILE = 16;
    __shared__ double As[16][16];
    __shared__ double Bs[16][16];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    double acc = 0.0;
    int nt = (n + TILE - 1) / TILE;
    for (int t = 0; t < nt; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? a[row * n + aCol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? b[bRow * n + col] : 0.0;
        __syncthreads();
        for (int k = 0; k < TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        c[row * n + col] = acc;
    }
}"#;
        const OBJ_WEAK: &str = r#"OBJECTIF: VITESSE GPU. `extern "C" __global__ void compute_kernel(double* c, const double* a, const double* b, int n)` calcule C = A x B (matrices carrees n x n, row-major). Score = latency_ns (CUDA Events), plus petit = meilleur ; second objectif = instructions PTX. Correction OBLIGATOIRE : compare a une reference CPU naive sur entrees aleatoires (tolerance 1e-6 * N), un kernel FAUX est rejete. Garde EXACTEMENT la signature et le extern "C".
CONTEXTE (non controle) : blocs 16x16 threads, grille ceil(n/16) au carre. c n'est PAS initialise -> ecris chaque c[row*n+col] (ecrasement). Compilation nvcc -O3 -arch=native.
Implementation NAIVE actuelle (baseline a battre). Rends-la plus rapide en exploitant la memoire partagee, la coalescence et le parallelisme GPU, sans changer le resultat ni la signature :
extern "C" __global__ void compute_kernel(double* c, const double* a, const double* b, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double acc = 0.0;
        for (int k = 0; k < n; k++) {
            acc += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = acc;
    }
}"#;
        let objectives = vec![OBJ.to_string(), OBJ_WEAK.to_string()];

        #[cfg(feature = "bandit")]
        {
            if matches!(std::env::var("FORGE_MAB").as_deref(), Ok("1")) {
                let objectives = vec![OBJ.to_string(), OBJ_WEAK.to_string()];
                let base = crate::mutation::llm_mutator::LlmMutator::new(endpoint, model)
                    .with_timeout(600);
                self.bandit = std::sync::Arc::new(std::sync::Mutex::new(
                    crate::mutation::bandit::MutationBandit::new(base, objectives),
                ));
            } else {
                let objective = match std::env::var("FORGE_FEWSHOT").unwrap_or_default().as_str() {
                    "weak" => OBJ_WEAK,
                    _ => OBJ,
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
                _ => OBJ,
            };
            self.llm = Some(
                crate::mutation::llm_mutator::LlmMutator::new(endpoint, model)
                    .with_timeout(600)
                    .with_objective(objective),
            );
        }

        self
    }

    /// Prépare l'environnement de build CUDA isolé :
    /// - `kernel.cu` : code candidat
    /// - `main.cu` : harnais hôte C++ avec entrées aléatoires, référence CPU
    ///   et CUDA Events. `seed` rend les entrées déterministes par génération.
    fn write_environment(
        &self,
        cand_id: CandidateId,
        raw_code: &str,
        size: usize,
        seed: u64,
    ) -> Result<PathBuf> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ENV_NONCE: AtomicU64 = AtomicU64::new(0);
        let nonce = ENV_NONCE.fetch_add(1, Ordering::Relaxed);
        let env_path = self
            .workspace_root
            .join(format!("cuda_kernel_{}_{}", cand_id, nonce));
        fs::create_dir_all(&env_path).map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 1. kernel.cu — code candidat
        let kernel_path = env_path.join("kernel.cu");
        let mut kernel_file =
            File::create(&kernel_path).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        write!(kernel_file, "{}", raw_code).map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        // 2. main.cu — harnais hôte : entrées aléatoires + référence CPU + comparaison.
        let main_src = VERIFY_MAIN_CU
            .replace("__N__", &size.to_string())
            .replace("__SEED__", &seed.to_string());
        let main_path = env_path.join("main.cu");
        let mut main_file =
            File::create(&main_path).map_err(|e| ForgeError::Evaluation(e.to_string()))?;
        main_file
            .write_all(main_src.as_bytes())
            .map_err(|e| ForgeError::Evaluation(e.to_string()))?;

        Ok(env_path)
    }

    /// Extrait la latence GPU depuis le stdout du binaire.
    fn extract_latency(stdout: &str) -> Option<f64> {
        for line in stdout.lines() {
            if let Some(val) = line.strip_prefix("CUDA_LATENCY_NS=") {
                return val.trim().parse::<f64>().ok();
            }
        }
        None
    }

    /// Extrait le nombre d'instructions PTX depuis le fichier `.ptx` généré.
    fn extract_ptx_count(env_path: &Path) -> f64 {
        let ptx_path = env_path.join("kernel.ptx");
        if !ptx_path.exists() {
            return 0.0; // métrique absente = invalide
        }
        match fs::read_to_string(&ptx_path) {
            Ok(content) => content
                .lines()
                .filter(|line| {
                    let trimmed = line.trim();
                    !trimmed.is_empty()
                        && !trimmed.starts_with("//")
                        && !trimmed.starts_with('.')
                        && !trimmed.starts_with('{')
                        && !trimmed.starts_with('}')
                        && trimmed.contains(';')
                })
                .count() as f64,
            Err(_) => 0.0,
        }
    }
}

impl Domain for CudaKernelDomain {
    type Cand = CudaCode;

    fn name(&self) -> &str {
        "cuda_gemm"
    }

    fn seed(&self, _rng: &mut StdRng) -> Self::Cand {
        let baseline = r#"// Kernel GEMM naif de reference (baseline a battre)
extern "C" __global__ void compute_kernel(double* c, const double* a, const double* b, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        double acc = 0.0;
        for (int k = 0; k < n; k++) {
            acc += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = acc;
    }
}
"#;
        CudaCode {
            source: baseline.to_string(),
            id: crate::fnv1a(baseline),
        }
    }

    fn mutate(&self, _rng: &mut StdRng, parents: &[&Self::Cand]) -> Result<Self::Cand> {
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
                    let id = crate::fnv1a(&new_src);
                    if parents.first().map(|p| p.id) != Some(id) {
                        bandit.track_candidate_arm(id, arm);
                    }
                    return Ok(CudaCode {
                        source: new_src,
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
                }
                let id = crate::fnv1a(&new_src);
                return Ok(CudaCode {
                    source: new_src,
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

    /// Porte de correction : compilation nvcc + exécution + validation
    /// mathématique sur entrées aléatoires (`trial.seed`) contre la référence CPU.
    fn verify(&self, cand: &Self::Cand, trial: &Trial) -> Result<bool> {
        let size = 32; // Petit pour vérification rapide
        let env_path = self.write_environment(cand.id, &cand.source, size, trial.seed)?;

        // Étape 1 : Compilation nvcc
        let output_bin = env_path.join("cuda_verify");
        let mut comp_cmd = Command::new("nvcc");
        comp_cmd
            .arg("-O3")
            .arg("-arch=native")
            .arg("--ptxas-options=-v")
            .arg("-o")
            .arg(&output_bin)
            .arg(env_path.join("kernel.cu"))
            .arg(env_path.join("main.cu"))
            .current_dir(&env_path);

        if run_with_timeout(comp_cmd, self.compile_timeout).is_err() {
            let _ = fs::remove_dir_all(&env_path);
            return Ok(false);
        }

        // Étape 2 : Exécution du binaire GPU (exit 101 si la sortie diverge de la référence)
        let mut run_cmd = Command::new(&output_bin);
        run_cmd.current_dir(&env_path);

        let run_res = run_with_timeout(run_cmd, self.exec_timeout);
        let _ = fs::remove_dir_all(&env_path);

        match run_res {
            Ok(_stdout) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Mesure de performance GPU : [latence_ns, ptx_instruction_count]
    fn measure(&self, cand: &Self::Cand, trial: &Trial) -> Result<Vec<f64>> {
        let size = self.matrix_size;
        let env_path = self.write_environment(cand.id, &cand.source, size, trial.seed)?;

        // Compilation nvcc avec flags d'optimisation
        let output_bin = env_path.join("cuda_verify");
        let mut comp_cmd = Command::new("nvcc");
        comp_cmd
            .arg("-O3")
            .arg("-arch=native")
            .arg("--ptxas-options=-v")
            .arg("-o")
            .arg(&output_bin)
            .arg(env_path.join("kernel.cu"))
            .arg(env_path.join("main.cu"))
            .current_dir(&env_path);

        match run_with_timeout(comp_cmd, self.compile_timeout) {
            Ok(_) => {}
            Err(e) => {
                let _ = fs::remove_dir_all(&env_path);
                return Err(ForgeError::Evaluation(format!(
                    "Échec compilation nvcc: {e}"
                )));
            }
        }

        // Génère explicitement le PTX mesuré par le second objectif.
        let ptx_path = env_path.join("kernel.ptx");
        let mut ptx_cmd = Command::new("nvcc");
        ptx_cmd
            .arg("-O3")
            .arg("-arch=native")
            .arg("--ptx")
            .arg("-o")
            .arg(&ptx_path)
            .arg(env_path.join("kernel.cu"))
            .current_dir(&env_path);
        if let Err(e) = run_with_timeout(ptx_cmd, self.compile_timeout) {
            let _ = fs::remove_dir_all(&env_path);
            return Err(ForgeError::Evaluation(format!("Échec génération PTX: {e}")));
        }

        // Exécution du binaire GPU
        let mut run_cmd = Command::new(&output_bin);
        run_cmd.current_dir(&env_path);

        let latency_ns = match run_with_timeout(run_cmd, self.exec_timeout) {
            Ok(stdout) => Self::extract_latency(&stdout).unwrap_or(1_000_000.0),
            Err(e) => {
                let _ = fs::remove_dir_all(&env_path);
                return Err(ForgeError::Evaluation(format!("Échec exécution GPU: {e}")));
            }
        };

        // Comptage des instructions PTX depuis le fichier .ptx généré
        let ptx_count = Self::extract_ptx_count(&env_path);
        if ptx_count <= 0.0 {
            let _ = fs::remove_dir_all(&env_path);
            return Err(ForgeError::Evaluation(
                "PTX absent ou vide après génération explicite".into(),
            ));
        }

        let _ = fs::remove_dir_all(&env_path);

        Ok(vec![latency_ns, ptx_count])
    }

    fn objective_names(&self) -> Vec<String> {
        vec!["latency_ns".into(), "ptx_instruction_count".into()]
    }

    fn baseline(&self, trial: &Trial) -> Result<Score> {
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

    #[test]
    fn test_seed_is_valid_cuda() {
        let domain = CudaKernelDomain::new("/tmp/forge_cuda_v1");
        let mut rng = StdRng::seed_from_u64(42);
        let cand = domain.seed(&mut rng);
        assert!(cand.source.contains("compute_kernel"));
        assert!(cand.source.contains("__global__"));
        assert!(cand.source.contains("blockIdx"));
    }

    #[test]
    fn test_domain_name() {
        let domain = CudaKernelDomain::new("/tmp/irrelevant_cuda");
        assert_eq!(domain.name(), "cuda_gemm");
    }

    #[test]
    fn test_extract_latency_from_stdout() {
        let stdout = "Some output\nCUDA_LATENCY_NS=12345.678900\nMore output\n";
        let lat = CudaKernelDomain::extract_latency(stdout);
        assert!(lat.is_some());
        assert!((lat.unwrap() - 12345.6789).abs() < 0.001);
    }

    #[test]
    fn test_extract_latency_missing() {
        assert!(CudaKernelDomain::extract_latency("no latency here").is_none());
    }

    #[test]
    fn test_objective_names() {
        let domain = CudaKernelDomain::new("/tmp/irrelevant");
        let names = domain.objective_names();
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "latency_ns");
        assert_eq!(names[1], "ptx_instruction_count");
    }
}
