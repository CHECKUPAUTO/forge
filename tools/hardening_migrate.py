from pathlib import Path


def replace(path: str, old: str, new: str, count: int | None = None) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"expected pattern missing in {path}: {old[:120]!r}")
    text = text.replace(old, new) if count is None else text.replace(old, new, count)
    p.write_text(text)


# Framed distributed mock fixes.
replace(
    "forge-core/tests/distributed_infra_test.rs",
    "let payload: EvaluationPayload = match Ok(read_frame(&mut stream)) {\n                            Ok(p) => p,\n                            Err(_) => return,\n                        };",
    "let payload: EvaluationPayload = read_frame(&mut stream);",
)
replace(
    "forge-core/tests/distributed_infra_test.rs",
    "let payload: EvaluationPayload = Ok(read_frame(&mut stream)).unwrap();",
    "let payload: EvaluationPayload = read_frame(&mut stream);",
)

# Registry warning already migrated.

# Context-aware cache, deterministic post-reproduction checkpoint, true Pareto front.
p = Path("forge-core/src/evolve.rs")
text = p.read_text()
if "cache.get(cand.id())" not in text:
    raise SystemExit("local cache lookup pattern missing")
text = text.replace(
    "cache.get(cand.id())",
    "cache.get_scoped(self.domain.name(), cand.id(), trial.seed)",
    1,
)
if "cache.insert(cand.id(), score.objectives.clone());" not in text:
    raise SystemExit("local cache insert pattern missing")
text = text.replace(
    "cache.insert(cand.id(), score.objectives.clone());",
    "cache.insert_scoped(self.domain.name(), cand.id(), trial.seed, score.objectives.clone());",
    1,
)

dynamic_start = text.index("fn evaluate_distributed_dynamic")
dynamic_end = text.index("// Évaluation parallèle distribuée (Round-Robin legacy)", dynamic_start)
dynamic = text[dynamic_start:dynamic_end]
for old, new in [
    ("c.get(cand.id())", "c.get_scoped(domain.name(), cand.id(), trial.seed)"),
    (
        "c.insert(cand.id(), eval_res.objectives.clone());",
        "c.insert_scoped(domain.name(), cand.id(), trial.seed, eval_res.objectives.clone());",
    ),
    (
        "c.insert(cand.id(), score.objectives.clone());",
        "c.insert_scoped(domain.name(), cand.id(), trial.seed, score.objectives.clone());",
    ),
]:
    if old not in dynamic:
        raise SystemExit(f"dynamic cache pattern missing: {old}")
    dynamic = dynamic.replace(old, new)
text = text[:dynamic_start] + dynamic + text[dynamic_end:]

old = '''            // ── Checkpoint atomique Sled ──
            if let Some(ref reg) = self.registry {
                let state = EngineState {
                    current_generation: g + 1,
                    master_seed: self.config.base_seed,
                    population_sources: pop.iter().map(|c| c.repr()).collect(),
                    archive: archive.clone(),
                    history: history.clone(),
                    failure_diagnostics: all_failure_diagnostics_mutex
                        .lock()
                        .map(|fd| fd.clone())
                        .unwrap_or_default(),
                };
                let _ = state.commit_to_sled(reg);
            }

            // Checkpoint JSON legacy
            let cp = Checkpoint {
                generation: g,
                config: self.config.clone(),
                archive: archive.clone(),
            };
            let _ = cp.save_atomic(Path::new("forge_checkpoint.json"));

            if let Some(ref cache) = self.cache {
                if g % 10 == 0 {
                    let _ = cache.persist();
                }
            }

            // Reproduction
            let survivors: Vec<D::Cand> = archive.iter().map(|i| i.cand.clone()).collect();
            if survivors.is_empty() {
                pop = (0..self.config.population)
                    .map(|_| self.domain.seed(&mut master))
                    .collect();
                continue;
            }

            let mut next: Vec<D::Cand> = survivors.clone();
            while next.len() < self.config.population {
                let parent = &survivors[master.gen_range(0..survivors.len())];
                next.push(self.domain.mutate(&mut master, &[parent])?);
            }
            pop = next;
'''
new = '''            // Reproduction déterministe par génération. Le checkpoint est écrit
            // après cette étape afin de contenir exactement la population de g+1.
            let survivors: Vec<D::Cand> = archive.iter().map(|i| i.cand.clone()).collect();
            let mut reproduction_rng =
                StdRng::seed_from_u64(trial.seed ^ 0xA5A5_5A5A_D3C4_B2E1);
            if survivors.is_empty() {
                pop = (0..self.config.population)
                    .map(|_| self.domain.seed(&mut reproduction_rng))
                    .collect();
            } else {
                let mut next: Vec<D::Cand> = survivors.clone();
                while next.len() < self.config.population {
                    let parent = &survivors[reproduction_rng.gen_range(0..survivors.len())];
                    next.push(self.domain.mutate(&mut reproduction_rng, &[parent])?);
                }
                pop = next;
            }

            // ── Checkpoint atomique Sled ──
            if let Some(ref reg) = self.registry {
                let state = EngineState {
                    current_generation: g + 1,
                    master_seed: self.config.base_seed,
                    population_sources: pop.iter().map(|c| c.repr()).collect(),
                    archive: archive.clone(),
                    history: history.clone(),
                    failure_diagnostics: all_failure_diagnostics_mutex
                        .lock()
                        .map(|fd| fd.clone())
                        .unwrap_or_default(),
                };
                state.commit_to_sled(reg)?;
            }

            // Checkpoint JSON legacy : non critique, mais jamais silencieux.
            let cp = Checkpoint {
                generation: g,
                config: self.config.clone(),
                archive: archive.clone(),
            };
            if let Err(e) = cp.save_atomic(Path::new("forge_checkpoint.json")) {
                eprintln!("[forge:checkpoint] échec checkpoint JSON: {e}");
            }

            if let Some(ref cache) = self.cache {
                if g % 10 == 0 {
                    if let Err(e) = cache.persist() {
                        eprintln!("[forge:cache] échec persistance cache: {e}");
                    }
                }
            }
'''
if old not in text:
    raise SystemExit("checkpoint/reproduction block changed unexpectedly")
text = text.replace(old, new, 1)
old = "        let final_front = archive.clone();\n        let best = archive.into_iter().next();"
new = '''        let best = archive.first().cloned();
        let final_front: Vec<Individual<D::Cand>> = archive
            .iter()
            .filter(|candidate| {
                !archive.iter().any(|other| {
                    other.cand.id() != candidate.cand.id()
                        && other.score.dominates(&candidate.score)
                })
            })
            .cloned()
            .collect();'''
if old not in text:
    raise SystemExit("final_front block missing")
text = text.replace(old, new, 1)
p.write_text(text)

# Low-rank side-channel gate + real baseline measurement.
p = Path("forge-core/src/domains/low_rank.rs")
text = p.read_text()
if "use rand::rngs::StdRng;" not in text:
    raise SystemExit("low-rank rand import missing")
text = text.replace("use rand::rngs::StdRng;", "use rand::{rngs::StdRng, SeedableRng};", 1)
old = '''const BANNED_GLOBAL_STATE: &[&str] = &[
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
    BANNED_GLOBAL_STATE
        .iter()
        .any(|needle| source.contains(needle))
}
'''
new = '''const BANNED_CAPABILITIES: &[&str] = &[
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
    "extern \\\"C\\\"",
];

/// Défense en profondeur contre les canaux cachés évidents. Ceci n'est PAS
/// une frontière de sécurité : les candidats non fiables doivent toujours être
/// exécutés dans un sandbox OS externe au worker Forge.
fn uses_forbidden_capability(source: &str) -> bool {
    BANNED_CAPABILITIES
        .iter()
        .any(|needle| source.contains(needle))
}
'''
if old not in text:
    raise SystemExit("low_rank banned capability block missing")
text = text.replace(old, new, 1)
text = text.replace("uses_global_state", "uses_forbidden_capability")
old = '''    fn baseline(&self, _trial: &Trial) -> crate::error::Result<Score> {
        // Baseline : identité (stocke tout → params = 8^4 = 4096, erreur ~0).
        Ok(Score::valid(vec![0.0, 5000.0, 4096.0]))
    }
'''
new = '''    fn baseline(&self, trial: &Trial) -> crate::error::Result<Score> {
        // Mesure la vraie baseline identité sur le même trial et la même machine.
        let base = self.seed(&mut StdRng::seed_from_u64(0));
        Ok(Score::valid(self.measure(&base, trial)?))
    }
'''
if old not in text:
    raise SystemExit("low_rank baseline block missing")
text = text.replace(old, new, 1)
p.write_text(text)

# CUDA explicit PTX generation + no fabricated baseline.
p = Path("forge-core/src/domains/cuda_kernel.rs")
text = p.read_text()
text = text.replace(
    "return 256.0; // valeur par défaut conservative",
    "return 0.0; // métrique absente = invalide",
)
text = text.replace("Err(_) => 256.0,", "Err(_) => 0.0,")
marker = "        // Exécution du binaire GPU\n"
ptx = '''        // Génère explicitement le PTX mesuré par le second objectif.
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

'''
if marker not in text:
    raise SystemExit("CUDA execution marker missing")
text = text.replace(marker, ptx + marker, 1)
old = "        let ptx_count = Self::extract_ptx_count(&env_path);\n\n        let _ = fs::remove_dir_all(&env_path);"
new = '''        let ptx_count = Self::extract_ptx_count(&env_path);
        if ptx_count <= 0.0 {
            let _ = fs::remove_dir_all(&env_path);
            return Err(ForgeError::Evaluation(
                "PTX absent ou vide après génération explicite".into(),
            ));
        }

        let _ = fs::remove_dir_all(&env_path);'''
if old not in text:
    raise SystemExit("CUDA ptx_count block missing")
text = text.replace(old, new, 1)
old = '''    fn baseline(&self, trial: &Trial) -> Result<Score> {
        let base = self.seed(&mut StdRng::seed_from_u64(0));
        match self.measure(&base, trial) {
            Ok(objs) => Ok(Score::valid(objs)),
            Err(_) => Ok(Score::valid(vec![1_000_000.0, 256.0])),
        }
    }
'''
new = '''    fn baseline(&self, trial: &Trial) -> Result<Score> {
        let base = self.seed(&mut StdRng::seed_from_u64(0));
        Ok(Score::valid(self.measure(&base, trial)?))
    }
'''
if old not in text:
    raise SystemExit("CUDA baseline block missing")
text = text.replace(old, new, 1)
p.write_text(text)

# UCB1 formula exactly matches the documented sqrt(2 ln(t) / n) policy.
replace(
    "forge-core/src/mutation/bandit.rs",
    "let bonus = self.exploration * (2.0 * t as f64).ln() / self.pulls[k] as f64;\n            let ucb = mean + bonus.sqrt();",
    "let bonus = ((t as f64).ln() / self.pulls[k] as f64).sqrt();\n            let ucb = mean + self.exploration * bonus;",
)
