//! Boucle évolutionnaire générique industrielle. Le moteur orchestre de manière
//! parallélisée les étapes d'évaluation (porte de correction puis mesure) tout en
//! protégeant le framework contre le sur-apprentissage grâce à des graines rotatives.
//! L'archive d'élites utilise un tri de non-domination de Pareto multi-objectif.
//!
//! ## Mode distribué
//! Quand `Config::worker_addresses` est `Some(addrs)`, l'évaluation est déléguée
//! à des Workers distants via un pool dynamique avec leasing pull-based et
//! failover automatique pour éliminer l'effet straggler.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cache::EvaluationCache;
use crate::candidate::Candidate;
use crate::candidate::CandidateId;
use crate::diagnostics::{FailureDiagnostics, FailureStage};
use crate::domain::{Domain, Score};
use crate::error::{ForgeError, Result};
use crate::protocol::{dispatch_evaluation_to_worker, EvaluationPayload};
use crate::registry::AlgorithmRegistry;
use crate::trial::Trial;

// ---------------------------------------------------------------------------
// Configuration & Structures
// ---------------------------------------------------------------------------

/// Paramètres de la recherche.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    /// Nombre de générations.
    pub generations: u64,
    /// Taille de la population à chaque génération.
    pub population: usize,
    /// Nombre d'élites conservés (et reproducteurs) entre générations.
    pub survivors: usize,
    /// Graine maîtresse (rend toute la campagne reproductible).
    pub base_seed: u64,
    /// Topologie réseau optionnelle : liste des adresses Worker
    /// (ex: `["192.168.1.10:9000", "192.168.1.11:9000"]`).
    /// Quand `Some`, l'évaluation est distribuée ; quand `None`, locale.
    pub worker_addresses: Option<Vec<String>>,
}

/// Un individu évalué.
#[derive(Clone, Serialize, Deserialize)]
pub struct Individual<C: Candidate> {
    pub cand: C,
    pub score: Score,
}

/// Structure de persistance pour sauvegarder/charger l'état de l'archive.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint<C: Candidate> {
    pub generation: u64,
    pub config: Config,
    pub archive: Vec<Individual<C>>,
}

impl<C: Candidate + Serialize + for<'a> Deserialize<'a>> Checkpoint<C> {
    /// Sauvegarde de l'archive de manière atomique (écriture temporaire puis renommage POSIX).
    pub fn save_atomic(&self, path: &Path) -> std::io::Result<()> {
        let tmp_path = path.with_extension("tmp");
        let mut file = File::create(&tmp_path)?;
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;

        file.write_all(json.as_bytes())?;
        file.sync_all()?; // Force le flush sur le support physique (NVMe)
        fs::rename(tmp_path, path)?;
        Ok(())
    }

    /// Charge un checkpoint sauvegardé.
    #[allow(dead_code)]
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let data = fs::read_to_string(path)?;
        serde_json::from_str(&data).map_err(std::io::Error::other)
    }
}

/// État complet du moteur pour checkpointing et reprise sur crash.
/// Sérialisable, commité dans Sled à chaque génération.
#[derive(Serialize, Deserialize, Clone)]
pub struct EngineState<C: Candidate> {
    /// Index de la génération courante (la prochaine à exécuter).
    pub current_generation: u64,
    /// Graine maîtresse actuelle (reproductibilité).
    pub master_seed: u64,
    /// Population de la génération en cours ou à venir.
    pub population_sources: Vec<String>,
    /// Archive d'élites accumulée.
    pub archive: Vec<Individual<C>>,
    /// Historique du meilleur objectif principal par génération.
    pub history: Vec<f64>,
    /// Diagnostics d'échec cumulés.
    pub failure_diagnostics: Vec<FailureDiagnostics>,
}

impl<C: Candidate + Serialize + for<'a> Deserialize<'a>> EngineState<C> {
    /// Commit atomique de l'état dans un bucket Sled dédié.
    pub fn commit_to_sled(&self, registry: &AlgorithmRegistry) -> Result<()> {
        let key = b"__engine_checkpoint__";
        let payload = serde_json::to_vec(self)
            .map_err(|e| ForgeError::Evaluation(format!("Sérialisation EngineState: {e}")))?;
        registry.commit_raw(key, &payload)
    }

    /// Charge l'état depuis le bucket Sled.
    pub fn load_from_sled(registry: &AlgorithmRegistry) -> Result<Option<Self>> {
        registry.get_raw(b"__engine_checkpoint__").and_then(|opt| {
            opt.map(|bytes| {
                serde_json::from_slice(&bytes).map_err(|e| {
                    ForgeError::Evaluation(format!("Désérialisation EngineState: {e}"))
                })
            })
            .transpose()
        })
    }
}

/// Compte-rendu de campagne.
pub struct Report<C: Candidate> {
    /// Meilleur candidat archivé (objectif principal minimal).
    pub best: Option<Individual<C>>,
    /// Baseline mesurée au dernier essai d'evolution.
    pub final_baseline: Option<Score>,
    /// Score du meilleur candidat sur le holdout (graine inédite).
    pub holdout_best: Option<Score>,
    /// Baseline sur le holdout, pour comparaison équitable.
    pub holdout_baseline: Option<Score>,
    /// Meilleur objectif principal par génération (courbe d'apprentissage).
    pub history: Vec<f64>,
    /// Diagnostics d'échec collectés durant la campagne (mode distribué).
    pub failure_diagnostics: Vec<FailureDiagnostics>,
    /// Front de Pareto final (archive non dominée tronquée aux survivants).
    pub final_front: Vec<Individual<C>>,
    /// Score de chaque membre du front re-mesuré sur le holdout (même ordre que final_front).
    pub final_front_holdout: Vec<Option<Score>>,
}

/// Le moteur, paramétré par un domaine.
pub struct Engine<D: Domain> {
    domain: D,
    config: Config,
    /// Cache optionnel pour court-circuiter les évaluations redondantes.
    cache: Option<EvaluationCache>,
    /// Registre de persistance Sled pour le lignage.
    registry: Option<AlgorithmRegistry>,
    /// Génération de départ (0 par défaut, modifiable pour reprise).
    start_generation: u64,
    /// Seed de départ (base_seed par défaut, avancée pour reprise).
    start_seed: u64,
    /// Population initiale pré-remplie (pour reprise sur crash).
    initial_population: Option<Vec<D::Cand>>,
    /// Archive pré-existante (pour reprise sur crash).
    initial_archive: Option<Vec<Individual<D::Cand>>>,
    /// Historique pré-existant.
    initial_history: Vec<f64>,
}

impl<D: Domain> Engine<D>
where
    D::Cand: Serialize + for<'a> Deserialize<'a>,
{
    pub fn new(domain: D, config: Config) -> Self {
        let base_seed = config.base_seed;
        Engine {
            domain,
            config,
            cache: None,
            registry: None,
            start_generation: 0,
            start_seed: base_seed,
            initial_population: None,
            initial_archive: None,
            initial_history: Vec::new(),
        }
    }

    /// Active le cache d'evaluation persistant avec le chemin donne.
    pub fn with_cache(mut self, cache_path: &str) -> Self {
        self.cache = Some(EvaluationCache::new(cache_path));
        self
    }

    /// Active le registre Sled pour la persistance transactionnelle.
    pub fn with_registry(mut self, registry: AlgorithmRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Référence au registre (pour les tests).
    pub fn registry(&self) -> Option<&AlgorithmRegistry> {
        self.registry.as_ref()
    }

    /// Reprend l'exécution à partir d'un EngineState sauvegardé.
    pub fn resume_from_state(
        domain: D,
        config: Config,
        state: EngineState<D::Cand>,
        registry: AlgorithmRegistry,
    ) -> Self
    where
        D::Cand: DeserializeFromSource,
    {
        // Reconstruction de la population a partir des sources sauvegardees.
        let mut pop: Vec<D::Cand> = state
            .population_sources
            .iter()
            .filter_map(|src| D::Cand::deserialize_from_source(src))
            .collect();

        let mut rng = StdRng::seed_from_u64(state.master_seed);

        // Completer avec des seeds si la population sauvegardee est incomplete
        while pop.len() < config.population {
            pop.push(domain.seed(&mut rng));
        }

        Engine {
            domain,
            config,
            cache: None,
            registry: Some(registry),
            start_generation: state.current_generation,
            start_seed: state.master_seed,
            initial_population: Some(pop),
            initial_archive: Some(state.archive),
            initial_history: state.history,
        }
    }

    /// Lance la campagne complète avec parallélisation native et tri de Pareto.
    /// Si `worker_addresses` est configuré, bascule en mode distribué.
    /// Supporte la reprise sur crash via un checkpoint Sled.
    pub fn run(&self) -> Result<Report<D::Cand>> {
        let mut master = if self.start_generation > 0 {
            StdRng::seed_from_u64(self.start_seed)
        } else {
            StdRng::seed_from_u64(self.config.base_seed)
        };

        // Population initiale (seed ou reprise)
        let mut pop: Vec<D::Cand> = if let Some(ref initial) = self.initial_population {
            initial.clone()
        } else {
            (0..self.config.population)
                .map(|_| self.domain.seed(&mut master))
                .collect()
        };

        let mut archive: Vec<Individual<D::Cand>> =
            self.initial_archive.clone().unwrap_or_default();
        let mut history: Vec<f64> = self.initial_history.clone();
        let mut final_baseline: Option<Score> = None;
        let all_failure_diagnostics_mutex = Mutex::new(Vec::<FailureDiagnostics>::new());

        // Pool de workers dynamique pour le mode distribué
        let worker_pool: Option<Arc<WorkerPool>> = self
            .config
            .worker_addresses
            .as_ref()
            .filter(|addrs| !addrs.is_empty())
            .map(|addrs| Arc::new(WorkerPool::new(addrs.clone())));

        let start_gen = self.start_generation;
        for g in start_gen..self.config.generations {
            let trial = Trial {
                generation: g,
                seed: self
                    .config
                    .base_seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(g.wrapping_add(1)),
            };

            // ── BRANCHEMENT DISTRIBUÉ DYNAMIQUE VS LOCAL ──────────────
            let evaluated: Vec<Individual<D::Cand>> = if let Some(ref pool) = worker_pool {
                evaluate_distributed_dynamic(
                    &self.domain,
                    &pop,
                    pool,
                    &trial,
                    self.registry.as_ref(),
                    self.cache.as_ref(),
                    g,
                    &all_failure_diagnostics_mutex,
                )
            } else {
                // Mode local -- un echec d'evaluation n'invalide QUE le candidat
                // concerne, pas toute la generation (un seul candidat non-compilable
                // ne doit pas, via le court-circuit du collect, invalider tous les autres).
                let eval_threads = std::env::var("FORGE_EVAL_CONCURRENCY")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&n| n >= 1)
                    .unwrap_or(2);
                let eval_closure = || {
                    pop.par_iter()
                        .map(|cand| {
                            // Hit cache ?
                            if let Some(ref cache) = self.cache {
                                if let Some(objs) = cache.get_scoped(self.domain.name(), cand.id(), trial.seed) {
                                    return Individual {
                                        cand: cand.clone(),
                                        score: Score::valid(objs),
                                    };
                                }
                            }

                            match evaluate(&self.domain, cand, &trial) {
                                Ok(score) => {
                                    // Insertion cache
                                    if score.valid {
                                        if let Some(ref cache) = self.cache {
                                            cache.insert_scoped(self.domain.name(), cand.id(), trial.seed, score.objectives.clone());
                                        }
                                    }
                                    Individual { cand: cand.clone(), score }
                                }
                                Err(e) => {
                                    if std::env::var("FORGE_VERBOSE").is_ok() {
                                        eprintln!("[forge:eval] echec d'evaluation ({e}) -> candidat invalide");
                                    }
                                    Individual { cand: cand.clone(), score: Score::invalid() }
                                }
                            }
                        })
                        .collect::<Vec<_>>()
                };
                match rayon::ThreadPoolBuilder::new()
                    .num_threads(eval_threads)
                    .build()
                {
                    Ok(eval_pool) => eval_pool.install(eval_closure),
                    Err(_) => eval_closure(),
                }
            };

            let mut valids: Vec<Individual<D::Cand>> = evaluated
                .into_iter()
                .filter(|ind| ind.score.valid)
                .collect();

            final_baseline = Some(self.domain.baseline(&trial)?);

            valids.sort_by(|a, b| cmp_primary(&a.score, &b.score));
            history.push(
                valids
                    .first()
                    .map(|i| i.score.objectives[0])
                    .unwrap_or(f64::INFINITY),
            );

            archive.append(&mut valids);
            let mut seen: HashSet<u64> = HashSet::new();
            archive.retain(|i| seen.insert(i.cand.id()));

            sort_by_pareto_domination(&mut archive);
            archive.truncate(self.config.survivors.max(1));

            // Reproduction déterministe par génération. Le checkpoint est écrit
            // après cette étape afin de contenir exactement la population de g+1.
            let survivors: Vec<D::Cand> = archive.iter().map(|i| i.cand.clone()).collect();
            let mut reproduction_rng = StdRng::seed_from_u64(trial.seed ^ 0xA5A5_5A5A_D3C4_B2E1);
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
        }

        let best = archive.first().cloned();
        let final_front: Vec<Individual<D::Cand>> = archive
            .iter()
            .filter(|candidate| {
                !archive.iter().any(|other| {
                    other.cand.id() != candidate.cand.id()
                        && other.score.dominates(&candidate.score)
                })
            })
            .cloned()
            .collect();
        let (holdout_best, holdout_baseline) = match &best {
            Some(b) => {
                let holdout = Trial {
                    generation: u64::MAX,
                    seed: self.config.base_seed ^ 0xDEAD_BEEF_CAFE_F00D,
                };
                (
                    Some(evaluate(&self.domain, &b.cand, &holdout)?),
                    Some(self.domain.baseline(&holdout)?),
                )
            }
            None => (None, None),
        };

        let holdout_front_trial = Trial {
            generation: u64::MAX,
            seed: self.config.base_seed ^ 0xDEAD_BEEF_CAFE_F00D,
        };
        let final_front_holdout: Vec<Option<Score>> = final_front
            .iter()
            .map(|ind| evaluate(&self.domain, &ind.cand, &holdout_front_trial).ok())
            .collect();

        Ok(Report {
            best,
            final_front,
            final_front_holdout,
            final_baseline,
            holdout_best,
            holdout_baseline,
            history,
            failure_diagnostics: all_failure_diagnostics_mutex
                .into_inner()
                .unwrap_or_default(),
        })
    }
}

// ---------------------------------------------------------------------------
// WorkerPool — Ordonnanceur dynamique pull-based avec failover
// ---------------------------------------------------------------------------

/// État de charge d'un Worker individuel.
#[derive(Debug)]
struct WorkerSlot {
    addr: String,
    /// Nombre de tâches actuellement actives sur ce worker.
    active_tasks: AtomicUsize,
    /// Nombre total de tâches attribuées (pour métriques).
    _total_assigned: AtomicUsize,
    /// Le worker est-il marqué comme défaillant ?
    failed: AtomicBool,
}

/// Pool de Workers thread-safe avec ordonnancement par moindre charge.
///
/// Stratégie : chaque thread Rayon interroge le pool pour "louer" un worker
/// (lease/release). Le worker le moins chargé est sélectionné. En cas d'échec
/// réseau, la tâche est ré-ordonnancée sur un autre worker (failover).
pub struct WorkerPool {
    slots: Vec<WorkerSlot>,
}

impl WorkerPool {
    pub fn new(addresses: Vec<String>) -> Self {
        let slots: Vec<WorkerSlot> = addresses
            .into_iter()
            .map(|addr| WorkerSlot {
                addr,
                active_tasks: AtomicUsize::new(0),
                _total_assigned: AtomicUsize::new(0),
                failed: AtomicBool::new(false),
            })
            .collect();
        WorkerPool { slots }
    }

    /// Sélectionne le worker le moins chargé parmi les workers sains.
    /// Retourne l'adresse du worker sélectionné.
    fn lease(&self) -> Option<String> {
        let mut best_idx: Option<usize> = None;
        let mut best_load = usize::MAX;

        for (i, slot) in self.slots.iter().enumerate() {
            if slot.failed.load(Ordering::Relaxed) {
                continue;
            }
            let load = slot.active_tasks.load(Ordering::Relaxed);
            if load < best_load {
                best_load = load;
                best_idx = Some(i);
            }
        }

        best_idx.map(|i| {
            self.slots[i].active_tasks.fetch_add(1, Ordering::Relaxed);
            self.slots[i]
                ._total_assigned
                .fetch_add(1, Ordering::Relaxed);
            self.slots[i].addr.clone()
        })
    }

    /// Libère un worker après utilisation (succès ou échec).
    fn release(&self, addr: &str) {
        for slot in &self.slots {
            if slot.addr == addr {
                let prev = slot.active_tasks.fetch_sub(1, Ordering::Relaxed);
                if prev == 0 {
                    // Underflow guard
                    slot.active_tasks.store(0, Ordering::Relaxed);
                }
                return;
            }
        }
    }

    /// Marque un worker comme défaillant.
    fn mark_failed(&self, addr: &str) {
        for slot in &self.slots {
            if slot.addr == addr {
                slot.failed.store(true, Ordering::Relaxed);
                slot.active_tasks.store(0, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Réinitialise tous les workers (nouvelle génération).
    fn reset_all(&self) {
        for slot in &self.slots {
            slot.failed.store(false, Ordering::Relaxed);
            slot.active_tasks.store(0, Ordering::Relaxed);
        }
    }

    /// Vérifie si au moins un worker est sain.
    #[allow(dead_code)]
    fn any_healthy(&self) -> bool {
        self.slots.iter().any(|s| !s.failed.load(Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------
// Évaluation distribuée dynamique avec leasing et failover
// ---------------------------------------------------------------------------

/// Évalue une population via le WorkerPool avec ordonnancement dynamique.
/// Chaque thread Rayon loue un worker, tente l'évaluation, et en cas d'échec
/// réseau ré-essaie sur un autre worker. Si tous les workers sont morts,
/// bascule sur une évaluation locale de secours.
fn evaluate_distributed_dynamic<C: Candidate>(
    domain: &impl Domain<Cand = C>,
    population: &[C],
    pool: &Arc<WorkerPool>,
    trial: &Trial,
    registry: Option<&AlgorithmRegistry>,
    cache: Option<&EvaluationCache>,
    current_gen: u64,
    failure_sink: &Mutex<Vec<FailureDiagnostics>>,
) -> Vec<Individual<C>> {
    let timeout = Duration::from_secs(60);

    // Réinitialiser le pool pour cette génération
    pool.reset_all();

    population
        .par_iter()
        .map(|cand| {
            // Hit cache ?
            if let Some(ref c) = cache {
                if let Some(objs) = c.get_scoped(domain.name(), cand.id(), trial.seed) {
                    return Individual {
                        cand: cand.clone(),
                        score: Score::valid(objs),
                    };
                }
            }

            let mut remaining_attempts = pool.slots.len().max(1);

            loop {
                // Tenter de louer un worker
                let worker_addr = match pool.lease() {
                    Some(addr) => addr,
                    None => {
                        // Aucun worker sain → fallback local silencieux
                        let score = evaluate(domain, cand, trial).unwrap_or(Score::invalid());
                        return Individual {
                            cand: cand.clone(),
                            score,
                        };
                    }
                };

                let payload = EvaluationPayload {
                    candidate_id: cand.id(),
                    source_code: cand.repr(),
                    seed: trial.seed,
                    generation: current_gen,
                };

                match dispatch_evaluation_to_worker(&worker_addr, &payload, timeout) {
                    Ok(eval_res) => {
                        // Succès : libérer le worker
                        pool.release(&worker_addr);

                        if eval_res.is_valid {
                            // Insertion cache
                            if let Some(ref c) = cache {
                                c.insert_scoped(
                                    domain.name(),
                                    cand.id(),
                                    trial.seed,
                                    eval_res.objectives.clone(),
                                );
                            }

                            if let Some(reg) = registry {
                                let record = crate::registry::GenerationRecord {
                                    candidate_id: eval_res.candidate_id,
                                    source_code: payload.source_code,
                                    objectives: eval_res.objectives.clone(),
                                    generation: current_gen,
                                    parent_ids: vec![],
                                };
                                let _ = reg.commit_candidate(&record);
                            }
                        }

                        return Individual {
                            cand: cand.clone(),
                            score: Score {
                                objectives: eval_res.objectives,
                                valid: eval_res.is_valid,
                            },
                        };
                    }
                    Err(e) => {
                        // Échec réseau sur ce worker
                        pool.release(&worker_addr);
                        remaining_attempts -= 1;

                        if remaining_attempts == 0 {
                            // Tous les workers ont échoué → fallback local
                            pool.mark_failed(&worker_addr);

                            if let Ok(mut sink) = failure_sink.lock() {
                                sink.push(FailureDiagnostics::new(
                                    FailureStage::Execution,
                                    String::new(),
                                    format!(
                                        "Tous les workers ont échoué pour candidat id={}. \
                                         Dernière erreur: {e}. Fallback local.",
                                        cand.id()
                                    ),
                                ));
                            }

                            let score = evaluate(domain, cand, trial).unwrap_or(Score::invalid());

                            // Insertion cache fallback
                            if score.valid {
                                if let Some(ref c) = cache {
                                    c.insert_scoped(
                                        domain.name(),
                                        cand.id(),
                                        trial.seed,
                                        score.objectives.clone(),
                                    );
                                }
                            }

                            return Individual {
                                cand: cand.clone(),
                                score,
                            };
                        }

                        // Ré-essayer sur un autre worker
                        if let Ok(mut sink) = failure_sink.lock() {
                            sink.push(FailureDiagnostics::new(
                                FailureStage::Execution,
                                String::new(),
                                format!(
                                    "Échec worker '{worker_addr}' (candidat id={}), \
                                     failover vers un autre worker. Erreur: {e}",
                                    cand.id()
                                ),
                            ));
                        }
                        // Continue la boucle → prochain worker
                    }
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Évaluation parallèle distribuée (Round-Robin legacy)
// ---------------------------------------------------------------------------

/// Évalue une population entière en distribuant les calculs sur un pool
/// de Workers distants via le protocole binaire TCP.
///
/// # Stratégie de répartition
/// Round-Robin : le candidat d'index `i` est envoyé au Worker `i % workers.len()`.
///
/// # Résilience
/// - Si un appel réseau échoue (timeout, connexion refusée), le candidat est
///   marqué comme invalide et un `FailureDiagnostics` est collecté.
/// - La génération n'est **pas** interrompue par les échecs réseau.
/// - Les résultats valides sont persistés dans le registre Sled si fourni.
pub fn evaluate_parallel_distributed<C: Candidate>(
    population: &[C],
    workers: &[String],
    trial: &Trial,
    registry: Option<&AlgorithmRegistry>,
    cache: Option<&EvaluationCache>,
    current_gen: u64,
    failure_sink: &Mutex<Vec<FailureDiagnostics>>,
) -> Vec<Individual<C>> {
    let timeout = Duration::from_secs(60);

    population
        .par_iter()
        .enumerate()
        .map(|(idx, cand)| {
            // Hit cache ?
            if let Some(ref c) = cache {
                if let Some(objs) = c.get(cand.id()) {
                    return Individual {
                        cand: cand.clone(),
                        score: Score::valid(objs),
                    };
                }
            }

            let worker_addr = &workers[idx % workers.len()];

            let payload = EvaluationPayload {
                candidate_id: cand.id(),
                source_code: cand.repr(),
                seed: trial.seed,
                generation: current_gen,
            };

            match dispatch_evaluation_to_worker(worker_addr, &payload, timeout) {
                Ok(eval_res) => {
                    if eval_res.is_valid {
                        // Insertion cache
                        if let Some(ref c) = cache {
                            c.insert(cand.id(), eval_res.objectives.clone());
                        }

                        if let Some(reg) = registry {
                            let record = crate::registry::GenerationRecord {
                                candidate_id: eval_res.candidate_id,
                                source_code: payload.source_code,
                                objectives: eval_res.objectives.clone(),
                                generation: current_gen,
                                parent_ids: vec![],
                            };
                            let _ = reg.commit_candidate(&record);
                        }
                    }

                    Individual {
                        cand: cand.clone(),
                        score: Score {
                            objectives: eval_res.objectives,
                            valid: eval_res.is_valid,
                        },
                    }
                }
                Err(e) => {
                    if let Ok(mut sink) = failure_sink.lock() {
                        sink.push(FailureDiagnostics::new(
                            FailureStage::Execution,
                            String::new(),
                            format!(
                                "Échec réseau vers worker '{worker_addr}' \
                                 (candidat id={}): {e}",
                                cand.id()
                            ),
                        ));
                    }

                    Individual {
                        cand: cand.clone(),
                        score: Score::invalid(),
                    }
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Harnais d'évaluation unifié (local)
// ---------------------------------------------------------------------------

fn evaluate<D: Domain>(domain: &D, cand: &D::Cand, trial: &Trial) -> Result<Score> {
    if !domain.verify(cand, trial)? {
        return Ok(Score::invalid());
    }
    let objectives = domain.measure(cand, trial)?;
    if objectives.is_empty() || !objectives.iter().all(|x| x.is_finite()) {
        return Ok(Score::invalid());
    }
    Ok(Score::valid(objectives))
}

fn cmp_primary(a: &Score, b: &Score) -> std::cmp::Ordering {
    let av = a.objectives.first().copied().unwrap_or(f64::INFINITY);
    let bv = b.objectives.first().copied().unwrap_or(f64::INFINITY);
    av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
}

/// Trie l'archive en appliquant une affectation par fronts de non-domination de Pareto,
/// puis départage les individus d'un même front par leur Distance d'Encombrement.
pub fn sort_by_pareto_domination<C: Candidate>(archive: &mut Vec<Individual<C>>) {
    let len = archive.len();
    if len <= 1 {
        return;
    }

    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut domination_counts = vec![0usize; len];
    let mut dominates_list: Vec<Vec<usize>> = vec![Vec::new(); len];

    for i in 0..len {
        for j in 0..len {
            if i == j {
                continue;
            }
            if archive[i].score.dominates(&archive[j].score) {
                dominates_list[i].push(j);
            } else if archive[j].score.dominates(&archive[i].score) {
                domination_counts[i] += 1;
            }
        }
        if domination_counts[i] == 0 {
            if fronts.is_empty() {
                fronts.push(Vec::new());
            }
            fronts[0].push(i);
        }
    }

    let mut current_front_idx = 0;
    while current_front_idx < fronts.len() {
        let mut next_front = Vec::new();
        for &p in &fronts[current_front_idx] {
            for &q in &dominates_list[p] {
                domination_counts[q] -= 1;
                if domination_counts[q] == 0 {
                    next_front.push(q);
                }
            }
        }
        if !next_front.is_empty() {
            fronts.push(next_front);
        }
        current_front_idx += 1;
    }

    let mut crowding_distances = vec![0.0f64; len];
    let num_objectives = archive[0].score.objectives.len();

    for front in &fronts {
        let front_len = front.len();
        if front_len == 0 {
            continue;
        }
        if front_len <= 2 {
            for &idx in front {
                crowding_distances[idx] = f64::INFINITY;
            }
            continue;
        }

        for obj_idx in 0..num_objectives {
            let mut sorted_front_indices = front.clone();
            sorted_front_indices.sort_by(|&a, &b| {
                archive[a].score.objectives[obj_idx]
                    .partial_cmp(&archive[b].score.objectives[obj_idx])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            crowding_distances[sorted_front_indices[0]] = f64::INFINITY;
            crowding_distances[sorted_front_indices[front_len - 1]] = f64::INFINITY;

            let min_val = archive[sorted_front_indices[0]].score.objectives[obj_idx];
            let max_val = archive[sorted_front_indices[front_len - 1]]
                .score
                .objectives[obj_idx];
            let norm = max_val - min_val;

            if norm > 1e-9 {
                for k in 1..(front_len - 1) {
                    let prev_idx = sorted_front_indices[k - 1];
                    let next_idx = sorted_front_indices[k + 1];
                    let curr_idx = sorted_front_indices[k];

                    if crowding_distances[curr_idx] != f64::INFINITY {
                        crowding_distances[curr_idx] += (archive[next_idx].score.objectives
                            [obj_idx]
                            - archive[prev_idx].score.objectives[obj_idx])
                            / norm;
                    }
                }
            }
        }
    }

    let mut flat_ranks = vec![0usize; len];
    for (front_rank, front) in fronts.iter().enumerate() {
        for &idx in front {
            flat_ranks[idx] = front_rank;
        }
    }

    let mut annotated: Vec<(Individual<C>, usize, f64)> = archive
        .drain(..)
        .enumerate()
        .map(|(idx, ind)| (ind, flat_ranks[idx], crowding_distances[idx]))
        .collect();

    annotated.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    archive.extend(annotated.into_iter().map(|(ind, _, _)| ind));
}

// ── Évaluation avec boucle de rétroaction ─────────────────────────────

use crate::registry::GenerationRecord;

pub fn evaluate_with_feedback<D: Domain>(
    domain: &D,
    cand: &D::Cand,
    trial: &Trial,
    registry: &AlgorithmRegistry,
    current_gen: u64,
    parents: &[CandidateId],
) -> std::result::Result<Score, FailureDiagnostics> {
    match domain.verify(cand, trial) {
        Ok(true) => match domain.measure(cand, trial) {
            Ok(objectives) => {
                let record = GenerationRecord {
                    candidate_id: cand.id(),
                    source_code: cand.repr(),
                    objectives: objectives.clone(),
                    generation: current_gen,
                    parent_ids: parents.to_vec(),
                };
                let _ = registry.commit_candidate(&record);
                Ok(Score::valid(objectives))
            }
            Err(e) => Err(FailureDiagnostics::new(
                FailureStage::Execution,
                String::new(),
                e.to_string(),
            )),
        },
        Ok(false) => Err(FailureDiagnostics::new(
            FailureStage::Compilation,
            String::new(),
            "Code rejete a la compilation ou validation syntaxique.".into(),
        )),
        Err(e) => Err(FailureDiagnostics::new(
            FailureStage::Verification,
            String::new(),
            format!("Crash critique ou violation mathematique: {e}"),
        )),
    }
}

/// Helper trait pour désérialiser un candidat depuis sa source (reprise).
pub trait DeserializeFromSource: Sized {
    fn deserialize_from_source(_source: &str) -> Option<Self> {
        None
    }
}

// Implémentations par défaut (domaines concrets override si nécessaire)
impl DeserializeFromSource for crate::domains::low_rank::TensorCode {
    fn deserialize_from_source(source: &str) -> Option<Self> {
        Some(Self {
            raw_source: source.to_string(),
            id: crate::fnv1a(source),
        })
    }
}

impl DeserializeFromSource for crate::domains::simd_kernel::SimdKernelCode {
    fn deserialize_from_source(source: &str) -> Option<Self> {
        Some(Self {
            source: source.to_string(),
            id: crate::fnv1a(source),
        })
    }
}

impl DeserializeFromSource for crate::domains::cuda_kernel::CudaCode {
    fn deserialize_from_source(source: &str) -> Option<Self> {
        Some(Self {
            source: source.to_string(),
            id: crate::fnv1a(source),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_with_workers() {
        let config = Config {
            generations: 1,
            population: 4,
            survivors: 2,
            base_seed: 42,
            worker_addresses: Some(vec!["127.0.0.1:9999".to_string()]),
        };
        assert!(config.worker_addresses.is_some());
        assert_eq!(config.worker_addresses.unwrap().len(), 1);
    }

    #[test]
    fn test_config_without_workers() {
        let config = Config {
            generations: 1,
            population: 4,
            survivors: 2,
            base_seed: 42,
            worker_addresses: None,
        };
        assert!(config.worker_addresses.is_none());
    }

    #[test]
    fn test_worker_pool_lease_release() {
        let pool = WorkerPool::new(vec![
            "127.0.0.1:9000".into(),
            "127.0.0.1:9001".into(),
            "127.0.0.1:9002".into(),
        ]);

        let w1 = pool.lease().unwrap();
        let w2 = pool.lease().unwrap();
        let w3 = pool.lease().unwrap();

        // Les 3 workers devraient être attribués (un par lease)
        assert!(w1 != w2 || w1 != w3);

        pool.release(&w1);
        pool.release(&w2);
        pool.release(&w3);
    }

    #[test]
    fn test_worker_pool_failover() {
        let pool = WorkerPool::new(vec!["127.0.0.1:9000".into(), "127.0.0.1:9001".into()]);

        let w1 = pool.lease().unwrap();
        pool.mark_failed(&w1);
        pool.release(&w1);

        // Le worker 0 est mort, le lease suivant doit donner l'autre
        let w2 = pool.lease().unwrap();
        assert_ne!(w2, w1);
        pool.release(&w2);

        // Après reset, tous les workers sont à nouveau sains
        pool.reset_all();
        assert!(pool.any_healthy());
        assert_eq!(
            pool.slots
                .iter()
                .filter(|s| !s.failed.load(Ordering::Relaxed))
                .count(),
            2
        );
    }

    // ---- Domaine mock : un echec d'evaluation isole ne doit pas contaminer
    // toute la generation (regression du court-circuit collect). ----
    use crate::{Candidate, CandidateId, Domain, ForgeError, Score};
    use std::sync::atomic::AtomicUsize;

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct MockCand {
        id: u64,
        poison: bool,
    }
    impl Candidate for MockCand {
        fn id(&self) -> CandidateId {
            self.id
        }
        fn repr(&self) -> String {
            format!("mock-{}", self.id)
        }
    }

    struct MockDomain {
        seed_counter: AtomicUsize,
    }
    impl Domain for MockDomain {
        type Cand = MockCand;
        fn name(&self) -> &str {
            "mock"
        }
        fn seed(&self, _rng: &mut StdRng) -> MockCand {
            // 1er candidat tire => echoue a l'eval ; les suivants sont valides.
            let i = self.seed_counter.fetch_add(1, Ordering::Relaxed);
            MockCand {
                id: i as u64,
                poison: i == 0,
            }
        }
        fn mutate(&self, _rng: &mut StdRng, _parents: &[&MockCand]) -> Result<MockCand> {
            Ok(MockCand {
                id: 9999,
                poison: false,
            })
        }
        fn verify(&self, cand: &MockCand, _trial: &Trial) -> Result<bool> {
            if cand.poison {
                Err(ForgeError::Evaluation("echec simule".into()))
            } else {
                Ok(true)
            }
        }
        fn measure(&self, cand: &MockCand, _trial: &Trial) -> Result<Vec<f64>> {
            Ok(vec![cand.id as f64])
        }
        fn objective_names(&self) -> Vec<String> {
            vec!["mock_obj".into()]
        }
        fn baseline(&self, _trial: &Trial) -> Result<Score> {
            Ok(Score::valid(vec![999.0]))
        }
    }

    #[test]
    fn eval_failure_does_not_poison_generation() {
        let config = Config {
            generations: 1,
            population: 4,
            survivors: 3,
            base_seed: 7,
            worker_addresses: None,
        };
        let domain = MockDomain {
            seed_counter: AtomicUsize::new(0),
        };
        let report = Engine::new(domain, config).run().expect("run ok");
        // 1 candidat sur 4 echoue ; sans le fix tout serait invalide (front vide, history INF).
        assert!(
            report.best.is_some(),
            "un candidat valide doit survivre malgre l'echec d'un autre"
        );
        assert!(
            !report.final_front.is_empty(),
            "le front ne doit pas etre vide"
        );
        assert!(
            report.history[0].is_finite(),
            "history[0] doit etre fini, valeur={}",
            report.history[0]
        );
    }
}
