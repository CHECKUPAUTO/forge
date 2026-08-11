//! forge-cli — Outil d'administration et d'analytics pour l'écosystème forge.
//!
//! ## Sous-commandes
//! - `analytics --db <path>` : Extrait et affiche le front de Pareto depuis Sled DB.
//! - `resume --db <path> --domain <name>` : Reprend une campagne interrompue
//!   via le checkpoint Sled.
//!
//! ## Utilisation
//! ```bash
//! forge-cli analytics --db /path/to/sled_registry
//! forge-cli resume --db /path/to/sled_registry --domain low_rank
//! ```

use std::env;
use std::process;

use forge_core::registry::{AlgorithmRegistry, GenerationRecord};
use forge_core::sort_by_pareto_domination;
use forge_core::{ForgeError, Individual, Score};

// ---------------------------------------------------------------------------
// Point d'entrée
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: forge-cli <analytics|resume> [options]");
        eprintln!("  analytics --db <path>     Extraire le front de Pareto");
        eprintln!("  resume    --db <path> --domain <name>  Reprendre une campagne");
        process::exit(1);
    }

    let cmd = &args[1];

    match cmd.as_str() {
        "analytics" => {
            if let Err(e) = run_analytics(&args) {
                eprintln!("Erreur analytics: {e}");
                process::exit(1);
            }
        }
        "resume" => {
            if let Err(e) = run_resume(&args) {
                eprintln!("Erreur resume: {e}");
                process::exit(1);
            }
        }
        other => {
            eprintln!("Commande inconnue: '{other}'");
            eprintln!("Commandes disponibles: analytics, resume");
            process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Sous-commande: analytics
// ---------------------------------------------------------------------------

fn run_analytics(args: &[String]) -> Result<(), ForgeError> {
    let db_path = parse_flag(args, "--db")?;

    println!("══════════════════════════════════════════════");
    println!("  forge-cli — Pareto Front Analytics");
    println!("══════════════════════════════════════════════");
    println!();
    println!("📂 Base Sled : {db_path}");
    println!();

    let registry = AlgorithmRegistry::open(&db_path)?;

    // Collecte de tous les enregistrements
    let records: Vec<GenerationRecord> = registry.iter().collect::<Result<Vec<_>, ForgeError>>()?;

    if records.is_empty() {
        println!("⚠️  Aucun enregistrement trouvé dans la base.");
        return Ok(());
    }

    println!("📊 {count} candidats enregistrés", count = records.len());
    println!();

    // Statistiques globales
    let total = records.len();
    let valid_count = records.iter().filter(|r| !r.objectives.is_empty()).count();
    let validity_rate = if total > 0 {
        (valid_count as f64) / (total as f64) * 100.0
    } else {
        0.0
    };

    println!("── Statistiques Globales ──");
    println!("  Candidats totaux   : {total}");
    println!("  Candidats valides  : {valid_count}");
    println!("  Taux de validité   : {validity_rate:.1}%");
    println!();

    // Conversion en individus pour le tri de Pareto
    let mut individuals: Vec<Individual<StubCandidate>> = records
        .iter()
        .filter(|r| !r.objectives.is_empty())
        .map(|r| Individual {
            cand: StubCandidate {
                id: r.candidate_id,
                source: r.source_code.clone(),
            },
            score: Score::valid(r.objectives.clone()),
        })
        .collect();

    if individuals.is_empty() {
        println!("⚠️  Aucun candidat valide avec objectifs.");
        return Ok(());
    }

    // Tri par non-domination de Pareto
    sort_by_pareto_domination(&mut individuals);

    // Extraction du Front de Pareto (premier front uniquement)
    let pareto_front = extract_pareto_front(&individuals);

    println!("── Front de Pareto (candidats non-dominés) ──");
    println!(
        "  {count} individus sur le front",
        count = pareto_front.len()
    );
    println!();

    // Affichage ASCII
    println!(
        "  {0: <6} {1: <20} {2: <20}",
        "Rang", "Objectif 0", "Objectif 1"
    );
    println!("  {:-<6} {:-<20} {:-<20}", "", "", "");

    for (rank, ind) in pareto_front.iter().enumerate() {
        let obj0 = ind.score.objectives.first().copied().unwrap_or(f64::NAN);
        let obj1 = ind.score.objectives.get(1).copied().unwrap_or(f64::NAN);
        println!(
            "  #{rank:<4} {obj0:<20.6e} {obj1:<20.6e}",
            rank = rank + 1,
            obj0 = obj0,
            obj1 = obj1,
        );
    }
    println!();

    // Évolution de la latence médiane par génération
    let max_gen = records.iter().map(|r| r.generation).max().unwrap_or(0);
    println!("── Évolution de la latence médiane ──");
    for gen in 0..=max_gen {
        let gen_records: Vec<&GenerationRecord> = records
            .iter()
            .filter(|r| r.generation == gen && !r.objectives.is_empty())
            .collect();

        if gen_records.is_empty() {
            continue;
        }

        let mut latencies: Vec<f64> = gen_records
            .iter()
            .filter_map(|r| r.objectives.first().copied())
            .filter(|v| v.is_finite())
            .collect();

        if latencies.is_empty() {
            continue;
        }

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = latencies[latencies.len() / 2];
        let min = latencies.first().copied().unwrap_or(f64::NAN);
        let max = latencies.last().copied().unwrap_or(f64::NAN);

        println!(
            "  Génération {gen:>3} : médiane={median:>12.2e}  min={min:>12.2e}  max={max:>12.2e}",
        );
    }

    println!();
    println!("✅ Analyse terminée.");

    Ok(())
}

// ---------------------------------------------------------------------------
// Sous-commande: resume
// ---------------------------------------------------------------------------

fn run_resume(args: &[String]) -> Result<(), ForgeError> {
    let db_path = parse_flag(args, "--db")?;
    let domain_name = parse_flag(args, "--domain")?;

    println!("══════════════════════════════════════════════");
    println!("  forge-cli — Reprise sur crash");
    println!("══════════════════════════════════════════════");
    println!();
    println!("📂 Base Sled  : {db_path}");
    println!("🎯 Domaine    : {domain_name}");
    println!();

    let registry = AlgorithmRegistry::open(&db_path)?;

    // Vérifier qu'un checkpoint existe
    let state_opt =
        forge_core::EngineState::<forge_core::domains::low_rank::TensorCode>::load_from_sled(
            &registry,
        )?;

    match state_opt {
        Some(state) => {
            println!("✅ Checkpoint trouvé !");
            println!();
            println!("── État du checkpoint ──");
            println!("  Génération        : {}", state.current_generation);
            println!("  Archive d'élites  : {} individus", state.archive.len());
            println!("  Historique        : {} générations", state.history.len());
            println!(
                "  Échecs cumulés    : {} diagnostics",
                state.failure_diagnostics.len()
            );
            println!();

            // Affichage du meilleur objectif connu
            if let Some(best_obj) = state.history.last() {
                println!("  Meilleur obj. principal : {best_obj:.6e}",);
            }

            println!();
            println!("💡 Pour relancer le moteur depuis ce checkpoint :");
            println!("   Utilise Engine::resume_from_state() dans ton code");
            println!("   ou relance la campagne avec le même registre Sled.");
        }
        None => {
            println!("⚠️  Aucun checkpoint trouvé dans la base Sled.");
            println!();
            println!("   Vérifie que :");
            println!("   1. Le chemin de la base est correct");
            println!("   2. Une campagne a bien été exécutée avec with_registry()");
            println!("   3. Le moteur a sauvegardé au moins une génération");
        }
    }

    println!();
    println!("✅ Inspection terminée.");

    Ok(())
}

// ---------------------------------------------------------------------------
// Utilitaires
// ---------------------------------------------------------------------------

/// Parse un flag de la forme `--flag <value>`.
fn parse_flag(args: &[String], flag: &str) -> Result<String, ForgeError> {
    for i in 1..args.len() {
        if args[i] == flag {
            if i + 1 < args.len() {
                return Ok(args[i + 1].clone());
            } else {
                return Err(ForgeError::Evaluation(format!(
                    "Valeur manquante pour le flag '{flag}'"
                )));
            }
        }
    }
    Err(ForgeError::Evaluation(format!(
        "Flag requis '{flag}' non trouvé"
    )))
}

// ---------------------------------------------------------------------------
// Stub candidate pour l'affichage du front de Pareto
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct StubCandidate {
    id: u64,
    #[allow(dead_code)]
    source: String,
}

impl forge_core::Candidate for StubCandidate {
    fn id(&self) -> forge_core::CandidateId {
        self.id
    }
    fn repr(&self) -> String {
        self.source.clone()
    }
}

// ---------------------------------------------------------------------------
// Extraction du front de Pareto
// ---------------------------------------------------------------------------

/// Extrait les individus du premier front de Pareto (non-dominés).
/// Suppose que l'archive a déjà été triée par `sort_by_pareto_domination`.
fn extract_pareto_front<C: forge_core::Candidate>(
    individuals: &[Individual<C>],
) -> Vec<&Individual<C>> {
    if individuals.is_empty() {
        return vec![];
    }

    let mut front = Vec::new();

    for ind in individuals {
        // Vérifier si cet individu est dominé par un autre déjà dans le front
        let is_dominated = front.iter().any(|f: &&Individual<C>| {
            f.score.dominates(&ind.score) && f.cand.id() != ind.cand.id()
        });

        if !is_dominated {
            // Ajouter et purger les dominés
            front.retain(|f: &&Individual<C>| !ind.score.dominates(&f.score));
            front.push(ind);
        }
    }

    front
}
