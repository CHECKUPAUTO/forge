//! forge-cli — Outil d'administration et d'analytics pour l'écosystème Forge.
//!
//! ## Sous-commandes
//! - `analytics --db <path>` : extrait et affiche le front de Pareto depuis Sled.
//! - `checkpoint --db <path> --domain <name>` : inspecte un checkpoint moteur.
//! - `resume ...` : alias historique de `checkpoint`; il n'exécute pas la reprise.
//!
//! La reprise d'une campagne nécessite de reconstruire le domaine concret puis
//! d'appeler `Engine::resume_from_state`. Le CLI n'invente pas cette configuration.

use std::env;
use std::process;

use forge_core::registry::{AlgorithmRegistry, GenerationRecord};
use forge_core::sort_by_pareto_domination;
use forge_core::{Candidate, EngineState, ForgeError, Individual, Score};
use serde::de::DeserializeOwned;
use serde::Serialize;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let result = match args[1].as_str() {
        "analytics" => run_analytics(&args),
        "checkpoint" => run_checkpoint(&args),
        "resume" => {
            eprintln!(
                "Avertissement: `resume` est un alias historique d'inspection; \
                 le CLI ne relance pas automatiquement la campagne."
            );
            run_checkpoint(&args)
        }
        other => {
            eprintln!("Commande inconnue: '{other}'");
            print_usage();
            process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Erreur: {e}");
        process::exit(1);
    }
}

fn print_usage() {
    eprintln!("Usage: forge-cli <analytics|checkpoint|resume> [options]");
    eprintln!("  analytics  --db <path>");
    eprintln!("  checkpoint --db <path> --domain <low_rank|simd_gemm|cuda_gemm>");
    eprintln!("  resume     alias historique de checkpoint (inspection uniquement)");
}

fn run_analytics(args: &[String]) -> Result<(), ForgeError> {
    let db_path = parse_flag(args, "--db")?;

    println!("══════════════════════════════════════════════");
    println!("  forge-cli — Pareto Front Analytics");
    println!("══════════════════════════════════════════════");
    println!();
    println!("Base Sled : {db_path}");
    println!();

    let registry = AlgorithmRegistry::open(&db_path)?;
    let records: Vec<GenerationRecord> = registry.iter().collect::<Result<Vec<_>, ForgeError>>()?;

    if records.is_empty() {
        println!("Aucun enregistrement candidat trouvé dans la base.");
        return Ok(());
    }

    println!("{} candidats enregistrés", records.len());
    println!();

    let total = records.len();
    let valid_count = records.iter().filter(|r| !r.objectives.is_empty()).count();
    let validity_rate = (valid_count as f64) / (total as f64) * 100.0;

    println!("── Statistiques globales ──");
    println!("  Candidats totaux   : {total}");
    println!("  Candidats valides  : {valid_count}");
    println!("  Taux de validité   : {validity_rate:.1}%");
    println!();

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
        println!("Aucun candidat valide avec objectifs.");
        return Ok(());
    }

    sort_by_pareto_domination(&mut individuals);
    let pareto_front = extract_pareto_front(&individuals);

    println!("── Front de Pareto (candidats non dominés) ──");
    println!("  {} individus sur le front", pareto_front.len());
    println!();
    println!(
        "  {0: <6} {1: <20} {2: <20}",
        "Rang", "Objectif 0", "Objectif 1"
    );
    println!("  {:-<6} {:-<20} {:-<20}", "", "", "");

    for (rank, ind) in pareto_front.iter().enumerate() {
        let obj0 = ind.score.objectives.first().copied().unwrap_or(f64::NAN);
        let obj1 = ind.score.objectives.get(1).copied().unwrap_or(f64::NAN);
        println!("  #{rank:<4} {obj0:<20.6e} {obj1:<20.6e}", rank = rank + 1,);
    }
    println!();

    let max_gen = records.iter().map(|r| r.generation).max().unwrap_or(0);
    println!("── Évolution de l'objectif principal par génération ──");
    for generation in 0..=max_gen {
        let mut values: Vec<f64> = records
            .iter()
            .filter(|r| r.generation == generation && !r.objectives.is_empty())
            .filter_map(|r| r.objectives.first().copied())
            .filter(|v| v.is_finite())
            .collect();

        if values.is_empty() {
            continue;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = values[values.len() / 2];
        let min = values.first().copied().unwrap_or(f64::NAN);
        let max = values.last().copied().unwrap_or(f64::NAN);
        println!(
            "  Génération {generation:>3} : médiane={median:>12.2e}  min={min:>12.2e}  max={max:>12.2e}"
        );
    }

    Ok(())
}

fn run_checkpoint(args: &[String]) -> Result<(), ForgeError> {
    let db_path = parse_flag(args, "--db")?;
    let domain_name = parse_flag(args, "--domain")?;
    let registry = AlgorithmRegistry::open(&db_path)?;

    println!("══════════════════════════════════════════════");
    println!("  forge-cli — Inspection checkpoint");
    println!("══════════════════════════════════════════════");
    println!("Base Sled : {db_path}");
    println!("Domaine   : {domain_name}");
    println!();

    match domain_name.as_str() {
        "low_rank" | "low_rank_compression" => {
            inspect_checkpoint::<forge_core::domains::low_rank::TensorCode>(&registry)
        }
        "simd" | "simd_gemm" | "simd_kernel" => {
            inspect_checkpoint::<forge_core::domains::simd_kernel::SimdKernelCode>(&registry)
        }
        "cuda" | "cuda_gemm" | "cuda_kernel" => {
            inspect_checkpoint::<forge_core::domains::cuda_kernel::CudaCode>(&registry)
        }
        other => Err(ForgeError::Evaluation(format!(
            "Domaine checkpoint non supporté: '{other}'. Utiliser low_rank, simd_gemm ou cuda_gemm."
        ))),
    }
}

fn inspect_checkpoint<C>(registry: &AlgorithmRegistry) -> Result<(), ForgeError>
where
    C: Candidate + Serialize + DeserializeOwned,
{
    match EngineState::<C>::load_from_sled(registry)? {
        Some(state) => {
            println!("Checkpoint trouvé.");
            println!("  Prochaine génération : {}", state.current_generation);
            println!(
                "  Population sauvegardée: {}",
                state.population_sources.len()
            );
            println!("  Archive d'élites      : {}", state.archive.len());
            println!(
                "  Historique             : {} générations",
                state.history.len()
            );
            println!(
                "  Diagnostics d'échec    : {}",
                state.failure_diagnostics.len()
            );
            if let Some(best_obj) = state.history.last() {
                println!("  Dernier meilleur obj.  : {best_obj:.6e}");
            }
            println!();
            println!(
                "Inspection uniquement. Pour reprendre l'exécution, reconstruire le domaine \
                 avec la même configuration puis utiliser Engine::resume_from_state()."
            );
        }
        None => {
            println!("Aucun checkpoint moteur trouvé dans cette base Sled.");
        }
    }
    Ok(())
}

fn parse_flag(args: &[String], flag: &str) -> Result<String, ForgeError> {
    for (index, value) in args.iter().enumerate().skip(1) {
        if value == flag {
            return args.get(index + 1).cloned().ok_or_else(|| {
                ForgeError::Evaluation(format!("Valeur manquante pour le flag '{flag}'"))
            });
        }
    }
    Err(ForgeError::Evaluation(format!(
        "Flag requis '{flag}' non trouvé"
    )))
}

#[derive(Clone)]
struct StubCandidate {
    id: u64,
    source: String,
}

impl Candidate for StubCandidate {
    fn id(&self) -> forge_core::CandidateId {
        self.id
    }

    fn repr(&self) -> String {
        self.source.clone()
    }
}

fn extract_pareto_front<C: Candidate>(individuals: &[Individual<C>]) -> Vec<&Individual<C>> {
    let mut front = Vec::new();

    for ind in individuals {
        let is_dominated = front.iter().any(|f: &&Individual<C>| {
            f.score.dominates(&ind.score) && f.cand.id() != ind.cand.id()
        });

        if !is_dominated {
            front.retain(|f: &&Individual<C>| !ind.score.dominates(&f.score));
            front.push(ind);
        }
    }

    front
}
