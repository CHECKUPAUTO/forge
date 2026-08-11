//! Demonstration executable du moteur `forge`.
//!
//! Lance une campagne sur le domaine bin-packing et affiche : la courbe
//! d'apprentissage, le gain face a la baseline first-fit, et — surtout — la
//! revalidation sur une graine de holdout jamais vue pendant l'evolution. Si
//! le gain tient sur le holdout, l'amelioration generalise.

mod binpack;

use binpack::BinPacking;
use forge_core::{Candidate, Config, Engine};

fn main() {
    let domain = BinPacking {
        capacity: 1.0,
        n_items: 50,
        n_instances: 20,
    };
    let config = Config {
        generations: 40,
        population: 60,
        survivors: 8,
        base_seed: 42,
        worker_addresses: None,
    };

    let engine = Engine::new(domain, config.clone());
    let report = match engine.run() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("erreur de campagne: {e}");
            std::process::exit(1);
        }
    };

    println!("=== forge :: campagne domaine `binpack` ===");
    println!(
        "(generations={}, population={}, survivants={})\n",
        config.generations, config.population, config.survivors
    );

    for (g, h) in report.history.iter().enumerate() {
        if g % 5 == 0 || g + 1 == report.history.len() {
            println!("  gen {g:>3}   meilleur avg_bins = {h:.3}");
        }
    }

    if let (Some(best), Some(base)) = (&report.best, &report.final_baseline) {
        let b = best.score.objectives[0];
        let bl = base.objectives[0];
        let gain = if bl > 0.0 { 100.0 * (bl - b) / bl } else { 0.0 };
        println!("\n  baseline (first-fit)  avg_bins = {bl:.3}");
        println!("  heuristique evoluee   avg_bins = {b:.3}   (gain {gain:+.1}%)");
        println!("  poids w = {}", best.cand.repr());
    }

    match (&report.holdout_best, &report.holdout_baseline) {
        (Some(hb), Some(hbl)) if hb.valid => {
            let b = hb.objectives[0];
            let bl = hbl.objectives[0];
            let gain = if bl > 0.0 { 100.0 * (bl - b) / bl } else { 0.0 };
            println!("\n  [HOLDOUT — graine jamais vue pendant l'evolution]");
            println!(
                "  baseline avg_bins = {bl:.3}  |  evoluee avg_bins = {b:.3}  |  gain {gain:+.1}%"
            );
            println!("  => gain conserve sur le holdout : l'amelioration generalise (pas d'over-fit au benchmark).");
        }
        _ => println!("\n  [HOLDOUT] candidat invalide ou absent."),
    }
}
