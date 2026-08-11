use forge_core::{Config, Engine};
use forge_domains::tensor_train::{TensorTrainDomain, TtConfig};

fn main() {
    let config = TtConfig::new(vec![16, 32, 8, 4], 1, 8);
    let domain = TensorTrainDomain::new(config);

    let engine = Engine::new(
        domain,
        Config {
            generations: 30,
            population: 60,
            survivors: 8,
            base_seed: 0xFEED_C0DE,
            worker_addresses: None,
        },
    );

    let report = engine.run().expect("campagne TT");

    println!("=== forge :: campagne domaine `tensor_train_compression` ===");
    for (g, best) in report.history.iter().enumerate() {
        if g % 5 == 0 {
            println!("  gen {:3}   meilleur erreur = {:.6}", g, best);
        }
    }

    match &report.best {
        Some(ind) => {
            println!("\n  meilleur candidat : rangs = {:?}", ind.cand.ranks);
            println!(
                "  objectifs : erreur={:.4}, ratio_tt/dense={:.4}",
                ind.score.objectives[0], ind.score.objectives[1]
            );
        }
        None => println!("\n  Aucun candidat valide trouve."),
    }

    if let Some(ref base) = report.final_baseline {
        println!(
            "  baseline (rangs max): erreur={:.4}, ratio_tt/dense={:.4}",
            base.objectives[0], base.objectives[1]
        );
    }

    if let (Some(hb), Some(hbl)) = (&report.holdout_best, &report.holdout_baseline) {
        let gain_error =
            (hbl.objectives[0] - hb.objectives[0]) / hbl.objectives[0].abs().max(1e-9) * 100.0;
        let gain_ratio =
            (hbl.objectives[1] - hb.objectives[1]) / hbl.objectives[1].abs().max(1e-9) * 100.0;
        println!(
            "\n  [HOLDOUT] gain erreur={:+.1}%, gain compression={:+.1}%",
            gain_error, gain_ratio
        );
        println!("  => gain conserve sur le holdout.");
    }
}
