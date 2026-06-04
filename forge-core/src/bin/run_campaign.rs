use forge_core::domains::low_rank::TensorTrainDomain;
use forge_core::{Config, Engine};

fn main() {
    let endpoint = std::env::var("OLLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:11434/api/generate".to_string());
    let model = std::env::var("OLLAMA_MODEL")
        .unwrap_or_else(|_| "qwen2.5-coder:1.5b".to_string());
    println!("== Campagne low_rank :: Ollama {model} @ {endpoint} ==");

    let domain = TensorTrainDomain::new("/tmp/forge_campaign_lowrank").with_llm(&endpoint, &model);
    let envu = |k: &str, d: u64| -> u64 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) };
    let config = Config {
        generations: envu("GENERATIONS", 3),
        population: envu("POPULATION", 4) as usize,
        survivors: envu("SURVIVORS", 2) as usize,
        base_seed: envu("BASE_SEED", 42),
        worker_addresses: None,
    };
    eprintln!("[forge] campagne: generations={} population={} survivors={}", config.generations, config.population, config.survivors);

    match Engine::new(domain, config).run() {
        Ok(report) => {
            println!("\n=== campagne terminee ===");
            for (g, h) in report.history.iter().enumerate() {
                println!("  gen {g:>2}  meilleur reconstruction_error_L2 = {h:.6e}");
            }
            println!("\n--- front de Pareto final ({} candidats) ---", report.final_front.len());
            for (i, ind) in report.final_front.iter().enumerate() {
                let o = &ind.score.objectives;
                let g0 = o.get(0).copied().unwrap_or(f64::NAN);
                let g1 = o.get(1).copied().unwrap_or(f64::NAN);
                let g2 = o.get(2).copied().unwrap_or(f64::NAN);
                println!("  [{i}] L2={g0:.3e}  latency_ns={g1:.0}  params={g2:.0}");
            }
            if let Some(bl) = report.final_baseline.as_ref() {
                let o = &bl.objectives;
                let g0 = o.get(0).copied().unwrap_or(f64::NAN);
                let g1 = o.get(1).copied().unwrap_or(f64::NAN);
                let g2 = o.get(2).copied().unwrap_or(f64::NAN);
                println!("  baseline  L2={g0:.3e}  latency_ns={g1:.0}  params={g2:.0}");
            }
        }
        Err(e) => { eprintln!("erreur de campagne: {e}"); std::process::exit(1); }
    }
}
