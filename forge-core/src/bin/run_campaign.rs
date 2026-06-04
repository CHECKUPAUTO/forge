use forge_core::domains::low_rank::TensorTrainDomain;
use forge_core::{Config, Engine};

fn main() {
    let endpoint = std::env::var("OLLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:11434/api/generate".to_string());
    let model = std::env::var("OLLAMA_MODEL")
        .unwrap_or_else(|_| "qwen2.5-coder:1.5b".to_string());
    println!("== Campagne low_rank :: Ollama {model} @ {endpoint} ==");

    let domain = TensorTrainDomain::new("/tmp/forge_campaign_lowrank").with_llm(&endpoint, &model);
    let config = Config { generations: 3, population: 4, survivors: 2, base_seed: 42, worker_addresses: None };

    match Engine::new(domain, config).run() {
        Ok(report) => {
            println!("\n=== campagne terminee ===");
            for (g, h) in report.history.iter().enumerate() {
                println!("  gen {g:>2}  meilleur reconstruction_error_L2 = {h:.6e}");
            }
        }
        Err(e) => { eprintln!("erreur de campagne: {e}"); std::process::exit(1); }
    }
}
