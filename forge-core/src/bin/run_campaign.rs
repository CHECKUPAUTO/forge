use forge_core::domains::low_rank::TensorTrainDomain;
use forge_core::{Candidate, Config, Engine};

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
            const L2_TOL: f64 = 1e-6;
            let baseline_params = report.final_baseline.as_ref()
                .and_then(|b| b.objectives.get(2).copied()).unwrap_or(f64::INFINITY);
            let elite = report.final_front.iter().enumerate()
                .filter(|(i, ind)| {
                    let train_l2 = ind.score.objectives.get(0).copied().unwrap_or(f64::INFINITY);
                    let hold_ok = report.final_front_holdout.get(*i)
                        .and_then(|o| o.as_ref())
                        .map(|sc| sc.valid && sc.objectives.get(0).copied().unwrap_or(f64::INFINITY) <= L2_TOL)
                        .unwrap_or(false);
                    train_l2 <= L2_TOL && hold_ok
                })
                .min_by(|(_, a), (_, b)| {
                    let pa = a.score.objectives.get(2).copied().unwrap_or(f64::INFINITY);
                    let pb = b.score.objectives.get(2).copied().unwrap_or(f64::INFINITY);
                    pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
                });
            match elite {
                Some((_, e)) if e.score.objectives.get(2).copied().unwrap_or(f64::INFINITY) < baseline_params => {
                    use std::fmt::Write as _;
                    let params = e.score.objectives.get(2).copied().unwrap_or(f64::NAN);
                    let l2 = e.score.objectives.get(0).copied().unwrap_or(f64::NAN);
                    let ratio = baseline_params / params;
                    let src = e.cand.repr();
                    let dir = std::path::Path::new("/tmp/forge_elite");
                    let _ = std::fs::create_dir_all(dir);
                    let _ = std::fs::write(dir.join("elite_compressor.rs"), &src);
                    let mut manifest = String::new();
                    let _ = writeln!(manifest, "model = {model}");
                    let _ = writeln!(manifest, "params = {params:.0}");
                    let _ = writeln!(manifest, "baseline_params = {baseline_params:.0}");
                    let _ = writeln!(manifest, "ratio = {ratio:.2}x");
                    let _ = writeln!(manifest, "L2_train = {l2:.3e}");
                    let _ = writeln!(manifest, "bytes = {}", src.len());
                    let _ = writeln!(manifest, "verified_holdout = true");
                    let _ = std::fs::write(dir.join("manifest.txt"), &manifest);
                    println!();
                    println!(">>> ELITE (holdout-verifie) -> /tmp/forge_elite/elite_compressor.rs");
                    println!("    {params:.0} params vs {baseline_params:.0} baseline ({ratio:.2}x), L2={l2:.3e}, {} octets", src.len());
                }
                _ => {
                    println!();
                    println!(">>> aucun elite holdout-verifie ne bat le baseline (params={baseline_params:.0})");
                }
            }
        }
        Err(e) => { eprintln!("erreur de campagne: {e}"); std::process::exit(1); }
    }
}
