use forge_core::domains::simd_kernel::SimdKernelDomain;
use forge_core::{Candidate, Config, Engine};

fn main() {
    let endpoint = std::env::var("OLLAMA_URL")
        .unwrap_or_else(|_| "http://localhost:11434/api/generate".to_string());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5-coder:1.5b".to_string());
    println!("== Campagne simd_gemm :: Ollama {model} @ {endpoint} ==");

    let domain = SimdKernelDomain::new("/tmp/forge_campaign_simd").with_llm(&endpoint, &model);
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
                println!("  gen {g:>2}  meilleure latency_ns = {h:.0}");
            }
            println!("\n--- front de Pareto final ({} candidats) ---", report.final_front.len());
            for (i, ind) in report.final_front.iter().enumerate() {
                let lat = ind.score.objectives.get(0).copied().unwrap_or(f64::NAN);
                let hold = report.final_front_holdout.get(i).and_then(|o| o.as_ref())
                    .map(|s| if s.valid { format!("{:.0}", s.objectives.get(0).copied().unwrap_or(f64::NAN)) } else { "INVALIDE".to_string() })
                    .unwrap_or_else(|| "?".to_string());
                println!("  [{i}] latency_ns={lat:.0}  holdout={hold}");
            }
            if let Some(bl) = report.final_baseline.as_ref() {
                println!("  baseline  latency_ns={:.0}", bl.objectives.get(0).copied().unwrap_or(f64::NAN));
            }
            let baseline_lat = report.final_baseline.as_ref()
                .and_then(|b| b.objectives.get(0).copied()).unwrap_or(f64::INFINITY);
            let elite = report.final_front.iter().enumerate()
                .filter(|(i, _)| {
                    report.final_front_holdout.get(*i).and_then(|o| o.as_ref()).map(|s| s.valid).unwrap_or(false)
                })
                .min_by(|(_, a), (_, b)| {
                    let la = a.score.objectives.get(0).copied().unwrap_or(f64::INFINITY);
                    let lb = b.score.objectives.get(0).copied().unwrap_or(f64::INFINITY);
                    la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
                });
            match elite {
                Some((_, e)) if e.score.objectives.get(0).copied().unwrap_or(f64::INFINITY) < baseline_lat => {
                    use std::fmt::Write as _;
                    let lat = e.score.objectives.get(0).copied().unwrap_or(f64::NAN);
                    let speedup = baseline_lat / lat;
                    let src = e.cand.repr();
                    let dir = std::path::Path::new("/tmp/forge_elite_simd");
                    let _ = std::fs::create_dir_all(dir);
                    let _ = std::fs::write(dir.join("kernel.rs"), &src);
                    let mut manifest = String::new();
                    let _ = writeln!(manifest, "model = {model}");
                    let _ = writeln!(manifest, "latency_ns = {lat:.0}");
                    let _ = writeln!(manifest, "baseline_ns = {baseline_lat:.0}");
                    let _ = writeln!(manifest, "speedup = {speedup:.2}x");
                    let _ = writeln!(manifest, "bytes = {}", src.len());
                    let _ = writeln!(manifest, "verified_holdout = true");
                    let _ = std::fs::write(dir.join("manifest.txt"), &manifest);
                    println!();
                    println!(">>> ELITE (holdout-verifie) -> /tmp/forge_elite_simd/kernel.rs");
                    println!("    {lat:.0} ns vs {baseline_lat:.0} baseline ({speedup:.2}x plus rapide), {} octets", src.len());
                }
                _ => {
                    println!();
                    println!(">>> aucun elite holdout-verifie ne bat le baseline (latency_ns={baseline_lat:.0})");
                }
            }
        }
        Err(e) => { eprintln!("erreur de campagne: {e}"); std::process::exit(1); }
    }
}
