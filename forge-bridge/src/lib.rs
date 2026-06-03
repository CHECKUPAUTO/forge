//! Pont Forge (moteur évolutionnaire) ↔ SoulSystem.
//!
//! Expose les types de `forge-core` (Engine, Config, Domain, Score) en tant
//! qu'API typée consommable par les autres briques du système (synergie,
//! openevolve, mesh). Le transport HTTP (port 7890) vit dans le binaire
//! `forge-service` ; ce crate fournit la **vue fonctionnelle** du moteur.

use forge_core::{Candidate, Config, Domain, Score};
use serde::{Deserialize, Serialize};
use tracing::info;

pub mod binpack_demo;
pub mod llm_ollama;

pub type ForgeConfig = Config;

pub struct ForgeCampaign<D: Domain>
where
    D::Cand: Serialize + for<'a> Deserialize<'a>,
{
    pub config: ForgeConfig,
    pub domain: D,
}

impl<D: Domain> ForgeCampaign<D>
where
    D::Cand: Serialize + for<'a> Deserialize<'a>,
{
    pub fn new(config: ForgeConfig, domain: D) -> Self {
        Self { config, domain }
    }

    pub fn run(self) -> forge_core::Report<D::Cand> {
        info!(target: "forge-bridge", "lancement campagne domaine={}", self.domain.name());
        let engine = forge_core::Engine::new(self.domain, self.config);
        match engine.run() {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(target: "forge-bridge", "campagne en echec: {e}");
                forge_core::Report {
                    best: None,
                    final_baseline: None,
                    holdout_best: None,
                    holdout_baseline: None,
                    history: Vec::new(),
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ScoreDto { pub objectives: Vec<f64>, pub valid: bool }

impl From<Score> for ScoreDto {
    fn from(s: Score) -> Self { ScoreDto { objectives: s.objectives, valid: s.valid } }
}

#[derive(Serialize, Deserialize)]
pub struct CandidateDto { pub id: u64, pub repr: String }

impl<T: Candidate> From<&T> for CandidateDto {
    fn from(c: &T) -> Self { CandidateDto { id: c.id(), repr: c.repr() } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binpack_demo::BinPacking;

    #[test]
    fn forge_bridge_compiles_and_runs_binpack() {
        let cfg = ForgeConfig { generations: 3, population: 8, survivors: 2, base_seed: 1 };
        let domain = BinPacking { capacity: 1.0, n_items: 20, n_instances: 5 };
        let report = ForgeCampaign::new(cfg, domain).run();
        assert!(!report.history.is_empty(), "doit produire un historique");
    }
}
