//! Domaine de compression Tensor Train (TT).
//!
//! Un tenseur de dimensions `[n0, n1, ..., nd]` est décomposé en une chaîne de
//! cœurs `G_k` de dimensions `[r_k, n_k, r_{k+1}]` avec `r_0 = r_{d+1} = 1`.
//! Les rangs `r_1..r_d` contrôlent le compromis compression / précision.
//!
//! ## Paramètres du candidat
//! Chaque candidat est un vecteur de rangs `[r1, r2, ..., rd]` bornés entre
//! `min_rank` et `max_rank`.
//!
//! ## Objectifs (minimisation)
//! 1. Erreur de reconstruction relative : plus c'est bas, mieux c'est.
//! 2. Coût de stockage normalisé : ratio stockage_TT / stockage_dense.
//!    Plus c'est bas, plus la compression est forte.

use forge_core::{fnv1a, Candidate, CandidateId, Domain, Result, Score, Trial};
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration du problème Tensor Train.
#[derive(Clone, Debug)]
pub struct TtConfig {
    /// Dimensions du tenseur original (ex: [256, 256, 64]).
    pub dims: Vec<usize>,
    /// Rang minimal autorisé pour chaque lien.
    pub min_rank: usize,
    /// Rang maximal autorisé.
    pub max_rank: usize,
    /// Nombre de rangs à optimiser (= dims.len() - 1).
    pub num_ranks: usize,
}

impl TtConfig {
    pub fn new(dims: Vec<usize>, min_rank: usize, max_rank: usize) -> Self {
        let num_ranks = dims.len().saturating_sub(1);
        TtConfig {
            dims,
            min_rank,
            max_rank,
            num_ranks,
        }
    }

    /// Nombre d'éléments du tenseur dense.
    pub fn dense_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Nombre d'éléments du format TT pour des rangs donnés.
    pub fn tt_elements(&self, ranks: &[usize]) -> usize {
        let d = self.num_ranks;
        if d == 0 {
            return self.dims[0];
        }
        let mut total = 0usize;
        // Cœur G_0 : r0=1, n0, r1
        total += self.dims[0] * ranks[0];
        // Cœurs G_k pour k=1..d-1
        for k in 1..d {
            total += ranks[k - 1] * self.dims[k] * ranks[k];
        }
        // Cœur G_d : r_d, n_d, r_{d+1}=1
        if d > 0 {
            total += ranks[d - 1] * self.dims[d];
        }
        total
    }

    /// Estime l'erreur de reconstruction en fonction des rangs.
    /// Modèle simplifié : erreur ∝ 1 / rang_moyen (loi de décroissance
    /// exponentielle typique des décompositions SVD/TT).
    pub fn estimate_error(&self, ranks: &[usize]) -> f64 {
        if ranks.is_empty() {
            return 0.0;
        }
        let avg_rank = ranks.iter().sum::<usize>() as f64 / ranks.len() as f64;
        let max_possible = self.max_rank as f64;
        // Erreur normalisée : plus les rangs sont proches du max, plus l'erreur est faible
        (max_possible - avg_rank) / (max_possible - self.min_rank as f64).max(1.0)
    }
}

/// Candidat : un vecteur de rangs TT.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TtCandidate {
    pub ranks: Vec<usize>,
}

impl TtCandidate {
    pub fn new(ranks: Vec<usize>) -> Self {
        TtCandidate { ranks }
    }
}

impl Candidate for TtCandidate {
    fn id(&self) -> CandidateId {
        fnv1a(&self.repr())
    }

    fn repr(&self) -> String {
        format!(
            "[{}]",
            self.ranks
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Domaine Tensor Train.
pub struct TensorTrainDomain {
    pub config: TtConfig,
}

impl TensorTrainDomain {
    pub fn new(config: TtConfig) -> Self {
        TensorTrainDomain { config }
    }
}

impl Domain for TensorTrainDomain {
    type Cand = TtCandidate;

    fn name(&self) -> &str {
        "tensor_train_compression"
    }

    fn seed(&self, rng: &mut StdRng) -> Self::Cand {
        let ranks: Vec<usize> = (0..self.config.num_ranks)
            .map(|_| rng.gen_range(self.config.min_rank..=self.config.max_rank))
            .collect();
        TtCandidate::new(ranks)
    }

    fn verify(&self, cand: &Self::Cand, _trial: &Trial) -> Result<bool> {
        // Vérification structurelle : les rangs doivent être dans les bornes
        if cand.ranks.len() != self.config.num_ranks {
            return Ok(false);
        }
        for &r in &cand.ranks {
            if r < self.config.min_rank || r > self.config.max_rank {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn measure(&self, cand: &Self::Cand, _trial: &Trial) -> Result<Vec<f64>> {
        let tt_elems = self.config.tt_elements(&cand.ranks) as f64;
        let dense_elems = self.config.dense_elements() as f64;

        // Objectif 1 : erreur de reconstruction (minimisation)
        let error = self.config.estimate_error(&cand.ranks);

        // Objectif 2 : ratio de stockage TT / dense (minimisation)
        let storage_ratio = tt_elems / dense_elems.max(1.0);

        Ok(vec![error, storage_ratio])
    }

    fn mutate(&self, rng: &mut StdRng, parents: &[&Self::Cand]) -> Result<Self::Cand> {
        let _ = parents; // pour l'instant, mutation aléatoire simple
        let mut ranks: Vec<usize> = (0..self.config.num_ranks)
            .map(|_| rng.gen_range(self.config.min_rank..=self.config.max_rank))
            .collect();

        // Avec 30% de chance, on garde un parent et on perturbe légèrement
        if !parents.is_empty() && rng.gen_bool(0.3) {
            let parent_idx = rng.gen_range(0..parents.len());
            ranks = parents[parent_idx].ranks.clone();
            // Perturber un rang aléatoire
            let perturb_idx = rng.gen_range(0..ranks.len());
            let delta: i64 = if rng.gen_bool(0.5) { 1 } else { -1 };
            let new_rank = (ranks[perturb_idx] as i64 + delta)
                .clamp(self.config.min_rank as i64, self.config.max_rank as i64);
            ranks[perturb_idx] = new_rank as usize;
        }

        Ok(TtCandidate::new(ranks))
    }

    fn objective_names(&self) -> Vec<String> {
        vec![
            "erreur_reconstruction".into(),
            "ratio_stockage_TT_dense".into(),
        ]
    }

    fn baseline(&self, _trial: &Trial) -> Result<Score> {
        // Baseline : tous les rangs au maximum (aucune compression)
        let ranks: Vec<usize> = vec![self.config.max_rank; self.config.num_ranks];
        let error = self.config.estimate_error(&ranks);
        let tt_elems = self.config.tt_elements(&ranks) as f64;
        let dense_elems = self.config.dense_elements() as f64;
        let storage_ratio = tt_elems / dense_elems.max(1.0);
        Ok(Score::valid(vec![error, storage_ratio]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_tt_elements_small_3d() {
        let config = TtConfig::new(vec![4, 5, 6], 1, 4);
        // ranks = [2, 2]: G0(1x4x2) + G1(2x5x2) + G2(2x6x1)
        // = 8 + 20 + 12 = 40
        assert_eq!(config.tt_elements(&[2, 2]), 40);
        // dense = 4*5*6 = 120
        assert_eq!(config.dense_elements(), 120);
    }

    #[test]
    fn test_tt_elements_1d() {
        let config = TtConfig::new(vec![10], 1, 4);
        // 0 ranks, just dim
        assert_eq!(config.tt_elements(&[]), 10);
    }

    #[test]
    fn test_estimate_error_monotonic() {
        let config = TtConfig::new(vec![16, 32, 8], 1, 8);
        // Plus les rangs sont hauts, plus l'erreur est basse
        let err_low = config.estimate_error(&[1, 1]);
        let err_high = config.estimate_error(&[8, 8]);
        assert!(err_low > err_high);
    }

    #[test]
    fn test_domain_seed_respects_bounds() {
        let config = TtConfig::new(vec![4, 5, 6], 1, 4);
        let domain = TensorTrainDomain::new(config);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..100 {
            let cand = domain.seed(&mut rng);
            assert_eq!(cand.ranks.len(), 2);
            for &r in &cand.ranks {
                assert!((1..=4).contains(&r));
            }
        }
    }

    #[test]
    fn test_domain_verify_rejects_wrong_size() {
        let config = TtConfig::new(vec![4, 5, 6], 1, 4);
        let domain = TensorTrainDomain::new(config);
        let trial = Trial {
            generation: 0,
            seed: 42,
        };
        // Wrong number of ranks
        assert!(!domain.verify(&TtCandidate::new(vec![1]), &trial).unwrap());
        assert!(!domain
            .verify(&TtCandidate::new(vec![1, 2, 3]), &trial)
            .unwrap());
    }

    #[test]
    fn test_domain_measure_two_objectives() {
        let config = TtConfig::new(vec![16, 32, 8], 1, 8);
        let domain = TensorTrainDomain::new(config);
        let trial = Trial {
            generation: 0,
            seed: 42,
        };
        let objectives = domain
            .measure(&TtCandidate::new(vec![4, 4]), &trial)
            .unwrap();
        assert_eq!(objectives.len(), 2);
        // Both should be finite
        assert!(objectives[0].is_finite());
        assert!(objectives[1].is_finite());
        // Storage ratio should be between 0 and 1
        assert!(objectives[1] > 0.0 && objectives[1] <= 1.0);
    }

    #[test]
    fn test_full_campaign_runs() {
        use forge_core::{Config, Engine};
        let config = TtConfig::new(vec![16, 32, 8, 4], 1, 8);
        let domain = TensorTrainDomain::new(config);
        let engine = Engine::new(
            domain,
            Config {
                generations: 5,
                population: 20,
                survivors: 5,
                base_seed: 123,
                worker_addresses: None,
            },
        );
        let report = engine.run().unwrap();
        assert_eq!(report.history.len(), 5);
        assert!(report.best.is_some());
    }
}
