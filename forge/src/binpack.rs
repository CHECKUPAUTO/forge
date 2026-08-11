//! Domaine de reference : bin-packing 1D, facon FunSearch.
//!
//! Il sert uniquement a PROUVER que la boucle du moteur tourne, ameliore une
//! vraie metrique et generalise sur un holdout — le tout sans scirust ni LLM,
//! donc executable ici. Les 4 vrais domaines (compression, quantification,
//! kernels, routage MoE) implementeront le meme trait `Domain` ; seule la
//! nature du candidat change (code Rust genere) et `measure` deviendra une
//! compilation + un bench Criterion sur le Thor.
//!
//! Candidat : un vecteur de 4 poids parametrant une fonction de priorite qui
//! choisit, pour chaque objet, dans quel bac le placer. La baseline est le
//! "first-fit" (tous poids a zero). L'evolution peut decouvrir un comportement
//! de type best-fit (preferer le bac le plus rempli ou l'objet rentre) et
//! ainsi reduire le nombre de bacs.

use forge_core::{fnv1a, Candidate, CandidateId, Domain, ForgeError, Result, Score, Trial};
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Heuristique de placement : 4 poids reels.
#[derive(Clone, Serialize, Deserialize)]
pub struct PackHeuristic {
    pub w: [f64; 4],
}

impl Candidate for PackHeuristic {
    fn id(&self) -> CandidateId {
        fnv1a(&self.repr())
    }
    fn repr(&self) -> String {
        format!(
            "[{:.4}, {:.4}, {:.4}, {:.4}]",
            self.w[0], self.w[1], self.w[2], self.w[3]
        )
    }
}

/// Le domaine bin-packing.
pub struct BinPacking {
    /// Capacite d'un bac.
    pub capacity: f64,
    /// Nombre d'objets par instance.
    pub n_items: usize,
    /// Nombre d'instances moyennees par essai (reduit la variance).
    pub n_instances: usize,
}

/// Priorite (plus grand = preferer ce bac) de placer `item` dans un bac dont la
/// capacite restante est `remaining`.
fn priority(w: &[f64; 4], item: f64, remaining: f64, cap: f64) -> f64 {
    w[0] * (remaining - item)
        + w[1] * (item / cap)
        + w[2] * (remaining / cap)
        + w[3] * (remaining - item).powi(2)
}

/// Greedy parametre par les poids. Renvoie le nombre de bacs, ou `None` si une
/// instance est infaisable (objet plus grand que la capacite, ou priorite NaN).
fn pack(w: &[f64; 4], items: &[f64], cap: f64) -> Option<usize> {
    let mut remaining: Vec<f64> = Vec::new();
    for &it in items {
        if it > cap {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        for (i, &rem) in remaining.iter().enumerate() {
            if rem + 1e-9 >= it {
                let p = priority(w, it, rem, cap);
                if !p.is_finite() {
                    return None;
                }
                if best.is_none_or(|(_, bp)| p > bp) {
                    best = Some((i, p));
                }
            }
        }
        match best {
            Some((i, _)) => remaining[i] -= it,
            None => remaining.push(cap - it),
        }
    }
    Some(remaining.len())
}

/// Genere une instance aleatoire d'objets dans (0, 0.7*cap].
fn instance(rng: &mut StdRng, n: usize, cap: f64) -> Vec<f64> {
    (0..n).map(|_| rng.gen_range(0.05..=cap * 0.7)).collect()
}

/// Moyenne du nombre de bacs sur les instances de l'essai.
fn mean_bins(w: &[f64; 4], domain: &BinPacking, trial: &Trial) -> Option<f64> {
    let mut rng = trial.rng();
    let mut total = 0.0f64;
    for _ in 0..domain.n_instances {
        let items = instance(&mut rng, domain.n_items, domain.capacity);
        total += pack(w, &items, domain.capacity)? as f64;
    }
    Some(total / domain.n_instances.max(1) as f64)
}

impl Domain for BinPacking {
    type Cand = PackHeuristic;

    fn name(&self) -> &str {
        "binpack"
    }

    fn seed(&self, rng: &mut StdRng) -> PackHeuristic {
        PackHeuristic {
            w: [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ],
        }
    }

    fn mutate(&self, rng: &mut StdRng, parents: &[&PackHeuristic]) -> Result<PackHeuristic> {
        let parent = parents
            .first()
            .ok_or_else(|| ForgeError::Generation("aucun parent".into()))?;
        let mut w = parent.w;
        for x in w.iter_mut() {
            *x += rng.gen_range(-0.3..0.3);
        }
        Ok(PackHeuristic { w })
    }

    fn verify(&self, cand: &PackHeuristic, trial: &Trial) -> Result<bool> {
        // Porte de correction : sur des instances randomisees, le placement
        // doit etre faisable. (Le greedy est correct par construction ; la
        // porte reste demonstree et rejetterait un candidat produisant NaN.)
        Ok(mean_bins(&cand.w, self, trial).is_some())
    }

    fn measure(&self, cand: &PackHeuristic, trial: &Trial) -> Result<Vec<f64>> {
        let bins = mean_bins(&cand.w, self, trial)
            .ok_or_else(|| ForgeError::Evaluation("instance infaisable".into()))?;
        Ok(vec![bins]) // objectif unique : nombre moyen de bacs (a minimiser)
    }

    fn objective_names(&self) -> Vec<String> {
        vec!["avg_bins".into()]
    }

    fn baseline(&self, trial: &Trial) -> Result<Score> {
        // Poids nuls => priorite constante => premier bac faisable = first-fit.
        let ff = [0.0; 4];
        let bins = mean_bins(&ff, self, trial)
            .ok_or_else(|| ForgeError::Evaluation("baseline infaisable".into()))?;
        Ok(Score::valid(vec![bins]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn packing_never_overflows_capacity() {
        let mut rng = StdRng::seed_from_u64(7);
        let cap = 1.0;
        let items = instance(&mut rng, 40, cap);
        // On rejoue le greedy en verifiant chaque bac.
        let w = [-1.0, 0.0, 0.0, 0.0];
        let mut remaining: Vec<f64> = Vec::new();
        for &it in &items {
            let mut best: Option<(usize, f64)> = None;
            for (i, &rem) in remaining.iter().enumerate() {
                if rem + 1e-9 >= it {
                    let p = priority(&w, it, rem, cap);
                    if best.is_none_or(|(_, bp)| p > bp) {
                        best = Some((i, p));
                    }
                }
            }
            match best {
                Some((i, _)) => remaining[i] -= it,
                None => remaining.push(cap - it),
            }
        }
        for &rem in &remaining {
            assert!(rem >= -1e-9, "bac en depassement: {rem}");
        }
    }
}
