use forge_core::{fnv1a, Candidate, CandidateId, Domain, ForgeError, Result, Score, Trial};
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct PackHeuristic { pub w: [f64; 4] }

impl Candidate for PackHeuristic {
    fn id(&self) -> CandidateId { fnv1a(&self.repr()) }
    fn repr(&self) -> String {
        format!("[{:.4}, {:.4}, {:.4}, {:.4}]", self.w[0], self.w[1], self.w[2], self.w[3])
    }
}

pub struct BinPacking { pub capacity: f64, pub n_items: usize, pub n_instances: usize }

fn priority(w: &[f64; 4], item: f64, remaining: f64, cap: f64) -> f64 {
    w[0] * (remaining - item) + w[1] * (item / cap) + w[2] * (remaining / cap)
        + w[3] * (remaining - item).powi(2)
}

fn pack(w: &[f64; 4], items: &[f64], cap: f64) -> Option<usize> {
    let mut remaining: Vec<f64> = Vec::new();
    for &it in items {
        if it > cap { return None; }
        let mut best: Option<(usize, f64)> = None;
        for (i, &rem) in remaining.iter().enumerate() {
            if rem + 1e-9 >= it {
                let p = priority(w, it, rem, cap);
                if !p.is_finite() { return None; }
                if best.is_none_or(|(_, bp)| p > bp) { best = Some((i, p)); }
            }
        }
        match best { Some((i, _)) => remaining[i] -= it, None => remaining.push(cap - it) }
    }
    Some(remaining.len())
}

fn instance(rng: &mut StdRng, n: usize, cap: f64) -> Vec<f64> {
    (0..n).map(|_| rng.gen_range(0.05..=cap * 0.7)).collect()
}

fn mean_bins(w: &[f64; 4], d: &BinPacking, trial: &Trial) -> Option<f64> {
    let mut rng = trial.rng();
    let mut total = 0.0f64;
    for _ in 0..d.n_instances {
        let items = instance(&mut rng, d.n_items, d.capacity);
        total += pack(w, &items, d.capacity)? as f64;
    }
    Some(total / d.n_instances.max(1) as f64)
}

impl Domain for BinPacking {
    type Cand = PackHeuristic;
    fn name(&self) -> &str { "binpack" }
    fn seed(&self, rng: &mut StdRng) -> PackHeuristic {
        PackHeuristic { w: [rng.gen_range(-1.0..1.0); 4] }
    }
    fn mutate(&self, rng: &mut StdRng, parents: &[&PackHeuristic]) -> Result<PackHeuristic> {
        let parent = parents.first().ok_or_else(|| ForgeError::Generation("aucun parent".into()))?;
        let mut w = parent.w;
        for x in w.iter_mut() { *x += rng.gen_range(-0.3..0.3); }
        Ok(PackHeuristic { w })
    }
    fn verify(&self, cand: &PackHeuristic, trial: &Trial) -> Result<bool> {
        Ok(mean_bins(&cand.w, self, trial).is_some())
    }
    fn measure(&self, cand: &PackHeuristic, trial: &Trial) -> Result<Vec<f64>> {
        let bins = mean_bins(&cand.w, self, trial)
            .ok_or_else(|| ForgeError::Evaluation("instance infaisable".into()))?;
        Ok(vec![bins])
    }
    fn objective_names(&self) -> Vec<String> { vec!["avg_bins".into()] }
    fn baseline(&self, trial: &Trial) -> Result<Score> {
        let bins = mean_bins(&[0.0; 4], self, trial)
            .ok_or_else(|| ForgeError::Evaluation("baseline infaisable".into()))?;
        Ok(Score::valid(vec![bins]))
    }
}
