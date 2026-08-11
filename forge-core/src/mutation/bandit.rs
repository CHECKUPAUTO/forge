//! Adaptive mutation strategy selector using the Upper Confidence Bound (UCB1) algorithm.
//!
//! The arms are few-shot objective variants (full / mid / weak). At each mutation
//! the bandit picks an arm via UCB1; after evaluation the caller delivers a reward
//! (relative improvement of the primary objective vs parent). Over time the bandit
//! converges to the arm that consistently produces the best improvements.
//!
//! ## Algorithm: UCB1 (Auer et al. 2002)
//! At round `t`, arm k's score = (mean_reward_k + sqrt(2 * ln(t) / n_k)).
//! The exploration bonus shrinks as an arm is sampled more, shifting the bandit
//! toward exploitation of the best-performing arm.

/// Upper Confidence Bound bandit for selecting mutation strategies.
#[derive(Clone)]
pub struct Bandit {
    arms: Vec<f64>,   // total reward per arm
    pulls: Vec<u64>,  // number of times each arm has been pulled
    exploration: f64, // exploration parameter (default = sqrt(2))
}

impl Bandit {
    /// Create a bandit with the given number of arms.
    pub fn new(n_arms: usize) -> Self {
        assert!(n_arms > 0, "bandit must have at least one arm");
        Bandit {
            arms: vec![0.0; n_arms],
            pulls: vec![0u64; n_arms],
            exploration: 2.0_f64.sqrt(), // standard UCB1 coefficient
        }
    }

    /// Select an arm using the UCB1 policy. All arms are pulled at least once
    /// before any probabilistic selection.
    pub fn pull(&mut self) -> usize {
        // Pull every arm at least once (warm-up phase).
        for k in 0..self.pulls.len() {
            if self.pulls[k] == 0 {
                self.pulls[k] += 1;
                return k;
            }
        }

        let t: u64 = self.pulls.iter().copied().sum();
        let mut best_arm = 0;
        let mut best_ucb = f64::NEG_INFINITY;

        for k in 0..self.arms.len() {
            let mean = self.arms[k] / self.pulls[k] as f64;
            let bonus = ((t as f64).ln() / self.pulls[k] as f64).sqrt();
            let ucb = mean + self.exploration * bonus;
            if ucb > best_ucb {
                best_ucb = ucb;
                best_arm = k;
            }
        }

        self.pulls[best_arm] += 1;
        best_arm
    }

    /// Deliver a reward to the specified arm. Rewards are accumulated and averaged.
    pub fn reward(&mut self, arm: usize, reward: f64) {
        assert!(arm < self.arms.len(), "arm index out of bounds");
        self.arms[arm] += reward;
    }

    /// Return the total number of pulls across all arms.
    pub fn total_pulls(&self) -> u64 {
        self.pulls.iter().sum()
    }

    /// Return the arm with the highest empirical mean reward. Returns 0 if no pulls yet.
    pub fn best_arm(&self) -> usize {
        let mut best = 0;
        let mut best_mean = f64::NEG_INFINITY;
        for k in 0..self.pulls.len() {
            if self.pulls[k] == 0 {
                continue;
            }
            let mean = self.arms[k] / self.pulls[k] as f64;
            if mean > best_mean {
                best_mean = mean;
                best = k;
            }
        }
        best
    }

    /// Return the empirical mean reward per arm (0.0 for unpulled arms).
    pub fn means(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.pulls.len());
        for k in 0..self.pulls.len() {
            if self.pulls[k] > 0 {
                result.push(self.arms[k] / self.pulls[k] as f64);
            } else {
                result.push(0.0);
            }
        }
        result
    }

    /// Reset the bandit state (for testing or re-initialization).
    pub fn reset(&mut self) {
        for arm in &mut self.arms {
            *arm = 0.0;
        }
        for pull in &mut self.pulls {
            *pull = 0;
        }
    }
}

/// Wrapper that manages a set of LlmMutator arms, one per few-shot variant.
#[derive(Clone)]
pub struct MutationBandit {
    base: crate::mutation::llm_mutator::LlmMutator,
    objectives: Vec<String>,
    bandit: Bandit,
}

impl MutationBandit {
    /// Create a new mutation bandit from a base LlmMutator and a list of objectives.
    pub fn new(base: crate::mutation::llm_mutator::LlmMutator, objectives: Vec<String>) -> Self {
        let n = objectives.len();
        MutationBandit {
            base,
            objectives,
            bandit: Bandit::new(n),
        }
    }

    /// Select an arm and mutate with that objective. Returns the mutated code
    /// and records which arm was selected (accessible via best_arm after rewards are delivered).
    pub fn mutate_with_feedback(
        &mut self,
        parent_source: &str,
        diagnostics: Option<&crate::diagnostics::FailureDiagnostics>,
    ) -> String {
        let arm = self.bandit.pull();

        // Clone base mutator and set this arm's objective.
        let mut muttator = self.base.clone();
        muttator = muttator.with_objective(&self.objectives[arm]);

        muttator
            .mutate_with_feedback(parent_source, diagnostics)
            .unwrap_or_else(|_| parent_source.to_string())
    }

    /// Same as mutate_with_feedback but returns the selected arm index.
    pub fn mutate_and_record_arm(
        &mut self,
        parent_source: &str,
        diagnostics: Option<&crate::diagnostics::FailureDiagnostics>,
    ) -> (String, usize) {
        let arm = self.bandit.pull();
        let mut muttator = self.base.clone();
        muttator = muttator.with_objective(&self.objectives[arm]);
        let new_src = muttator
            .mutate_with_feedback(parent_source, diagnostics)
            .unwrap_or_else(|_| parent_source.to_string());
        (new_src, arm)
    }

    /// Deliver a reward to the arm selected by the most recent `mutate_*` call.
    /// The caller must track which arm was selected via a separate mechanism
    /// (see low_rank.rs for the pattern: capture the arm index alongside the result).
    pub fn deliver_reward(&mut self, arm: usize, reward: f64) {
        self.bandit.reward(arm, reward);
    }

    /// Return the index of the best-performing arm so far.
    pub fn best_arm(&self) -> usize {
        self.bandit.best_arm()
    }

    /// Return empirical mean rewards per arm.
    pub fn means(&self) -> Vec<f64> {
        self.bandit.means()
    }

    /// Total mutations attempted across all arms.
    pub fn total_pulls(&self) -> u64 {
        self.bandit.total_pulls()
    }

    /// Return the number of objectives (arms).
    pub fn n_arms(&self) -> usize {
        self.objectives.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    #[test]
    fn test_ucb1_converges_to_best_arm() {
        let mut bandit = Bandit::new(3);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // Arm 1 is strictly better than others: always rewards ~0.8-0.9
        // Arms 0 and 2 are worse: ~0.1-0.2
        let n_rounds = 100;
        for _ in 0..n_rounds {
            let arm = bandit.pull();
            let reward = match arm {
                1 => rng.gen::<f64>() * 0.1 + 0.8, // ~0.85 avg
                _ => rng.gen::<f64>() * 0.1 + 0.1, // ~0.15 avg
            };
            bandit.reward(arm, reward);
        }

        let best = bandit.best_arm();
        assert_eq!(
            best, 1,
            "UCB1 should converge to arm 1 (highest mean), got {best}"
        );
    }

    #[test]
    fn test_ucb1_exploration_phase() {
        let mut bandit = Bandit::new(4);

        let mut pulled: Vec<bool> = vec![false; 4];
        for _ in 0..4 {
            let arm = bandit.pull();
            assert!(!pulled[arm], "arm {} pulled twice during warm-up", arm);
            pulled[arm] = true;
        }
        assert!(pulled.iter().all(|&b| b), "not all arms pulled in warm-up");
    }

    #[test]
    fn test_ucb1_single_arm() {
        let mut bandit = Bandit::new(1);

        for _ in 0..50 {
            let arm = bandit.pull();
            assert_eq!(arm, 0); // only one arm exists
            bandit.reward(arm, 1.0);
        }

        assert_eq!(bandit.best_arm(), 0);
        assert_eq!(bandit.means()[0], 1.0);
    }

    #[test]
    fn test_ucb1_streaking_arm() {
        // Arm 0 dominates with very high probability — should be selected >80% after warm-up.
        let mut bandit = Bandit::new(3);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let n_rounds = 200;
        let arm0_count = {
            let mut count = 0u64;
            for _ in 0..n_rounds {
                let arm = bandit.pull();
                if arm == 0 {
                    count += 1;
                }
                // Arm 0 gives near-perfect rewards; others give near-zero.
                let reward = match arm {
                    0 => rng.gen::<f64>() * 0.05 + 0.9, // ~0.925 avg
                    _ => rng.gen::<f64>() * 0.05,       // ~0.025 avg
                };
                bandit.reward(arm, reward);
            }
            count
        };

        let fraction = arm0_count as f64 / n_rounds as f64;
        assert!(
            fraction > 0.80,
            "Arm 0 should be selected >80% of the time (got {fraction:.2}%), bandit means = {:?}",
            bandit.means()
        );
    }

    #[test]
    fn test_bandit_reset() {
        let mut bandit = Bandit::new(2);

        for _ in 0..10 {
            let arm = bandit.pull();
            bandit.reward(arm, if arm == 0 { 1.0 } else { 0.0 });

            // Before reset, arm 0 should dominate.
            assert_eq!(bandit.best_arm(), 0);
        }

        bandit.reset();
        assert_eq!(bandit.total_pulls(), 0);
        assert_eq!(bandit.means(), vec![0.0, 0.0]);
    }
}
