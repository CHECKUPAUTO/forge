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

use std::collections::HashMap;

use crate::candidate::CandidateId;
use crate::domain::Score;

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
            exploration: 2.0_f64.sqrt(),
        }
    }

    /// Select an arm using the UCB1 policy. All arms are pulled at least once
    /// before exploitation begins.
    pub fn pull(&mut self) -> usize {
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
        if reward.is_finite() {
            self.arms[arm] += reward;
        }
    }

    pub fn total_pulls(&self) -> u64 {
        self.pulls.iter().sum()
    }

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

    pub fn means(&self) -> Vec<f64> {
        self.pulls
            .iter()
            .enumerate()
            .map(|(k, &pulls)| {
                if pulls > 0 {
                    self.arms[k] / pulls as f64
                } else {
                    0.0
                }
            })
            .collect()
    }

    pub fn reset(&mut self) {
        self.arms.fill(0.0);
        self.pulls.fill(0);
    }
}

/// Wrapper that manages a set of LlmMutator arms, one per few-shot variant.
/// `pending_arms` ties an arm choice to the concrete candidate produced by that
/// mutation so concurrent mutations cannot overwrite each other's attribution.
#[derive(Clone)]
pub struct MutationBandit {
    base: crate::mutation::llm_mutator::LlmMutator,
    objectives: Vec<String>,
    bandit: Bandit,
    pending_arms: HashMap<CandidateId, usize>,
}

impl MutationBandit {
    pub fn new(base: crate::mutation::llm_mutator::LlmMutator, objectives: Vec<String>) -> Self {
        let n = objectives.len();
        MutationBandit {
            base,
            objectives,
            bandit: Bandit::new(n),
            pending_arms: HashMap::new(),
        }
    }

    pub fn mutate_with_feedback(
        &mut self,
        parent_source: &str,
        diagnostics: Option<&crate::diagnostics::FailureDiagnostics>,
    ) -> String {
        self.mutate_and_record_arm(parent_source, diagnostics).0
    }

    pub fn mutate_and_record_arm(
        &mut self,
        parent_source: &str,
        diagnostics: Option<&crate::diagnostics::FailureDiagnostics>,
    ) -> (String, usize) {
        let arm = self.bandit.pull();
        let mut mutator = self.base.clone();
        mutator = mutator.with_objective(&self.objectives[arm]);
        let new_src = mutator
            .mutate_with_feedback(parent_source, diagnostics)
            .unwrap_or_else(|_| parent_source.to_string());
        (new_src, arm)
    }

    /// Associate the selected arm with the concrete candidate it produced.
    pub fn track_candidate_arm(&mut self, candidate_id: CandidateId, arm: usize) {
        if arm < self.objectives.len() {
            self.pending_arms.insert(candidate_id, arm);
        }
    }

    /// Consume the pending arm for a candidate and deliver its reward exactly once.
    pub fn deliver_reward_for_candidate(
        &mut self,
        candidate_id: CandidateId,
        reward: f64,
    ) -> bool {
        let Some(arm) = self.pending_arms.remove(&candidate_id) else {
            return false;
        };
        self.bandit.reward(arm, reward.clamp(-1.0, 1.0));
        true
    }

    /// Reward for a minimization objective. Invalid children receive -1.0;
    /// otherwise the reward is the relative improvement over the parent,
    /// clamped to [-1, 1]. Missing or invalid parent data yields no reward.
    pub fn minimization_reward(
        parent: &Score,
        child: &Score,
        objective_idx: usize,
    ) -> Option<f64> {
        if !parent.valid {
            return None;
        }
        if !child.valid {
            return Some(-1.0);
        }
        let parent_value = *parent.objectives.get(objective_idx)?;
        let child_value = *child.objectives.get(objective_idx)?;
        if !parent_value.is_finite() || !child_value.is_finite() {
            return None;
        }
        let denom = parent_value.abs().max(1.0e-12);
        Some(((parent_value - child_value) / denom).clamp(-1.0, 1.0))
    }

    pub fn deliver_reward(&mut self, arm: usize, reward: f64) {
        self.bandit.reward(arm, reward.clamp(-1.0, 1.0));
    }

    pub fn best_arm(&self) -> usize {
        self.bandit.best_arm()
    }

    pub fn means(&self) -> Vec<f64> {
        self.bandit.means()
    }

    pub fn total_pulls(&self) -> u64 {
        self.bandit.total_pulls()
    }

    pub fn n_arms(&self) -> usize {
        self.objectives.len()
    }

    pub fn pending_len(&self) -> usize {
        self.pending_arms.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    #[test]
    fn test_ucb1_converges_to_best_arm() {
        let mut bandit = Bandit::new(3);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let arm = bandit.pull();
            let reward = match arm {
                1 => rng.gen::<f64>() * 0.1 + 0.8,
                _ => rng.gen::<f64>() * 0.1 + 0.1,
            };
            bandit.reward(arm, reward);
        }
        assert_eq!(bandit.best_arm(), 1);
    }

    #[test]
    fn test_ucb1_exploration_phase() {
        let mut bandit = Bandit::new(4);
        let mut pulled = [false; 4];
        for _ in 0..4 {
            let arm = bandit.pull();
            assert!(!pulled[arm]);
            pulled[arm] = true;
        }
        assert!(pulled.iter().all(|&v| v));
    }

    #[test]
    fn test_ucb1_single_arm() {
        let mut bandit = Bandit::new(1);
        for _ in 0..50 {
            let arm = bandit.pull();
            assert_eq!(arm, 0);
            bandit.reward(arm, 1.0);
        }
        assert_eq!(bandit.best_arm(), 0);
        assert_eq!(bandit.means()[0], 1.0);
    }

    #[test]
    fn test_ucb1_streaking_arm() {
        let mut bandit = Bandit::new(3);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut arm0_count = 0u64;
        for _ in 0..200 {
            let arm = bandit.pull();
            if arm == 0 {
                arm0_count += 1;
            }
            let reward = match arm {
                0 => rng.gen::<f64>() * 0.05 + 0.9,
                _ => rng.gen::<f64>() * 0.05,
            };
            bandit.reward(arm, reward);
        }
        assert!(arm0_count as f64 / 200.0 > 0.80);
    }

    #[test]
    fn test_bandit_reset() {
        let mut bandit = Bandit::new(2);
        for _ in 0..10 {
            let arm = bandit.pull();
            bandit.reward(arm, if arm == 0 { 1.0 } else { 0.0 });
        }
        bandit.reset();
        assert_eq!(bandit.total_pulls(), 0);
        assert_eq!(bandit.means(), vec![0.0, 0.0]);
    }

    #[test]
    fn reward_is_attributed_to_candidate_once() {
        let base = crate::mutation::llm_mutator::LlmMutator::new("http://invalid", "test");
        let mut mutation = MutationBandit::new(base, vec!["a".into(), "b".into()]);
        let arm = mutation.bandit.pull();
        mutation.track_candidate_arm(42, arm);
        assert_eq!(mutation.pending_len(), 1);
        assert!(mutation.deliver_reward_for_candidate(42, 0.5));
        assert!(!mutation.deliver_reward_for_candidate(42, 0.5));
        assert_eq!(mutation.pending_len(), 0);
        assert!((mutation.means()[arm] - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn minimization_reward_handles_invalid_and_improvement() {
        let parent = Score::valid(vec![100.0]);
        let child = Score::valid(vec![80.0]);
        assert_eq!(
            MutationBandit::minimization_reward(&parent, &child, 0),
            Some(0.2)
        );
        assert_eq!(
            MutationBandit::minimization_reward(&parent, &Score::invalid(), 0),
            Some(-1.0)
        );
    }
}
