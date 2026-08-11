from pathlib import Path


def replace(path, old, new, count=None):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f"expected pattern missing in {path}: {old[:120]!r}")
    text = text.replace(old, new) if count is None else text.replace(old, new, count)
    p.write_text(text)


# Engine: remember child -> parent identity, then compare both scores only after
# they have been measured on the same rotating Trial.
p = Path("forge-core/src/evolve.rs")
text = p.read_text()
text = text.replace(
    "use std::collections::HashSet;",
    "use std::collections::{HashMap, HashSet};",
    1,
)
marker = "        let start_gen = self.start_generation;\n"
insert = """        // Lignage temporaire d'une génération pour attribuer le feedback
        // mutation au bon parent, sans comparer des scores de trials différents.
        let mut parent_by_child: HashMap<CandidateId, CandidateId> = HashMap::new();

"""
if marker not in text:
    raise SystemExit("engine generation marker missing")
text = text.replace(marker, insert + marker, 1)

old = """            let mut valids: Vec<Individual<D::Cand>> = evaluated
                .into_iter()
                .filter(|ind| ind.score.valid)
                .collect();
"""
new = """            // Le parent survivant est lui aussi évalué dans cette population :
            // le reward compare donc enfant et parent sur exactement le même Trial.
            let scores_by_id: HashMap<CandidateId, Score> = evaluated
                .iter()
                .map(|ind| (ind.cand.id(), ind.score.clone()))
                .collect();
            for ind in &evaluated {
                let parent_score = parent_by_child
                    .get(&ind.cand.id())
                    .and_then(|parent_id| scores_by_id.get(parent_id));
                self.domain
                    .observe_evaluation(&ind.cand, &ind.score, parent_score);
            }
            parent_by_child.clear();

            let mut valids: Vec<Individual<D::Cand>> = evaluated
                .into_iter()
                .filter(|ind| ind.score.valid)
                .collect();
"""
if old not in text:
    raise SystemExit("engine evaluated block missing")
text = text.replace(old, new, 1)

old = """                while next.len() < self.config.population {
                    let parent = &survivors[reproduction_rng.gen_range(0..survivors.len())];
                    next.push(self.domain.mutate(&mut reproduction_rng, &[parent])?);
                }
"""
new = """                while next.len() < self.config.population {
                    let parent = &survivors[reproduction_rng.gen_range(0..survivors.len())];
                    let child = self.domain.mutate(&mut reproduction_rng, &[parent])?;
                    if child.id() != parent.id() {
                        parent_by_child.insert(child.id(), parent.id());
                    }
                    next.push(child);
                }
"""
if old not in text:
    raise SystemExit("engine reproduction block missing")
text = text.replace(old, new, 1)
p.write_text(text)


def wire_domain(path, reward_index=None):
    p = Path(path)
    text = p.read_text()
    text = text.replace(
        'std::env::var("FORGE_MAB").is_ok()',
        'matches!(std::env::var("FORGE_MAB").as_deref(), Ok("1"))',
    )
    if reward_index is not None:
        old_idx = "reward_objective_idx: 0, // default: first objective"
        if old_idx not in text:
            raise SystemExit(f"reward index pattern missing in {path}")
        text = text.replace(
            old_idx,
            f"reward_objective_idx: {reward_index}, // compression objective: parameters_count",
            1,
        )

    old = """                    std::env::set_var("FORGE_BANDIT_ARM", arm.to_string());

                    let id = crate::fnv1a(&new_src);
"""
    new = """                    let id = crate::fnv1a(&new_src);
                    if parents.first().map(|p| p.id) != Some(id) {
                        bandit.track_candidate_arm(id, arm);
                    }
"""
    if old not in text:
        raise SystemExit(f"bandit env attribution pattern missing in {path}")
    text = text.replace(old, new, 1)

    verify_marker = "    /// Porte de correction"
    if path.endswith("low_rank.rs"):
        verify_marker = "    /// PORTE DE CORRECTION"
    if verify_marker not in text:
        raise SystemExit(f"verify marker missing in {path}")
    observe = """    fn observe_evaluation(
        &self,
        _cand: &Self::Cand,
        _score: &Score,
        _parent_score: Option<&Score>,
    ) {
        #[cfg(feature = "bandit")]
        if matches!(std::env::var("FORGE_MAB").as_deref(), Ok("1")) {
            let Some(parent_score) = _parent_score else {
                return;
            };
            let Some(reward) = crate::mutation::bandit::MutationBandit::minimization_reward(
                parent_score,
                _score,
                self.reward_objective_idx,
            ) else {
                return;
            };
            if let Ok(mut bandit) = self.bandit.lock() {
                let delivered = bandit.deliver_reward_for_candidate(_cand.id, reward);
                if delivered && std::env::var("FORGE_VERBOSE").is_ok() {
                    eprintln!(
                        "[forge:bandit] candidate={} reward={:+.4} best_arm={} means={:?}",
                        _cand.id,
                        reward,
                        bandit.best_arm(),
                        bandit.means()
                    );
                }
            }
        }
    }

"""
    text = text.replace(verify_marker, observe + verify_marker, 1)
    p.write_text(text)


wire_domain("forge-core/src/domains/low_rank.rs", reward_index=2)
wire_domain("forge-core/src/domains/simd_kernel.rs")
wire_domain("forge-core/src/domains/cuda_kernel.rs")
