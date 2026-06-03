//! Contexte d'une evaluation.
//!
//! Point cle anti-triche n.1 : la graine change a chaque generation. Les
//! entrees sur lesquelles un candidat est note ne sont donc *jamais* fixes
//! pendant la recherche. Un candidat ne peut pas se sur-ajuster a un jeu de
//! test statique : pour survivre il doit etre bon sur une distribution qui
//! bouge sous ses pieds. La validation finale utilise une graine de holdout
//! jamais vue durant l'evolution (cf. `evolve`).

use rand::rngs::StdRng;
use rand::SeedableRng;

/// Contexte d'un essai d'evaluation.
#[derive(Clone, Copy, Debug)]
pub struct Trial {
    /// Numero de generation (u64::MAX pour le holdout final).
    pub generation: u64,
    /// Graine de cet essai. `verify` et `measure` la partagent : ils voient
    /// donc exactement les memes entrees dans un meme essai.
    pub seed: u64,
}

impl Trial {
    /// RNG deterministe de l'essai. Rappele a l'identique => memes entrees.
    pub fn rng(&self) -> StdRng {
        StdRng::seed_from_u64(self.seed)
    }
}
