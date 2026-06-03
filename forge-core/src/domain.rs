//! Le trait `Domain` est *la* frontiere d'extension du moteur. Chacune des 4
//! campagnes (compression, quantification, kernels SIMD/GPU, routage MoE) est
//! une implementation de ce trait. Le moteur, lui, ne connait aucun domaine.
//!
//! Anti-triche n.2 : `verify` (correction) est strictement separe de `measure`
//! (performance). Un candidat "rapide mais faux" est rejete par la porte de
//! correction avant meme qu'on regarde sa vitesse. Le candidat ne calcule
//! jamais son propre score : c'est le harnais (cf. `evolve::evaluate`) qui
//! appelle `verify` puis `measure`. Pour les domaines "code", `measure`
//! compilera le candidat dans un binaire Criterion isole et lira la mesure ;
//! le candidat n'a aucun acces a la note.

use crate::candidate::Candidate;
use crate::error::Result;
use crate::trial::Trial;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Score d'un candidat. Les objectifs sont **minimises** et doivent etre
/// commensurables *a l'interieur d'un domaine* (sinon la domination Pareto
/// n'a pas de sens). On ne melange jamais les objectifs de deux domaines.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Score {
    /// Valeurs d'objectif, plus petit = meilleur, une entree par objectif.
    pub objectives: Vec<f64>,
    /// `false` => le candidat a echoue la porte de correction.
    pub valid: bool,
}

impl Score {
    /// Score d'un candidat qui n'a pas passe la porte de correction.
    pub fn invalid() -> Self {
        Score { objectives: Vec::new(), valid: false }
    }

    /// Construit un score valide a partir d'objectifs finis.
    pub fn valid(objectives: Vec<f64>) -> Self {
        Score { objectives, valid: true }
    }

    /// Domination au sens de Pareto (minimisation). Un score invalide est
    /// domine par n'importe quel score valide, et ne domine rien.
    pub fn dominates(&self, other: &Score) -> bool {
        if !self.valid {
            return false;
        }
        if !other.valid {
            return true;
        }
        if self.objectives.len() != other.objectives.len() {
            return false;
        }
        let mut strictly_better = false;
        for (a, b) in self.objectives.iter().zip(other.objectives.iter()) {
            if a > b {
                return false; // pire sur au moins un objectif
            }
            if a < b {
                strictly_better = true;
            }
        }
        strictly_better
    }
}

/// Un domaine de recherche. C'est ici que vit la fitness function.
pub trait Domain: Send + Sync {
    /// Type concret de candidat de ce domaine.
    type Cand: Candidate;

    /// Nom court du domaine (pour les logs).
    fn name(&self) -> &str;

    /// Candidat initial (point de depart de l'evolution).
    fn seed(&self, rng: &mut StdRng) -> Self::Cand;

    /// Produit un enfant a partir d'un ou plusieurs parents. Pour un domaine
    /// "code", c'est ici qu'un LLM est appele (cf. feature `llm`).
    fn mutate(&self, rng: &mut StdRng, parents: &[&Self::Cand]) -> Result<Self::Cand>;

    /// Porte de correction, executee sur des entrees randomisees par l'essai.
    /// `Ok(false)` = candidat *faux* (pas une erreur), `Err` = echec d'execution.
    fn verify(&self, cand: &Self::Cand, trial: &Trial) -> Result<bool>;

    /// Objectifs de performance. N'a de sens que si `verify` a renvoye `true`.
    fn measure(&self, cand: &Self::Cand, trial: &Trial) -> Result<Vec<f64>>;

    /// Noms des objectifs, dans l'ordre de `measure`.
    fn objective_names(&self) -> Vec<String>;

    /// Score de reference (la baseline a battre) pour cet essai.
    fn baseline(&self, trial: &Trial) -> Result<Score>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pareto_domination() {
        let a = Score::valid(vec![1.0, 2.0]);
        let b = Score::valid(vec![1.0, 3.0]);
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
        let c = Score::valid(vec![0.0, 5.0]);
        assert!(!a.dominates(&c)); // a meilleur sur obj1, pire sur obj0 => non domine
        assert!(!c.dominates(&a));
    }

    #[test]
    fn valid_beats_invalid() {
        let v = Score::valid(vec![9.0]);
        let i = Score::invalid();
        assert!(v.dominates(&i));
        assert!(!i.dominates(&v));
    }
}
