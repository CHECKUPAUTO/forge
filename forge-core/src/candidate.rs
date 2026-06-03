//! Un candidat = une solution proposee dans un domaine donne.
//!
//! Pour un domaine "code" (les 4 cibles scirust), `repr()` renvoie le code
//! Rust genere ; pour un domaine parametrique (la demo bin-packing), il
//! renvoie la representation des parametres. Dans les deux cas, `id()` est un
//! hash stable du contenu : il sert a dedupliquer l'archive et a detecter la
//! reapparition d'un meme candidat (donc a mesurer la diversite reelle).

/// Identifiant stable, derive du contenu du candidat.
pub type CandidateId = u64;

/// Contrat minimal d'un candidat. `Clone + Send + Sync` pour permettre une
/// future evaluation parallele sans changer l'API.
pub trait Candidate: Clone + Send + Sync {
    /// Hash stable du contenu (typiquement `fnv1a(&self.repr())`).
    fn id(&self) -> CandidateId;
    /// Representation textuelle exacte (code source, ou parametres).
    fn repr(&self) -> String;
}

/// Hash FNV-1a 64 bits, deterministe et sans dependance.
pub fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_is_deterministic_and_sensitive() {
        assert_eq!(fnv1a("scirust"), fnv1a("scirust"));
        assert_ne!(fnv1a("scirust"), fnv1a("scirusu"));
    }
}
