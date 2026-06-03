//! Algorithme d'altération déterministe de constantes numériques pour les
//! mutations fines de l'AST. Utilisé par les domaines où le candidat est un
//! vecteur de poids (ex: heuristique de bin packing, coefficients de
//! compression) pour explorer localement l'espace des paramètres sans
//! passer par le LLM.
//!
//! La mutation est pilotée par un taux `mutation_rate` : chaque constante
//! a cette probabilité d'être légèrement décalée (±10% avec bruit gaussien).

use rand::rngs::StdRng;
use rand::Rng;

pub struct MicroMutator;

impl MicroMutator {
    /// Parcourt une représentation textuelle contenant des tableaux de
    /// constantes entre crochets (ex: `[0.1234, -0.5678, 1.0000]`) et
    /// applique des perturbations stochastiques aux valeurs numériques.
    pub fn mutate_embedded_weights(source: &str, rng: &mut StdRng, mutation_rate: f64) -> String {
        let mut result = String::new();
        let mut in_bracket = false;
        let mut current_num = String::new();

        for c in source.chars() {
            if c == '[' {
                in_bracket = true;
                result.push(c);
                continue;
            }
            if c == ']' {
                if !current_num.is_empty() {
                    result.push_str(&Self::tweak_token(&current_num, rng, mutation_rate));
                    current_num.clear();
                }
                in_bracket = false;
                result.push(c);
                continue;
            }
            if in_bracket {
                if c == ',' || c == ' ' {
                    if !current_num.is_empty() {
                        result.push_str(&Self::tweak_token(&current_num, rng, mutation_rate));
                        current_num.clear();
                    }
                    result.push(c);
                } else {
                    current_num.push(c);
                }
            } else {
                result.push(c);
            }
        }
        result
    }

    fn tweak_token(token: &str, rng: &mut StdRng, rate: f64) -> String {
        if let Ok(mut value) = token.trim().parse::<f64>() {
            if rng.gen_bool(rate) {
                let delta = rng.gen_range(-1.0..1.0) * 0.1 * value;
                value += if delta == 0.0 {
                    rng.gen_range(-0.01..0.01)
                } else {
                    delta
                };
            }
            format!("{:.4}", value)
        } else {
            token.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_no_mutation_on_zero_rate() {
        let mut rng = StdRng::seed_from_u64(42);
        let input = "[0.1234, -0.5678, 1.0000]";
        let output = MicroMutator::mutate_embedded_weights(input, &mut rng, 0.0);
        assert_eq!(output, input);
    }

    #[test]
    fn test_mutation_changes_something_with_rate_one() {
        let mut rng = StdRng::seed_from_u64(42);
        let input = "[0.5000, 0.5000, 0.5000, 0.5000]";
        let output = MicroMutator::mutate_embedded_weights(input, &mut rng, 1.0);
        // Avec un taux de 1.0, au moins une valeur devrait changer
        assert_ne!(output, input);
    }

    #[test]
    fn test_non_bracket_text_untouched() {
        let mut rng = StdRng::seed_from_u64(42);
        let input = "hello world 42.0";
        let output = MicroMutator::mutate_embedded_weights(input, &mut rng, 1.0);
        assert_eq!(output, input);
    }
}
