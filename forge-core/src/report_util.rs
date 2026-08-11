//! Utilitaires communs aux runners forge.

/// Retourne l'index dans le front du candidat elite (holdout-verifie, bat le baseline sur objective 0).
/// Necessite C: Clone + Candidate car Individual n'est pas Clone par defaut.
pub fn find_elite_index<C: crate::Candidate + Clone>(report: &crate::Report<C>) -> Option<usize> {
    let baseline_val = report
        .final_baseline
        .as_ref()
        .and_then(|b| b.objectives.first().copied())
        .unwrap_or(f64::INFINITY);

    let mut best_idx = None;
    let mut best_score = f64::INFINITY;

    for (i, ind) in report.final_front.iter().enumerate() {
        if !report
            .final_front_holdout
            .get(i)
            .and_then(|o| o.as_ref())
            .map(|s| s.valid)
            .unwrap_or(false)
        {
            continue;
        }
        let score = ind.score.objectives[0];
        if score < best_score {
            best_score = score;
            best_idx = Some(i);
        }
    }

    best_idx.filter(|_| best_score < baseline_val)
}
