//! Parser de rapports Criterion pour le profiling matériel réel.
//!
//! Remplace les métriques simulées par la lecture des fichiers JSON produits
//! par `cargo bench` en mode `--output-format bencher` ou via l'arborescence
//! standard `target/criterion/<bench_name>/new/estimates.json`.
//!
//! ## Format attendu (Criterion >= 0.5)
//! ```json
//! {
//!   "mean": {
//!     "confidence_interval": { "lower_bound": 1228.1, "upper_bound": 1272.9 },
//!     "point_estimate": 1250.5,
//!     "standard_error": 12.3
//!   },
//!   "median": { "point_estimate": 1245.0, ... },
//!   "std_dev": { "point_estimate": 45.6 }
//! }
//! ```
//!
//! Les valeurs sont en **nanosecondes** (ns) par défaut avec Criterion.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::ForgeError;

/// Intervalle de confiance tel que rapporté par Criterion.
#[derive(Deserialize, Debug, Clone)]
struct ConfidenceInterval {
    lower_bound: f64,
    upper_bound: f64,
}

/// Estimation ponctuelle + erreur standard telle que produite par Criterion.
#[derive(Deserialize, Debug, Clone)]
struct CriterionEstimate {
    point_estimate: f64,
    standard_error: Option<f64>,
    confidence_interval: Option<ConfidenceInterval>,
}

/// Structure complète du fichier `estimates.json`.
#[derive(Deserialize, Debug, Clone)]
struct CriterionEstimates {
    mean: CriterionEstimate,
    #[serde(rename = "std_dev")]
    std_dev: Option<CriterionEstimate>,
    median: Option<CriterionEstimate>,
}

/// Résultat parsé d'un benchmark Criterion.
#[derive(Debug, Clone)]
pub struct CriterionMetrics {
    /// Latence moyenne en nanosecondes.
    pub mean_latency_ns: f64,
    /// Erreur standard associée à la moyenne.
    pub standard_error_ns: f64,
    /// Écart-type (si disponible) — utile pour détecter le bruit thermique.
    pub std_dev_ns: Option<f64>,
    /// Médiane (si disponible).
    pub median_ns: Option<f64>,
}

// ---------------------------------------------------------------------------
// Fonction principale de parsing et validation de stabilité
// ---------------------------------------------------------------------------

/// Parse le fichier `estimates.json` d'un benchmark Criterion et valide
/// la stabilité statistique de la mesure.
///
/// # Arguments
/// * `workspace_dir` — répertoire racine de l'espace de travail (contient `target/criterion/`).
/// * `bench_id` — identifiant du benchmark (ex: `"gemm_target"`).
/// * `max_allowed_variance_ratio` — ratio maximal acceptable (`standard_error / mean`).
///   Par exemple `0.05` autorise jusqu'à 5% de bruit relatif.
///
/// # Retourne
/// * `Ok(Vec<f64>)` — les objectifs mesurés (ex: `[mean_latency_ns]`) si stables.
/// * `Err(ForgeError::Evaluation)` — si la mesure est absente, corrompue ou instable.
///
/// # Règle de filtrage du bruit thermique
/// Si le ratio `standard_error_ns / mean_latency_ns` dépasse
/// `max_allowed_variance_ratio`, la mesure est rejetée car considérée comme
/// parasitée par du bruit CPU (throttling thermique, interruptions, etc.),
/// ce qui invalide temporairement le candidat.
pub fn parse_and_validate_metrics(
    workspace_dir: &Path,
    bench_id: &str,
    max_allowed_variance_ratio: f64,
) -> Result<Vec<f64>, ForgeError> {
    // 1. Localiser le fichier estimates.json
    let path = workspace_dir
        .join("target")
        .join("criterion")
        .join(bench_id)
        .join("new")
        .join("estimates.json");

    let path = if path.exists() {
        path
    } else {
        // Fallback : certains projets utilisent `base/` au lieu de `new/`
        let alt_path = workspace_dir
            .join("target")
            .join("criterion")
            .join(bench_id)
            .join("base")
            .join("estimates.json");
        if alt_path.exists() {
            alt_path
        } else {
            return Err(ForgeError::Evaluation(format!(
                "Rapport Criterion estimates.json introuvable pour '{bench_id}' \
                 dans {workspace_dir}",
                workspace_dir = workspace_dir.display()
            )));
        }
    };

    // 2. Parser le JSON
    let metrics = parse_estimates_file(&path)
        .map_err(|e| ForgeError::Evaluation(format!("Erreur parsing estimates.json: {e}")))?;

    // 3. Extraire ou calculer l'erreur standard
    let mean_ns = metrics.mean_latency_ns;
    let std_error_ns = metrics.standard_error_ns;

    // Protection contre une moyenne nulle ou négative
    if mean_ns <= 0.0 {
        return Err(ForgeError::Evaluation(format!(
            "Moyenne Criterion invalide pour '{bench_id}': mean={mean_ns} ns"
        )));
    }

    // 4. Validation du ratio de variance (filtrage du bruit thermique)
    let variance_ratio = std_error_ns / mean_ns;
    if variance_ratio > max_allowed_variance_ratio {
        return Err(ForgeError::Evaluation(format!(
            "Mesure physique instable pour '{bench_id}': \
             std_error/mean = {variance_ratio:.4} > seuil={max_allowed_variance_ratio}. \
             Bruit CPU probable (throttling thermique, interruptions). \
             Candidat temporairement invalidé."
        )));
    }

    // 5. Succès : retourne la métrique de latence comme objectif
    Ok(vec![mean_ns])
}

// ---------------------------------------------------------------------------
// Fonctions auxiliaires de parsing
// ---------------------------------------------------------------------------

/// Extrait la latence moyenne en nanosecondes d'un fichier `estimates.json`
/// produit par Criterion.
///
/// # Arguments
/// * `target_dir` — répertoire racine contenant `target/criterion/`
/// * `bench_name` — nom du benchmark (ex: `compress_tensor`)
pub fn extract_criterion_metrics(
    target_dir: &Path,
    bench_name: &str,
) -> std::io::Result<CriterionMetrics> {
    let path = target_dir
        .join("target")
        .join("criterion")
        .join(bench_name)
        .join("new")
        .join("estimates.json");

    if !path.exists() {
        let alt_path = target_dir
            .join("target")
            .join("criterion")
            .join(bench_name)
            .join("base")
            .join("estimates.json");
        if alt_path.exists() {
            return parse_estimates_file(&alt_path);
        }
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Criterion estimates.json introuvable pour '{bench_name}'"),
        ));
    }

    parse_estimates_file(&path)
}

/// Extrait uniquement la latence moyenne (convenience wrapper).
pub fn extract_criterion_latency(target_dir: &Path, bench_name: &str) -> std::io::Result<f64> {
    extract_criterion_metrics(target_dir, bench_name).map(|m| m.mean_latency_ns)
}

// ---------------------------------------------------------------------------
// Parsing interne
// ---------------------------------------------------------------------------

fn parse_estimates_file(path: &Path) -> std::io::Result<CriterionMetrics> {
    let content = fs::read_to_string(path)?;
    let estimates: CriterionEstimates = serde_json::from_str(&content).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("JSON invalide: {e}"),
        )
    })?;

    // Récupérer l'erreur standard :
    //   - Priorité à `standard_error` si présent
    //   - Sinon, calcul à partir de `confidence_interval` (95% CI : ±1.96 std_err)
    let standard_error_ns = estimates.mean.standard_error.unwrap_or_else(|| {
        estimates
            .mean
            .confidence_interval
            .as_ref()
            .map(|ci| (ci.upper_bound - ci.lower_bound) / (2.0 * 1.96))
            .unwrap_or(0.0)
    });

    Ok(CriterionMetrics {
        mean_latency_ns: estimates.mean.point_estimate,
        standard_error_ns,
        std_dev_ns: estimates.std_dev.map(|s| s.point_estimate),
        median_ns: estimates.median.map(|m| m.point_estimate),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_estimates() {
        let json = r#"{
            "mean": {
                "point_estimate": 1250.5,
                "standard_error": 12.3,
                "confidence_interval": {
                    "lower_bound": 1226.4,
                    "upper_bound": 1274.6
                }
            },
            "std_dev": { "point_estimate": 45.6, "standard_error": 1.2 },
            "median": {
                "point_estimate": 1245.0,
                "standard_error": 11.0
            }
        }"#;
        let estimates: CriterionEstimates = serde_json::from_str(json).expect("parse");
        assert!((estimates.mean.point_estimate - 1250.5).abs() < 1e-9);
        assert_eq!(estimates.mean.standard_error, Some(12.3));
        assert_eq!(estimates.std_dev.unwrap().point_estimate, 45.6);
        assert_eq!(estimates.median.unwrap().point_estimate, 1245.0);
    }

    #[test]
    fn test_parse_minimal_estimates_no_std_error() {
        // Criterion peut ne pas inclure std_error mais avoir confidence_interval
        let json = r#"{
            "mean": {
                "point_estimate": 999.0,
                "confidence_interval": {
                    "lower_bound": 970.0,
                    "upper_bound": 1028.0
                }
            }
        }"#;
        let estimates: CriterionEstimates = serde_json::from_str(json).expect("parse");
        assert_eq!(estimates.mean.point_estimate, 999.0);
        assert!(estimates.mean.standard_error.is_none());
        assert!(estimates.std_dev.is_none());

        // Vérifier le calcul de l'erreur standard via confidence_interval
        let ci = estimates.mean.confidence_interval.as_ref().unwrap();
        let computed_se = (ci.upper_bound - ci.lower_bound) / (2.0 * 1.96);
        // 95% CI: SE = (1028 - 970) / (2 * 1.96) = 58 / 3.92 ≈ 14.796
        assert!((computed_se - 14.7959).abs() < 0.1);
    }

    #[test]
    fn test_parse_estimates_file_minimal() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("test_criterion_parser");
        let _ = std::fs::create_dir_all(&dir);
        let json = r#"{
            "mean": {
                "point_estimate": 500.0,
                "standard_error": 5.0
            }
        }"#;
        let path = dir.join("estimates.json");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let metrics = parse_estimates_file(&path).expect("parse");
        assert_eq!(metrics.mean_latency_ns, 500.0);
        assert_eq!(metrics.standard_error_ns, 5.0);
        assert!(metrics.std_dev_ns.is_none());
        assert!(metrics.median_ns.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_validate_stable_measurement() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("test_criterion_validate_stable");
        let criterion_dir = dir
            .join("target")
            .join("criterion")
            .join("my_bench")
            .join("new");
        let _ = std::fs::create_dir_all(&criterion_dir);

        // Mesure stable : std_error/mean = 5/500 = 0.01 (1%)
        let json = r#"{
            "mean": {
                "point_estimate": 500.0,
                "standard_error": 5.0
            }
        }"#;
        let mut f = std::fs::File::create(criterion_dir.join("estimates.json")).unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let result = parse_and_validate_metrics(&dir, "my_bench", 0.05);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![500.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_validate_unstable_measurement_rejected() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("test_criterion_validate_unstable");
        let criterion_dir = dir
            .join("target")
            .join("criterion")
            .join("noisy_bench")
            .join("new");
        let _ = std::fs::create_dir_all(&criterion_dir);

        // Mesure instable : std_error/mean = 60/500 = 0.12 (12%) > 5% seuil
        let json = r#"{
            "mean": {
                "point_estimate": 500.0,
                "standard_error": 60.0
            }
        }"#;
        let mut f = std::fs::File::create(criterion_dir.join("estimates.json")).unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let result = parse_and_validate_metrics(&dir, "noisy_bench", 0.05);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("instable"));
        assert!(err.contains("0.12"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_validate_missing_file() {
        let dir = std::env::temp_dir().join("test_criterion_missing");
        let result = parse_and_validate_metrics(&dir, "ghost_bench", 0.05);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_validate_zero_mean_rejected() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("test_criterion_zero_mean");
        let criterion_dir = dir
            .join("target")
            .join("criterion")
            .join("bad_bench")
            .join("new");
        let _ = std::fs::create_dir_all(&criterion_dir);

        let json = r#"{
            "mean": {
                "point_estimate": 0.0,
                "standard_error": 1.0
            }
        }"#;
        let mut f = std::fs::File::create(criterion_dir.join("estimates.json")).unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let result = parse_and_validate_metrics(&dir, "bad_bench", 0.05);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_validate_with_confidence_interval_fallback() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("test_criterion_ci_fallback");
        let criterion_dir = dir
            .join("target")
            .join("criterion")
            .join("ci_bench")
            .join("new");
        let _ = std::fs::create_dir_all(&criterion_dir);

        // Pas de standard_error, mais confidence_interval disponible
        // CI: [460, 540] → SE ≈ (540-460)/(2*1.96) ≈ 20.41
        // ratio = 20.41/500 ≈ 0.041 → sous le seuil 0.05
        let json = r#"{
            "mean": {
                "point_estimate": 500.0,
                "confidence_interval": {
                    "lower_bound": 460.0,
                    "upper_bound": 540.0
                }
            }
        }"#;
        let mut f = std::fs::File::create(criterion_dir.join("estimates.json")).unwrap();
        f.write_all(json.as_bytes()).unwrap();

        let result = parse_and_validate_metrics(&dir, "ci_bench", 0.05);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![500.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
