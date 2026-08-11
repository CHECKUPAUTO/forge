//! Management des diagnostics d'erreur pour la boucle de rétroaction
//! et extraction chirurgicale des métriques physiques de Criterion.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{ForgeError, Result};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FailureStage {
    Compilation,
    Verification,
    Execution,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailureDiagnostics {
    pub stage: FailureStage,
    pub stdout: String,
    pub stderr: String,
}

impl FailureDiagnostics {
    pub fn new(stage: FailureStage, stdout: String, stderr: String) -> Self {
        FailureDiagnostics {
            stage,
            stdout,
            stderr,
        }
    }

    pub fn to_prompt_fragment(&self) -> String {
        format!("[FAILURE at {:?}] stderr:\n{}", self.stage, self.stderr)
    }
}

/// Parse le fichier d'estimation de Criterion pour en extraire le temps
/// d'exécution moyen (en ns). Conçu pour éliminer tout risque de panique
/// via un parsing d'arbre JSON dynamique.
pub fn extract_criterion_latency(workspace_dir: &Path, bench_id: &str) -> Result<f64> {
    let estimates_path = workspace_dir
        .join("target")
        .join("criterion")
        .join(bench_id)
        .join("new")
        .join("estimates.json");

    if !estimates_path.exists() {
        return Err(ForgeError::Evaluation(format!(
            "Rapport Criterion manquant pour [{bench_id}] au chemin: {}",
            estimates_path.display()
        )));
    }

    let mut file = File::open(&estimates_path)
        .map_err(|e| ForgeError::Evaluation(format!("Erreur ouverture estimates.json: {e}")))?;

    let mut content = String::new();
    file.read_to_string(&mut content)
        .map_err(|e| ForgeError::Evaluation(format!("Erreur lecture estimates.json: {e}")))?;

    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| ForgeError::Evaluation(format!("JSON Criterion corrompu: {e}")))?;

    let nanoseconds = json
        .get("mean")
        .and_then(|m| m.get("point_estimate"))
        .and_then(|p| p.as_f64())
        .ok_or_else(|| {
            ForgeError::Evaluation(
                "Structure JSON Criterion non conforme (mean.point_estimate manquante)".into(),
            )
        })?;

    Ok(nanoseconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_latency_missing_file() {
        let result = extract_criterion_latency(Path::new("/nonexistent"), "ghost_bench");
        assert!(result.is_err());
    }

    #[test]
    fn test_failure_diagnostics_prompt() {
        let diag = FailureDiagnostics::new(
            FailureStage::Compilation,
            String::new(),
            "error[E0308]: mismatched types".into(),
        );
        let frag = diag.to_prompt_fragment();
        assert!(frag.contains("Compilation"));
        assert!(frag.contains("E0308"));
    }

    #[test]
    fn test_failure_stage_serialization() {
        let stage = FailureStage::Verification;
        let json = serde_json::to_string(&stage).unwrap();
        let back: FailureStage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, FailureStage::Verification);
    }
}
