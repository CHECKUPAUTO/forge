//! Cache d'évaluation persistant et thread-safe.
//!
//! Un score n'est réutilisable que dans le même contexte d'évaluation : domaine,
//! graine de trial et environnement matériel/logique. Le précédent cache indexé
//! uniquement par `CandidateId` pouvait réutiliser le score d'une génération
//! précédente malgré la rotation des entrées, ce qui invalidait l'anti-overfit.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read};
use std::path::Path;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::candidate::CandidateId;

const CACHE_SCHEMA_VERSION: u32 = 2;

#[derive(Serialize, Deserialize)]
pub struct CacheStore {
    #[serde(default = "cache_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub records: HashMap<String, Vec<f64>>,
}

const fn cache_schema_version() -> u32 {
    CACHE_SCHEMA_VERSION
}

impl Default for CacheStore {
    fn default() -> Self {
        Self {
            schema_version: CACHE_SCHEMA_VERSION,
            records: HashMap::new(),
        }
    }
}

pub struct EvaluationCache {
    store: RwLock<CacheStore>,
    persistent_path: String,
    environment_fingerprint: String,
}

impl EvaluationCache {
    pub fn new(path: &str) -> Self {
        let store = if Path::new(path).exists() {
            Self::load_from_disk(path).unwrap_or_default()
        } else {
            CacheStore::default()
        };
        Self {
            store: RwLock::new(store),
            persistent_path: path.to_string(),
            environment_fingerprint: Self::default_environment_fingerprint(),
        }
    }

    /// Permet à une campagne d'ajouter un identifiant matériel/toolchain plus
    /// précis (ex. hash de `rustc -Vv`, CPU/GPU, flags) sans changer le format.
    pub fn with_environment_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.environment_fingerprint = fingerprint.into();
        self
    }

    fn default_environment_fingerprint() -> String {
        let namespace = std::env::var("FORGE_CACHE_ENV").unwrap_or_else(|_| "default".into());
        format!(
            "forge-core:{}:{}:{}:{}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
            namespace
        )
    }

    pub fn scoped_key(
        &self,
        domain: &str,
        candidate_id: CandidateId,
        trial_seed: u64,
    ) -> String {
        format!(
            "v{CACHE_SCHEMA_VERSION}|{}|{domain}|{trial_seed:016x}|{candidate_id:016x}",
            self.environment_fingerprint
        )
    }

    pub fn get_scoped(
        &self,
        domain: &str,
        candidate_id: CandidateId,
        trial_seed: u64,
    ) -> Option<Vec<f64>> {
        let key = self.scoped_key(domain, candidate_id, trial_seed);
        self.store.read().ok()?.records.get(&key).cloned()
    }

    pub fn insert_scoped(
        &self,
        domain: &str,
        candidate_id: CandidateId,
        trial_seed: u64,
        objectives: Vec<f64>,
    ) {
        if objectives.is_empty() || !objectives.iter().all(|v| v.is_finite()) {
            return;
        }
        let key = self.scoped_key(domain, candidate_id, trial_seed);
        if let Ok(mut writer) = self.store.write() {
            writer.records.insert(key, objectives);
        }
    }

    /// Compatibilité API avec l'ancien moteur. Un lookup non contextualisé est
    /// volontairement refusé : mieux vaut réévaluer que réutiliser un score
    /// issu d'un autre trial ou d'une autre machine.
    pub fn get(&self, _id: CandidateId) -> Option<Vec<f64>> {
        None
    }

    /// Compatibilité API : aucune valeur non contextualisée n'est persistée.
    pub fn insert(&self, _id: CandidateId, _objectives: Vec<f64>) {}

    pub fn persist(&self) -> std::io::Result<()> {
        let reader = self
            .store
            .read()
            .map_err(|_| std::io::Error::other("Lock corrompu"))?;
        let tmp_path = format!("{}.tmp", self.persistent_path);
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);

        serde_json::to_writer(&mut writer, &*reader).map_err(std::io::Error::other)?;
        writer.into_inner()?.sync_all()?;
        std::fs::rename(tmp_path, &self.persistent_path)?;
        Ok(())
    }

    fn load_from_disk(path: &str) -> std::io::Result<CacheStore> {
        let mut file = File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        let store: CacheStore = serde_json::from_str(&content).map_err(std::io::Error::other)?;
        if store.schema_version != CACHE_SCHEMA_VERSION {
            return Ok(CacheStore::default());
        }
        Ok(store)
    }
}
