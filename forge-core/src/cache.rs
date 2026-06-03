//! Cache d'évaluation persistant thread-safe pour intercepter les candidats redondants.
//!
//! Chaque candidat est identifié par son `CandidateId` (hash FNV-1a de sa
//! représentation textuelle). Avant d'évaluer un candidat, le moteur consulte
//! le cache ; s'il y a un hit, l'évaluation est court-circuitée. La persistance
//! est atomique (write-tmp + sync + rename) pour survivre aux crashs.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::candidate::CandidateId;

/// Le store sérialisable sur disque.
#[derive(Serialize, Deserialize, Default)]
pub struct CacheStore {
    pub records: HashMap<CandidateId, Vec<f64>>,
}

/// Cache d'évaluation thread-safe avec persistance sur NVMe.
pub struct EvaluationCache {
    store: RwLock<CacheStore>,
    persistent_path: String,
}

impl EvaluationCache {
    /// Ouvre ou crée un cache persistant au chemin donné.
    pub fn new(path: &str) -> Self {
        let store = if Path::new(path).exists() {
            Self::load_from_disk(path).unwrap_or_default()
        } else {
            CacheStore::default()
        };
        EvaluationCache {
            store: RwLock::new(store),
            persistent_path: path.to_string(),
        }
    }

    /// Récupère les objectifs stockés pour un candidat, s'il existe.
    pub fn get(&self, id: CandidateId) -> Option<Vec<f64>> {
        let reader = self.store.read().ok()?;
        reader.records.get(&id).cloned()
    }

    /// Insère un nouveau résultat dans le cache (en mémoire uniquement —
    /// appeler `persist()` pour écrire sur disque).
    pub fn insert(&self, id: CandidateId, objectives: Vec<f64>) {
        if let Ok(mut writer) = self.store.write() {
            writer.records.insert(id, objectives);
        }
    }

    /// Persiste le cache sur disque de manière atomique (tmp + sync + rename).
    pub fn persist(&self) -> std::io::Result<()> {
        let reader = self
            .store
            .read()
            .map_err(|_| std::io::Error::other("Lock corrompu"))?;
        let tmp_path = format!("{}.tmp", self.persistent_path);
        let mut file = File::create(&tmp_path)?;
        let json = serde_json::to_string_pretty(&*reader)
            .map_err(std::io::Error::other)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        std::fs::rename(tmp_path, &self.persistent_path)?;
        Ok(())
    }

    fn load_from_disk(path: &str) -> std::io::Result<CacheStore> {
        let mut file = File::open(path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        serde_json::from_str(&content)
            .map_err(std::io::Error::other)
    }
}
