//! Registre transactionnel persistant basé sur Sled.
//! Gère l'historique et la traçabilité des lignées génétiques de candidats.
//!
//! Les enregistrements système utilisent le préfixe `__forge_system__/` et ne
//! sont jamais exposés par `iter()`, qui reste réservé aux `GenerationRecord`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::candidate::CandidateId;
use crate::error::{ForgeError, Result};

const SYSTEM_PREFIX: &[u8] = b"__forge_system__/";

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GenerationRecord {
    pub candidate_id: CandidateId,
    pub source_code: String,
    pub objectives: Vec<f64>,
    pub generation: u64,
    pub parent_ids: Vec<CandidateId>,
}

#[derive(Clone)]
pub struct AlgorithmRegistry {
    db: Arc<sled::Db>,
}

impl AlgorithmRegistry {
    pub fn open(path: &str) -> Result<Self> {
        let db = sled::open(path).map_err(|e| {
            ForgeError::Evaluation(format!("Échec de l'ouverture du stockage Sled: {e}"))
        })?;
        Ok(Self { db: Arc::new(db) })
    }

    pub fn commit_candidate(&self, record: &GenerationRecord) -> Result<()> {
        let key = record.candidate_id.to_be_bytes();
        let payload = serde_json::to_vec(record).map_err(|e| {
            ForgeError::Evaluation(format!("Erreur de sérialisation GenerationRecord: {e}"))
        })?;

        self.db.insert(key, payload).map_err(|e| {
            ForgeError::Evaluation(format!("Échec de l'insertion transactionnelle: {e}"))
        })?;
        self.db.flush().map_err(|e| {
            ForgeError::Evaluation(format!("Échec du flush matériel: {e}"))
        })?;
        Ok(())
    }

    pub fn get_candidate_record(&self, id: CandidateId) -> Result<Option<GenerationRecord>> {
        let key = id.to_be_bytes();
        match self.db.get(key) {
            Ok(Some(bytes)) => {
                let record: GenerationRecord = serde_json::from_slice(&bytes).map_err(|e| {
                    ForgeError::Evaluation(format!("Données de registre corrompues: {e}"))
                })?;
                Ok(Some(record))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(ForgeError::Evaluation(format!("Erreur d'accès à Sled DB: {e}"))),
        }
    }

    /// Parcourt uniquement les enregistrements candidats. Les clés système
    /// (checkpoint moteur, métadonnées) sont ignorées par construction.
    pub fn iter(&self) -> impl Iterator<Item = Result<GenerationRecord>> + '_ {
        self.db.iter().filter_map(|res| match res {
            Ok((key, ivec)) if key.as_ref().starts_with(SYSTEM_PREFIX) => None,
            Ok((_key, ivec)) => Some(
                serde_json::from_slice(&ivec)
                    .map_err(|e| ForgeError::Evaluation(format!("Désérialisation: {e}"))),
            ),
            Err(e) => Some(Err(ForgeError::Evaluation(format!("Sled iter: {e}")))),
        })
    }

    fn system_key(key: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(SYSTEM_PREFIX.len() + key.len());
        out.extend_from_slice(SYSTEM_PREFIX);
        out.extend_from_slice(key);
        out
    }

    /// Commit brut réservé aux données système du moteur.
    pub fn commit_raw(&self, key: &[u8], payload: &[u8]) -> Result<()> {
        self.db
            .insert(Self::system_key(key), payload)
            .map_err(|e| ForgeError::Evaluation(format!("Échec commit_raw Sled: {e}")))?;
        self.db
            .flush()
            .map_err(|e| ForgeError::Evaluation(format!("Échec flush commit_raw: {e}")))?;
        Ok(())
    }

    pub fn get_raw(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.db
            .get(Self::system_key(key))
            .map(|opt| opt.map(|ivec| ivec.to_vec()))
            .map_err(|e| ForgeError::Evaluation(format!("Erreur get_raw Sled: {e}")))
    }

    pub fn len(&self) -> usize {
        self.db
            .iter()
            .filter(|res| {
                res.as_ref()
                    .map(|(key, _)| !key.as_ref().starts_with(SYSTEM_PREFIX))
                    .unwrap_or(false)
            })
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> String {
        format!("/tmp/forge_registry_v4_{name}")
    }

    #[test]
    fn test_commit_and_get() {
        let path = tmp_path("commit");
        let _ = std::fs::remove_dir_all(&path);
        let reg = AlgorithmRegistry::open(&path).expect("open");

        let record = GenerationRecord {
            candidate_id: 42,
            source_code: "fn main() {}".into(),
            objectives: vec![1.0, 2.0],
            generation: 3,
            parent_ids: vec![10, 11],
        };
        reg.commit_candidate(&record).expect("commit");

        let fetched = reg.get_candidate_record(42).expect("get").expect("found");
        assert_eq!(fetched.parent_ids, vec![10, 11]);
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn system_records_do_not_pollute_candidate_iteration() {
        let path = tmp_path("system");
        let _ = std::fs::remove_dir_all(&path);
        let reg = AlgorithmRegistry::open(&path).expect("open");
        reg.commit_candidate(&GenerationRecord {
            candidate_id: 1,
            source_code: "v1".into(),
            objectives: vec![1.0],
            generation: 0,
            parent_ids: vec![],
        })
        .unwrap();
        reg.commit_raw(b"__engine_checkpoint__", br#"{"state":1}"#)
            .unwrap();

        let all: Vec<_> = reg.iter().collect::<Result<Vec<_>>>().expect("iter");
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].candidate_id, 1);
        assert_eq!(reg.len(), 1);
        assert!(reg.get_raw(b"__engine_checkpoint__").unwrap().is_some());
        let _ = std::fs::remove_dir_all(&path);
    }
}
