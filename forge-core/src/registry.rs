//! Registre transactionnel persistant basé sur Sled.
//! Gère l'historique et la traçabilité des lignées génétiques de candidats.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::candidate::CandidateId;
use crate::error::{ForgeError, Result};

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
    /// Initialise ou ouvre le stockage transactionnel NVMe.
    pub fn open(path: &str) -> Result<Self> {
        let db = sled::open(path).map_err(|e| {
            ForgeError::Evaluation(format!("Échec de l'ouverture du stockage Sled: {e}"))
        })?;
        Ok(Self { db: Arc::new(db) })
    }

    /// Commit de manière atomique et synchrone un candidat validé sur le disque.
    pub fn commit_candidate(&self, record: &GenerationRecord) -> Result<()> {
        let key = record.candidate_id.to_be_bytes();
        let payload = serde_json::to_vec(record).map_err(|e| {
            ForgeError::Evaluation(format!("Erreur de sérialisation binaire (Sled): {e}"))
        })?;

        self.db.insert(key, payload).map_err(|e| {
            ForgeError::Evaluation(format!("Échec de l'insertion transactionnelle: {e}"))
        })?;

        self.db.flush().map_err(|e| {
            ForgeError::Evaluation(format!("Échec du flush matériel: {e}"))
        })?;

        Ok(())
    }

    /// Extrait le profil complet d'un ancêtre par son identifiant.
    pub fn get_candidate_record(&self, id: CandidateId) -> Result<Option<GenerationRecord>> {
        let key = id.to_be_bytes();
        match self.db.get(key) {
            Ok(Some(bytes)) => {
                let record: GenerationRecord = serde_json::from_slice(&bytes)
                    .map_err(|e| ForgeError::Evaluation(format!("Données de registre corrompues: {e}")))?;
                Ok(Some(record))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(ForgeError::Evaluation(format!("Erreur d'accès à Sled DB: {e}"))),
        }
    }

    /// Parcourt tous les enregistrements.
    pub fn iter(&self) -> impl Iterator<Item = Result<GenerationRecord>> {
        self.db.iter().map(|res| {
            let (_key, ivec) =
                res.map_err(|e| ForgeError::Evaluation(format!("Sled iter: {e}")))?;
            let record: GenerationRecord = serde_json::from_slice(&ivec)
                .map_err(|e| ForgeError::Evaluation(format!("Désérialisation: {e}")))?;
            Ok(record)
        })
    }

    /// Commit brut (clé + payload arbitraire) pour le checkpointing moteur.
    pub fn commit_raw(&self, key: &[u8], payload: &[u8]) -> Result<()> {
        self.db.insert(key, payload).map_err(|e| {
            ForgeError::Evaluation(format!("Échec commit_raw Sled: {e}"))
        })?;
        self.db.flush().map_err(|e| {
            ForgeError::Evaluation(format!("Échec flush commit_raw: {e}"))
        })?;
        Ok(())
    }

    /// Récupère un payload brut par clé.
    pub fn get_raw(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.db.get(key).map(|opt| opt.map(|ivec| ivec.to_vec())).map_err(|e| {
            ForgeError::Evaluation(format!("Erreur get_raw Sled: {e}"))
        })
    }

    /// Nombre total d'enregistrements.
    pub fn len(&self) -> usize {
        self.db.len()
    }

    pub fn is_empty(&self) -> bool {
        self.db.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> String {
        format!("/tmp/forge_registry_v3_{name}")
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
        assert_eq!(fetched.candidate_id, 42);
        assert_eq!(fetched.generation, 3);
        assert_eq!(fetched.parent_ids, vec![10, 11]);

        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn test_iter_and_empty() {
        let path = tmp_path("iter");
        let _ = std::fs::remove_dir_all(&path);
        let reg = AlgorithmRegistry::open(&path).expect("open");
        assert!(reg.is_empty());

        for i in 0..5 {
            reg.commit_candidate(&GenerationRecord {
                candidate_id: i,
                source_code: format!("v{i}"),
                objectives: vec![i as f64],
                generation: i,
                parent_ids: if i == 0 { vec![] } else { vec![i - 1] },
            })
            .unwrap();
        }

        assert_eq!(reg.len(), 5);
        let all: Vec<_> = reg.iter().collect::<Result<Vec<_>>>().expect("iter");
        assert_eq!(all.len(), 5);

        let _ = std::fs::remove_dir_all(&path);
    }
}
