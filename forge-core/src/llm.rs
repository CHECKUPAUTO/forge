//! Generateur de candidats par LLM (client Ollama bloquant).
//!
//! Active par la feature `llm`. Les domaines "code" (compression,
//! quantification, kernels, routage) appellent ceci dans leur `mutate`/`seed`
//! pour faire proposer une variante de code Rust par un modele local. Ce module
//! n'est pas exerce par la demo bin-packing (qui mute des parametres, sans LLM).

use crate::error::{ForgeError, Result};

/// Appelle l'endpoint `/api/generate` d'Ollama et renvoie le texte produit.
///
/// `base_url` ex. "http://127.0.0.1:11434", `model` ex. "qwen2.5-coder:7b".
pub fn ollama_generate(base_url: &str, model: &str, prompt: &str) -> Result<String> {
    let url = format!("{}/api/generate", base_url.trim_end_matches('/'));
    let body = ureq::json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": { "temperature": 0.8 }
    });

    let value: serde_json::Value = ureq::post(&url)
        .send_json(body)
        .map_err(|e| ForgeError::Generation(format!("requete Ollama: {e}")))?
        .into_json()
        .map_err(|e| ForgeError::Generation(format!("reponse Ollama illisible: {e}")))?;

    value
        .get("response")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| ForgeError::Generation("champ 'response' absent".into()))
}
