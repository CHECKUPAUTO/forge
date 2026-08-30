use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::ServerConfig;
use tokio_rustls::TlsAcceptor;

fn load_certs(path: &str) -> Result<Vec<CertificateDer<'static>>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let certs = rustls_pemfile::certs(&mut reader).collect::<Result<Vec<_>, _>>()?;
    if certs.is_empty() {
        return Err(format!("Aucun certificat TLS trouvé dans '{path}'").into());
    }
    Ok(certs)
}

fn load_key(path: &str) -> Result<PrivateKeyDer<'static>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    rustls_pemfile::private_key(&mut reader)?
        .ok_or_else(|| format!("Aucune clé privée TLS trouvée dans '{path}'").into())
}

pub(crate) fn acceptor_from_env() -> Result<Option<TlsAcceptor>, Box<dyn std::error::Error>> {
    let cert = std::env::var("FORGE_WORKER_TLS_CERT").ok();
    let key = std::env::var("FORGE_WORKER_TLS_KEY").ok();

    match (cert, key) {
        (None, None) => Ok(None),
        (Some(cert_path), Some(key_path)) => {
            let certs = load_certs(&cert_path)?;
            let key = load_key(&key_path)?;
            let config = ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(certs, key)?;
            Ok(Some(TlsAcceptor::from(Arc::new(config))))
        }
        _ => Err(
            "FORGE_WORKER_TLS_CERT et FORGE_WORKER_TLS_KEY doivent être définis ensemble".into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tls_is_disabled_when_no_credentials_are_configured() {
        let cert = std::env::var_os("FORGE_WORKER_TLS_CERT");
        let key = std::env::var_os("FORGE_WORKER_TLS_KEY");
        if cert.is_none() && key.is_none() {
            assert!(acceptor_from_env().unwrap().is_none());
        }
    }
}
