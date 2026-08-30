use std::fs::File;
use std::io::BufReader;
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

use rustls::pki_types::ServerName;
use rustls::{ClientConfig, ClientConnection, RootCertStore, StreamOwned};

use crate::error::{ForgeError, Result};

pub(crate) struct TlsEndpoint {
    pub host: String,
    pub address: String,
}

pub(crate) fn parse_tls_endpoint(addr: &str) -> Result<Option<TlsEndpoint>> {
    let Some(rest) = addr.strip_prefix("tls://") else {
        return Ok(None);
    };
    let (host, port) = rest.rsplit_once(':').ok_or_else(|| {
        ForgeError::Evaluation(format!(
            "Adresse TLS worker invalide '{addr}': format attendu tls://host:port"
        ))
    })?;
    if host.trim().is_empty() || port.parse::<u16>().is_err() {
        return Err(ForgeError::Evaluation(format!(
            "Adresse TLS worker invalide '{addr}': format attendu tls://host:port"
        )));
    }
    Ok(Some(TlsEndpoint {
        host: host.to_string(),
        address: format!("{host}:{port}"),
    }))
}

fn load_root_store(path: &str) -> Result<RootCertStore> {
    let file = File::open(path).map_err(|e| {
        ForgeError::Evaluation(format!("Impossible d'ouvrir la CA TLS '{path}': {e}"))
    })?;
    let mut reader = BufReader::new(file);
    let certs = rustls_pemfile::certs(&mut reader)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| ForgeError::Evaluation(format!("CA TLS PEM invalide '{path}': {e}")))?;
    if certs.is_empty() {
        return Err(ForgeError::Evaluation(format!("CA TLS vide dans '{path}'")));
    }

    let mut roots = RootCertStore::empty();
    for cert in certs {
        roots.add(cert).map_err(|e| {
            ForgeError::Evaluation(format!("Certificat CA TLS invalide dans '{path}': {e}"))
        })?;
    }
    Ok(roots)
}

pub(crate) fn connect_tls(
    endpoint: &TlsEndpoint,
    timeout: Duration,
) -> Result<StreamOwned<ClientConnection, TcpStream>> {
    let ca_path = std::env::var("FORGE_TLS_CA_CERT").map_err(|_| {
        ForgeError::Evaluation("FORGE_TLS_CA_CERT est requis pour une adresse worker tls://".into())
    })?;
    let roots = load_root_store(&ca_path)?;
    let config = ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();

    let server_name = ServerName::try_from(endpoint.host.clone()).map_err(|e| {
        ForgeError::Evaluation(format!("Nom TLS worker invalide '{}': {e}", endpoint.host))
    })?;

    let socket_addr = endpoint
        .address
        .to_socket_addrs()
        .map_err(|e| {
            ForgeError::Evaluation(format!(
                "Résolution worker TLS impossible '{}': {e}",
                endpoint.address
            ))
        })?
        .next()
        .ok_or_else(|| {
            ForgeError::Evaluation(format!(
                "Aucune adresse résolue pour le worker TLS '{}'",
                endpoint.address
            ))
        })?;

    let tcp = TcpStream::connect_timeout(&socket_addr, timeout).map_err(|e| {
        ForgeError::Evaluation(format!(
            "Connexion worker TLS perdue ({}): {e}",
            endpoint.address
        ))
    })?;
    tcp.set_read_timeout(Some(timeout))
        .map_err(|e| ForgeError::Evaluation(format!("Configuration timeout lecture TLS: {e}")))?;
    tcp.set_write_timeout(Some(timeout))
        .map_err(|e| ForgeError::Evaluation(format!("Configuration timeout écriture TLS: {e}")))?;

    let connection = ClientConnection::new(Arc::new(config), server_name).map_err(|e| {
        ForgeError::Evaluation(format!("Initialisation client TLS impossible: {e}"))
    })?;
    Ok(StreamOwned::new(connection, tcp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tls_scheme_is_explicit() {
        assert!(parse_tls_endpoint("127.0.0.1:9000").unwrap().is_none());
        let endpoint = parse_tls_endpoint("tls://worker.example:9443")
            .unwrap()
            .expect("tls endpoint");
        assert_eq!(endpoint.host, "worker.example");
        assert_eq!(endpoint.address, "worker.example:9443");
    }

    #[test]
    fn malformed_tls_endpoint_is_rejected() {
        assert!(parse_tls_endpoint("tls://worker.example").is_err());
        assert!(parse_tls_endpoint("tls://:9443").is_err());
        assert!(parse_tls_endpoint("tls://worker.example:not-a-port").is_err());
    }
}
