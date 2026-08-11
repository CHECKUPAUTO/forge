//! Pipeline d'injection : pousse l'élite découverte par Forge dans un crate cible.
//! Agnostique au domaine : ELITE_DIR/SRC_FILE -> <TARGET>/src/<MODULE>.rs (+ provenance),
//! déclare `pub mod` dans lib.rs, puis `cargo test --release` comme porte de validation.
//!
//! L'injection est transactionnelle du point de vue des fichiers modifiés : si
//! l'écriture de `lib.rs`, le lancement de Cargo ou les tests échouent, le module
//! et `lib.rs` sont restaurés dans leur état précédent.

use std::io;
use std::path::Path;
use std::process::{Command, ExitStatus};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn valid_module_name(module: &str) -> bool {
    let mut chars = module.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn read_optional(path: &Path) -> io::Result<Option<Vec<u8>>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn restore_file(path: &Path, original: Option<&[u8]>) -> io::Result<()> {
    match original {
        Some(bytes) => std::fs::write(path, bytes),
        None => match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        },
    }
}

fn rollback(
    module_file: &Path,
    original_module: Option<&[u8]>,
    lib_path: &Path,
    original_lib: &[u8],
) {
    if let Err(error) = restore_file(module_file, original_module) {
        eprintln!(
            "ERREUR CRITIQUE: restauration de {} échouée: {error}",
            module_file.display()
        );
    }
    if let Err(error) = std::fs::write(lib_path, original_lib) {
        eprintln!(
            "ERREUR CRITIQUE: restauration de {} échouée: {error}",
            lib_path.display()
        );
    }
}

fn run_validation(target: &str) -> io::Result<ExitStatus> {
    Command::new("cargo")
        .args(["test", "--release"])
        .current_dir(target)
        .status()
}

fn main() {
    let elite_dir = env_or("ELITE_DIR", "/tmp/forge_elite");
    let src_file = env_or("SRC_FILE", "elite_compressor.rs");
    let target = env_or("TARGET", "/root/soulsystem-audit/scirust-tn");
    let module = env_or("MODULE", "discovered");

    if !valid_module_name(&module) {
        eprintln!(
            "MODULE invalide: '{module}'. Utiliser uniquement un identifiant Rust ASCII sans séparateur de chemin."
        );
        std::process::exit(1);
    }

    let src_path = Path::new(&elite_dir).join(&src_file);
    let manifest_path = Path::new(&elite_dir).join("manifest.txt");
    let target_src = Path::new(&target).join("src");
    let module_file = target_src.join(format!("{module}.rs"));
    let lib_path = target_src.join("lib.rs");

    let source = match std::fs::read_to_string(&src_path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("impossible de lire {} : {error}", src_path.display());
            eprintln!("(lance d'abord une campagne pour produire l'élite)");
            std::process::exit(1);
        }
    };

    let manifest = std::fs::read_to_string(&manifest_path).unwrap_or_default();
    if !manifest.is_empty() {
        println!("--- manifeste ---\n{manifest}");
    }

    let date = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|| "?".to_string());

    let mut header = String::new();
    header.push_str(
        "//! Algorithme découvert automatiquement par Forge (FunSearch/AlphaEvolve-style).\n",
    );
    header.push_str(&format!("//! Injecté le {date}.\n//!\n"));
    for line in manifest.lines() {
        header.push_str(&format!("//! {line}\n"));
    }
    header.push_str(
        "//!\n//! NE PAS éditer à la main : régénéré par le binaire `inject_elite`.\n\n",
    );

    let test_block = match std::env::var("ELITE_TEST_FILE") {
        Ok(test_file) if !test_file.is_empty() => match std::fs::read_to_string(&test_file) {
            Ok(test) => format!("\n\n{test}\n"),
            Err(error) => {
                eprintln!("test introuvable {test_file} : {error}");
                std::process::exit(1);
            }
        },
        _ => String::new(),
    };

    let original_module = match read_optional(&module_file) {
        Ok(value) => value,
        Err(error) => {
            eprintln!(
                "lecture de l'état initial {} échouée : {error}",
                module_file.display()
            );
            std::process::exit(1);
        }
    };
    let original_lib = match std::fs::read(&lib_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("lecture {} échouée : {error}", lib_path.display());
            std::process::exit(1);
        }
    };
    let lib = match String::from_utf8(original_lib.clone()) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("{} n'est pas UTF-8 : {error}", lib_path.display());
            std::process::exit(1);
        }
    };

    let contents = format!("{header}{source}{test_block}");
    let declaration = format!("pub mod {module};");
    let new_lib = if lib.contains(&declaration) {
        lib.clone()
    } else {
        let mut lines: Vec<String> = lib.lines().map(str::to_string).collect();
        match lines
            .iter()
            .rposition(|line| line.trim_start().starts_with("pub mod "))
        {
            Some(index) => lines.insert(index + 1, declaration.clone()),
            None => lines.insert(0, declaration.clone()),
        }
        lines.join("\n") + "\n"
    };

    if let Err(error) = std::fs::write(&module_file, contents.as_bytes()) {
        eprintln!("écriture {} échouée : {error}", module_file.display());
        std::process::exit(1);
    }
    println!("écrit {}", module_file.display());

    if let Err(error) = std::fs::write(&lib_path, new_lib.as_bytes()) {
        eprintln!("mise à jour {} échouée : {error}", lib_path.display());
        rollback(
            &module_file,
            original_module.as_deref(),
            &lib_path,
            &original_lib,
        );
        std::process::exit(1);
    }
    if lib.contains(&declaration) {
        println!("lib.rs : `{declaration}` déjà présent");
    } else {
        println!("lib.rs : `{declaration}` ajouté");
    }

    println!("--- cargo test --release dans {target} (porte CI) ---");
    match run_validation(&target) {
        Ok(status) if status.success() => {
            println!(">>> INJECTION OK : {module}.rs intégré et tests verts");
        }
        Ok(status) => {
            eprintln!(
                ">>> ÉCHEC : cargo test code {:?}; restauration automatique des fichiers cible",
                status.code()
            );
            rollback(
                &module_file,
                original_module.as_deref(),
                &lib_path,
                &original_lib,
            );
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!(
                ">>> impossible de lancer cargo : {error}; restauration automatique des fichiers cible"
            );
            rollback(
                &module_file,
                original_module.as_deref(),
                &lib_path,
                &original_lib,
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_name_rejects_path_traversal() {
        assert!(valid_module_name("discovered"));
        assert!(valid_module_name("_candidate_2"));
        assert!(!valid_module_name("../escape"));
        assert!(!valid_module_name("foo/bar"));
        assert!(!valid_module_name(""));
        assert!(!valid_module_name("9bad"));
    }

    #[test]
    fn restore_file_restores_or_removes_target() {
        let dir = std::env::temp_dir().join("forge_inject_restore_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("candidate.rs");

        std::fs::write(&path, b"new").unwrap();
        restore_file(&path, Some(b"old")).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"old");

        restore_file(&path, None).unwrap();
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
