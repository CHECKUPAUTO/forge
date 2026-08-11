//! Pipeline d'injection : pousse l'elite decouvert par forge dans un crate cible.
//! Agnostique au domaine : ELITE_DIR/SRC_FILE -> <TARGET>/src/<MODULE>.rs (+ provenance),
//! declare `pub mod` dans lib.rs, puis `cargo test` (porte CI). Test optionnel via ELITE_TEST_FILE.
use std::path::Path;
use std::process::Command;

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let elite_dir = env_or("ELITE_DIR", "/tmp/forge_elite");
    let src_file = env_or("SRC_FILE", "elite_compressor.rs");
    let target = env_or("TARGET", "/root/soulsystem-audit/scirust-tn");
    let module = env_or("MODULE", "discovered");

    let src_path = Path::new(&elite_dir).join(&src_file);
    let manifest_path = Path::new(&elite_dir).join("manifest.txt");

    let source = match std::fs::read_to_string(&src_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("impossible de lire {} : {e}", src_path.display());
            eprintln!("(lance d'abord une campagne pour produire l'elite)");
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
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "?".to_string());

    let mut header = String::new();
    header.push_str(
        "//! Algorithme decouvert automatiquement par forge (FunSearch/AlphaEvolve-style).\n",
    );
    header.push_str(&format!("//! Injecte le {date}.\n//!\n"));
    for line in manifest.lines() {
        header.push_str(&format!("//! {line}\n"));
    }
    header
        .push_str("//!\n//! NE PAS editer a la main : regenere par le binaire `inject_elite`.\n\n");

    // Test optionnel fourni par le domaine ; sinon la correction est attestee par le holdout de forge.
    let test_block = match std::env::var("ELITE_TEST_FILE") {
        Ok(tf) if !tf.is_empty() => match std::fs::read_to_string(&tf) {
            Ok(t) => format!("\n\n{t}\n"),
            Err(e) => {
                eprintln!("test introuvable {tf} : {e}");
                std::process::exit(1);
            }
        },
        _ => String::new(),
    };

    let module_file = Path::new(&target).join("src").join(format!("{module}.rs"));
    let contents = format!("{header}{source}{test_block}");
    if let Err(e) = std::fs::write(&module_file, &contents) {
        eprintln!("ecriture {} echouee : {e}", module_file.display());
        std::process::exit(1);
    }
    println!("ecrit {}", module_file.display());

    let lib_path = Path::new(&target).join("src").join("lib.rs");
    let lib = match std::fs::read_to_string(&lib_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("lecture {} echouee : {e}", lib_path.display());
            std::process::exit(1);
        }
    };
    let decl = format!("pub mod {module};");
    if lib.contains(&decl) {
        println!("lib.rs : `{decl}` deja present");
    } else {
        let mut lines: Vec<String> = lib.lines().map(|l| l.to_string()).collect();
        match lines
            .iter()
            .rposition(|l| l.trim_start().starts_with("pub mod "))
        {
            Some(i) => lines.insert(i + 1, decl.clone()),
            None => lines.insert(0, decl.clone()),
        }
        let new_lib = lines.join("\n") + "\n";
        if let Err(e) = std::fs::write(&lib_path, &new_lib) {
            eprintln!("maj lib.rs echouee : {e}");
            std::process::exit(1);
        }
        println!("lib.rs : `{decl}` ajoute");
    }

    println!("--- cargo test --release dans {target} (porte CI) ---");
    match Command::new("cargo")
        .args(["test", "--release"])
        .current_dir(&target)
        .status()
    {
        Ok(s) if s.success() => println!(">>> INJECTION OK : {module}.rs integre et teste vert"),
        Ok(s) => {
            eprintln!(">>> ECHEC : cargo test code {:?} (module ecrit ; `git checkout .` dans la cible pour annuler)", s.code());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!(">>> impossible de lancer cargo : {e}");
            std::process::exit(1);
        }
    }
}
