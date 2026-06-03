//! Garde-fou anti-régression du domaine SIMD.
//!
//! Avec l'ancien `verify` (entrées constantes a=0.5, b=2.0, c==n), le kernel
//! tricheur `c[i]=n` PASSAIT — c'était le reward-hack. Avec le `verify` corrigé
//! (entrées aléatoires tirées de `trial.seed` + comparaison à la référence),
//! il doit être recalé. Ce test casse le build si jamais le trou revient.

use forge_core::domains::simd_kernel::{SimdKernelCode, SimdKernelDomain};
use forge_core::{fnv1a, Domain, Trial};
use rand::SeedableRng;

#[test]
fn cheat_kernel_is_rejected() {
    let domain = SimdKernelDomain::new("/tmp/forge_simd_cheat");
    // Ignore a et b, écrit la constante que l'ancien harnais attendait.
    let cheat = "pub fn compute_kernel(c: &mut [f64], _a: &[f64], _b: &[f64], n: usize) {\n    \
                 for x in c.iter_mut() { *x = n as f64; }\n}\n";
    let cand = SimdKernelCode { source: cheat.to_string(), id: fnv1a(cheat) };
    let trial = Trial { generation: 0, seed: 123 };

    let ok = domain
        .verify(&cand, &trial)
        .expect("verify ne doit pas renvoyer d'erreur");
    assert!(
        !ok,
        "un kernel c[i]=n DOIT être recalé par le verify à entrées aléatoires"
    );
}

#[test]
fn honest_baseline_passes() {
    let domain = SimdKernelDomain::new("/tmp/forge_simd_honest");
    let cand = domain.seed(&mut rand::rngs::StdRng::seed_from_u64(0)); // GEMM naïf de référence
    let trial = Trial { generation: 1, seed: 777 };

    let ok = domain
        .verify(&cand, &trial)
        .expect("verify ne doit pas renvoyer d'erreur");
    assert!(ok, "le GEMM naïf de référence DOIT passer la vérification");
}
