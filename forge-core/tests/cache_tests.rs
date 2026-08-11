use forge_core::cache::EvaluationCache;
use forge_core::fnv1a;

#[test]
fn scoped_cache_insert_and_get() {
    let path = "/tmp/test_cache_forge_v2.json";
    let _ = std::fs::remove_file(path);
    let cache = EvaluationCache::new(path).with_environment_fingerprint("test-env");
    let id = fnv1a("test_candidate_1");
    let objectives = vec![1.0, 2.0, 3.0];

    assert!(cache.get_scoped("simd", id, 10).is_none());
    cache.insert_scoped("simd", id, 10, objectives.clone());
    assert_eq!(cache.get_scoped("simd", id, 10), Some(objectives));
    let _ = std::fs::remove_file(path);
}

#[test]
fn cache_never_crosses_trial_domain_or_environment() {
    let path = "/tmp/test_cache_scope_forge_v2.json";
    let _ = std::fs::remove_file(path);
    let id = fnv1a("same_source");

    let cache = EvaluationCache::new(path).with_environment_fingerprint("machine-a");
    cache.insert_scoped("simd", id, 111, vec![5.0]);

    assert_eq!(cache.get_scoped("simd", id, 111), Some(vec![5.0]));
    assert!(cache.get_scoped("simd", id, 222).is_none());
    assert!(cache.get_scoped("cuda", id, 111).is_none());

    // L'ancienne API non contextualisée est volontairement un miss.
    assert!(cache.get(id).is_none());
    let _ = std::fs::remove_file(path);
}

#[test]
fn scoped_cache_persistence() {
    let path = "/tmp/test_cache_persist_v2.json";
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(format!("{path}.tmp"));

    let id = fnv1a("persist_test_cand");
    let objectives = vec![42.0, -7.5];

    {
        let cache = EvaluationCache::new(path).with_environment_fingerprint("persist-env");
        cache.insert_scoped("low_rank", id, 77, objectives.clone());
        cache.persist().expect("persist should succeed");
    }

    let cache2 = EvaluationCache::new(path).with_environment_fingerprint("persist-env");
    assert_eq!(
        cache2.get_scoped("low_rank", id, 77),
        Some(objectives)
    );

    let cache3 = EvaluationCache::new(path).with_environment_fingerprint("other-env");
    assert!(cache3.get_scoped("low_rank", id, 77).is_none());

    let _ = std::fs::remove_file(path);
}
