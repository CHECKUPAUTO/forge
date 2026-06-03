use forge_core::cache::EvaluationCache;
use forge_core::fnv1a;

#[test]
fn test_cache_insert_and_get() {
    let cache = EvaluationCache::new("/tmp/test_cache_forge.json");
    let id = fnv1a("test_candidate_1");
    let objectives = vec![1.0, 2.0, 3.0];

    // Should be None initially
    assert!(cache.get(id).is_none());

    cache.insert(id, objectives.clone());

    // After insert, should return the objectives
    assert_eq!(cache.get(id), Some(objectives));
}

#[test]
fn test_cache_persistence() {
    let path = "/tmp/test_cache_persist.json";
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(format!("{path}.tmp"));

    let id = fnv1a("persist_test_cand");
    let objectives = vec![42.0, -7.5];

    {
        let cache = EvaluationCache::new(path);
        cache.insert(id, objectives.clone());
        cache.persist().expect("persist should succeed");
    }

    let cache2 = EvaluationCache::new(path);
    assert_eq!(cache2.get(id), Some(objectives));

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_cache_empty_on_new() {
    let path = "/tmp/test_cache_nonexistent.json";
    let _ = std::fs::remove_file(path);
    let cache = EvaluationCache::new(path);
    assert!(cache.get(42).is_none());
    let _ = std::fs::remove_file(path);
}
