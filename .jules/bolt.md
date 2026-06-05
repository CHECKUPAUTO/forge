## 2025-05-15 - Redundant Evaluation Bottleneck
**Learning:** In evolutionary algorithms, especially when using LLMs for mutation, the same candidate source code can reappear frequently across generations. Evaluating these candidates repeatedly is extremely expensive (up to 45s per evaluation in SIMD domains).
**Action:** Always implement short-circuiting logic using the `EvaluationCache` in both local and distributed evaluation paths to avoid redundant computations.

## 2025-05-15 - Efficient Cache Persistence
**Learning:** Pretty-printing large JSON caches using `serde_json::to_string_pretty` causes unnecessary memory allocations and slow I/O.
**Action:** Use `serde_json::to_writer` with a `BufWriter` to stream data directly to disk, reducing memory overhead and improving persistence speed.
