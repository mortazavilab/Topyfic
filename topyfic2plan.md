
## Plan: Topyfic 2.0 Strict PyTorchLDA

Topyfic 2.0 should ship a staged, strict-LDA PyTorch backend while keeping sklearn as the reference and fallback until parity gates are met. The plan below assumes dual-backend rollout first, tier-1 support for CPU, NVIDIA CUDA, and Apple Silicon MPS, and a high numerical-parity target against the current sklearn path.

**Steps**
1. Phase 1: freeze the current LDA contract and build a regression harness. Document exactly what the current backend must provide for training, inference, model state export/import, aggregation, and persistence. This blocks implementation because the repo currently has no tests.
2. Build baseline fixtures from small deterministic count matrices plus one or two fixed slices from the tutorial datasets. Generate golden outputs from the current sklearn backend for `fit_transform`, `transform`, `components_`, `exp_dirichlet_component_`, topic ordering, HDF5 round-trips, and rLDA aggregation.
3. Introduce a backend adapter layer without changing behavior yet. Start by wrapping the current sklearn implementation behind a smaller internal interface so surrounding code can be refactored before PyTorch lands.
4. Refactor current call sites to depend on that adapter instead of directly depending on sklearn internals. The highest-risk surfaces are train.py, utilsMakeModel.py, topModel.py, and analysis.py.
5. Phase 2: implement a strict PyTorchLDA CPU reference backend first. Prioritize correctness and compatibility over acceleration. This backend should preserve LDA semantics, expose stable model state, and support `fit_transform`, `transform`, exported state, and imported state.
6. Define a backend state contract explicitly: topic-word matrix, exp(E[log beta]) or equivalent stored form, priors, iteration counts, convergence/bound metrics, input feature count, and deterministic seed handling. Reconstruction should move away from ad hoc sklearn attribute injection.
7. Add parity tests between sklearn and PyTorch on CPU. Gate progress on tolerance-based checks for document-topic distributions, topic-word distributions, topic ranking stability, `calculate_leiden_clustering`, and `combine_topModels`. Handle topic permutation explicitly in the test logic.
8. Phase 3: add device support and execution policy. Extend the PyTorch backend to support CPU, CUDA, and MPS with explicit device selection and safe CPU fallback. Keep default numerics in FP32 for the entire first rollout.
9. Expose backend and device selection in the Python API first, then wire the same controls into the CLI and docs. Keep sklearn available as a supported fallback path during the staged migration.
10. Add cross-device regression coverage. Validate that CPU, CUDA, and MPS all train and infer successfully, emit the same shapes and state schema, and stay within device-specific tolerance bands relative to sklearn and the PyTorch CPU reference.
11. Phase 4: migrate persistence and reconstruction to backend-aware state files. Preserve HDF5 support, but make the file format versioned and backend-aware so sklearn and PyTorch models round-trip predictably. Keep unsafe pickle out of the new critical path.
12. Add persistence regressions covering save/load identity, cross-session reconstruction, legacy artifact compatibility where feasible, and explicit failure behavior for unsupported cross-backend loads.
13. Phase 5: performance and release hardening. Optimize batching, sparse-to-dense strategy, multiprocessing/device interaction, and memory usage on representative dataset slices. Benchmark sklearn CPU vs PyTorch CPU vs PyTorch CUDA vs PyTorch MPS.
14. Update release-facing surfaces only after parity and persistence gates are satisfied: installation docs, tutorials, examples, dependency setup, and backend migration notes. Defer sklearn deprecation to a later release rather than 2.0 GA.

**Relevant files**
- train.py — training entry points and multi-run orchestration
- utilsMakeModel.py — reconstruction, aggregation, filtering, and persistence logic
- topModel.py — model-state accessors and serialization helpers
- analysis.py — inference path via `transform(data.X)`
- main.py — CLI backend/device selection
- __init__.py — public export surface
- setup.py — dependency and packaging changes
- requirements.txt — environment pinning strategy
- README.md — top-level 2.0 backend story
- installation.rst — CPU/CUDA/MPS install paths
- tutorials.rst — tutorial index updates
- make_train_object.ipynb — small workflow validation
- C2C12.ipynb — representative notebook parity target
- microglia.ipynb — larger analysis workflow validation

**Verification**
1. Add unit tests for the backend contract: fit, fit_transform, transform, exported state, imported state, and shape invariants.
2. Add sklearn-vs-PyTorch CPU regression tests on deterministic synthetic data with explicit tolerance thresholds and topic-alignment logic.
3. Add integration tests for `Train.run_LDA_models`, `calculate_leiden_clustering`, `Analysis.calculate_cell_participation`, `combine_topModels`, and HDF5 round-trips.
4. Add conditional device-matrix tests for CPU always, CUDA when available, and MPS when available.
5. Add persistence tests for legacy artifacts, versioned 2.0 artifacts, and explicit unsupported-load failures.
6. Add tutorial smoke tests or scripted equivalents for at least the smaller public workflows.
7. Add benchmark scripts comparing runtime, peak memory, and numerical drift across sklearn CPU, PyTorch CPU, CUDA, and MPS.
8. Set release gates: contract tests, parity regressions, persistence round-trips, tutorial smoke tests, and platform validation must all pass before PyTorch becomes the default backend.

**Decisions**
- Dual backend first: sklearn remains the reference and fallback during migration.
- Tier-1 platforms: CPU fallback, NVIDIA CUDA, and Apple Silicon MPS.
- Compatibility goal: high numerical parity with sklearn using tolerance-based tests.
- Strict LDA means preserving LDA semantics and workflow compatibility; neural topic models are out of scope for 2.0.
- Mixed precision is excluded from the initial rollout.

I saved this plan to session memory so it can be handed off cleanly. If you want, I can now refine it into one of these formats:

1. A milestone-based roadmap with estimated effort by phase.
2. A test strategy document with concrete test cases and acceptance thresholds.
3. A technical design breakdown of the backend adapter and PyTorchLDA interface.