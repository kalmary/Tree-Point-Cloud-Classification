# TreeClassification Rebuild Plan

**Goal:** Rebuild the tree-species inference, BDL resolution, preprocessing, training, and evaluation code incrementally while preserving the existing model behavior and root-facing classes.

**Root-facing contracts:** `TreeClassifier.predict(point_cloud)` and the used `BDLCall` construction, mapping, data-map, and prediction methods.

**Design:** `../../../docs/rebuild.md`

## Task 1: Establish the uv project

- [ ] Create `.python-version`, `pyproject.toml`, and `uv.lock` for Python 3.12.
- [ ] Derive minimal direct dependencies from imports rather than copying the frozen requirements file.
- [ ] Define `basic` for inference/headless BDL use and `test` including `basic`, `pytest`, `matplotlib`, and `pyvista`.
- [ ] Configure one compatible current-stable PyTorch/Torchvision/Torchaudio set for official CPU and CUDA indexes.
- [ ] Declare the shared `nn_utils` relationship explicitly and verify against its rebuilt contract.
- [ ] Verify clean basic/test syncs and PyTorch capability output.

## Task 2: Characterize inference and BDL behavior

- [ ] Test model/config basename resolution and missing/malformed artifacts.
- [ ] Test accepted cloud shapes, projection output, dtype/device conversion, no-grad inference, and returned label shape/type.
- [ ] Preserve current species mapping, exact-name rules, shrub handling, CRS use, and model-versus-BDL precedence.
- [ ] Mock HTTP/BDL interactions for deterministic tests, including timeouts, invalid responses, empty results, and no-match cases.
- [ ] Test imports from both the standalone repository and the BRIK parent.

## Task 3: Normalize package and entry-point imports

- [ ] Replace `sys.path` mutation and generic `utils` imports with explicit package-relative imports.
- [ ] Replace wildcard imports with named utilities.
- [ ] Keep direct scripts as thin compatibility wrappers and add module entry points.
- [ ] Preserve direct and module execution for preprocessing, training, evaluation, and inference scripts.
- [ ] Resolve species metadata, configs, models, datasets, and outputs without accidental working-directory dependence.

## Task 4: Separate online inference

- [ ] Keep `TreeClassifier` as the compatibility facade while separating artifact loading, projection, tensor conversion, and prediction.
- [ ] Keep `BDLCall` as the compatibility facade while separating coordinate handling, request transport, response parsing, and label policy.
- [ ] Ensure computational prediction tests need neither network access nor production weights.
- [ ] Make external failures explicit without silently changing a valid model prediction unless that is current documented behavior.

## Task 5: Separate offline workflows

- [ ] Isolate point-cloud preprocessing and dataset construction from runtime inference imports.
- [ ] Separate model definitions from training orchestration and Optuna search.
- [ ] Separate evaluation metrics from plot/report generation.
- [ ] Preserve existing JSON schemas, CLI flags, dataset splits, model/output names, and training result layout.
- [ ] Load plotting dependencies only on plotting/reporting paths.

## Task 6: Verify

- [ ] Run all unit and mocked BDL tests on CPU.
- [ ] Run direct/module invocation tests from documented working directories.
- [ ] Run deterministic small-model inference without network access.
- [ ] Run the production model CUDA smoke test on supported Linux hardware and record effective device and output shape.
- [ ] Run root integration tests for interior trees, shrubs, removed border trees, no-tree input, and BDL failure behavior.

## Completion Gate

The project is independently reproducible with uv; `TreeClassifier` and `BDLCall` remain root-compatible; network behavior is isolated and tested; all invocation modes pass; CPU and CUDA verification gates are satisfied.
