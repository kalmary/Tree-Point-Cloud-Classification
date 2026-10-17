# Tree Classification Import Repair Plan

**Goal:** Separate root-facing inference and BDL imports from standalone data
preparation, training, evaluation, and plotting imports without changing model
or label-selection behavior.

**Root-facing contracts:** `TreeClassifier.predict(point_cloud)` and the
currently used `BdlCall` construction, mapping, data-map, and prediction
methods.

**Parent plan:** `../../../docs/plan.md`

## Frozen behavior

- Do not split, rewrite, optimize, or retune point projection, tensor/model
  inference, species mapping, shrub handling, CRS handling, BDL requests, or
  model-versus-BDL precedence.
- Preserve model/config/species filenames, public signatures, CLI flags,
  dataset formats, and output layout.
- Preserve the nested `nn_utils` revision and API unless a separately approved
  canonical update is synchronized into every checkout.

## Completed foundation

- [x] Establish the Python 3.12 uv project and dependency groups.
- [x] Configure supported PyTorch profiles and preserve the nested `nn_utils`.
- [x] Verify current public imports independently and through the BRIK parent.

## Task 1: Characterize imports and invocations

- [x] Complete deterministic `TreeClassifier` and mocked `BdlCall` behavior
  tests before changing their imports.
- [x] Inventory every `__main__` guard without removing any executable.
- [x] Add separate blocked-import tests for classifier and BDL imports so each
  exposes its current accidental dependencies.
- [x] Add paired direct/module `--help` tests for current preparation,
  training, evaluation, inference, and diagnostic commands.

## Task 2: Repair root-facing imports

- [x] Replace `sys.path` mutation in reusable dependency loaders, generic
  utility imports, and wildcard imports with explicit imports from defining
  modules; direct scripts bootstrap only their own package entry point.
- [x] Keep classifier inference independent from dataset preparation, training,
  evaluation, reporting, and plotting modules.
- [x] Keep BDL imports independent from Torch, model definitions, data
  preparation, and visualization.
- [x] Keep inference projection imports separate from training-only
  augmentation imports without changing projection calculations.
- [x] Resolve metadata, configs, and weights through explicit or stable
  project-relative paths while preserving naming conventions.

## Task 3: Repair standalone workflow imports

- [x] Keep Laspy/HDF5/fpsample/scikit-learn dependencies local to preparation
  commands that use them.
- [x] Keep Optuna, Torchinfo, metrics, reporting, and plotting imports local to
  training/evaluation commands.
- [x] Remove only confirmed unused imports in a file when repairing that file's
  invocation path.
- [x] Ensure every current command handles `--help` before loading data,
  weights, CUDA, plotting, or making BDL requests.
- [x] Preserve direct-script and module execution from this project root.

## Task 4: Verify dependency ownership

- [x] Run classifier and BDL import-boundary tests under `basic` with the
  selected PyTorch extra where required.
- [x] Run all standalone invocation tests and the complete suite under `dev`.
- [x] Verify parent-root imports and existing root classification/BDL behavior.
- [ ] Run real BDL-network and production CUDA smoke tests separately in
  prepared environments.
- [ ] Change dependency groups only after these workflow checks pass.

## Completion gate

Classifier and BDL imports load only their real runtime dependencies, every
current standalone tool works in both declared forms, and all predictions and
label policies remain unchanged.
