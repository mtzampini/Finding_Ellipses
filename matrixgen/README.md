# Kippenhahn Dataset Builder

Pipeline to generate a synthetic dataset of 3×3 complex matrices, labeled
according to the Kippenhahn classification of their numerical range W(A),
to be used for training a neural classifier.

Project connected to the **Polymath Jr. Research Program**, on the
intersection between numerical range, Blaschke products, and Poncelet's
closure theorem.

## The four cases

1. **Normal**: A is normal, W(A) is the convex hull of the eigenvalues.
2. **Reducible**: A ~ 2×2 ⊕ 1×1 block (ellipse + point).
3. **Irreducible, flat boundary**: deliberate construction (repeated top
   eigenvalue of H(θ) over a portion of the angle).
4. **Irreducible, smooth boundary**: generic/default case.

## Repo structure

```
kippenhahn_symbolic.py     symbolic derivation (sympy) of the coefficients
                            of the Kippenhahn polynomial L_A(u,v,w); lambdify
                            -> pure numpy function (coefficient_extractor)
generators.py               gen_case1..gen_case4, common schema _result()
feature_extractor.py        extraction of H,K from A, normalization, batch
                            feature extractor (9 real coefficients per matrix)
dataset_config.py           DatasetConfig dataclass (proportions, augmentation,
                            split, seed, generator kwargs)
dataset_builder.py           orchestrator: build_base_pool, split_base_pool,
                            augment_train_split, extract_features_for_splits,
                            standardize, sanity_check, save_dataset,
                            build_final_dataset
```

## Pipeline (order matters)

```
1. build_base_pool(config)          generate BASE matrices for the 4 cases, no augmentation
2. split_base_pool(...)             stratified train/val split ON THE BASE MATRICES
3. augment_train_split(...)         augmentation (unitary conjugation) ONLY on train
4. extract_features_for_splits(...) extract the 9 coefficients of L_A for each split
5. standardize(splits)              fit mean/std ONLY on train, apply to all splits
6. save_dataset(...)                save to disk
```

Critical point: augmentation must be done **after** the split, never before —
otherwise conjugated copies of the same matrix could end up in both train and
val, causing data leakage and artificially optimistic validation metrics.

## Format of the saved data

Two files per generated dataset, sharing a common prefix `{path_prefix}`:

- **`{path_prefix}_arrays.npz`** - pure numpy arrays, ready for training:
  - `X_train`, `y_train`, `X_val`, `y_val`
  - `X_*` has shape `(N, 9)` (9 standardized coefficients of L_A, the
    constant coefficient w³=1 is already dropped)
  - `y_*` has shape `(N,)`, values in {1,2,3,4}

- **`{path_prefix}_meta.pkl`** - everything else, for audit/debug/reproducibility:
  - `metadata_train`, `metadata_val`: lists of dicts aligned by index to
    `X_*`/`y_*` (contain `case`, `eigenvalues`, `certificate`,
    `certificate_threshold`, `certificate_passed`, `is_augmented`,
    `source_index`)
  - `scaler`: dict with `mean` and `std` (computed on train, to be reused
    identically at inference time on new matrices)
  - `config`: the `DatasetConfig` instance used to generate the dataset

## Current status / TODO

- [x] Generators for the 4 cases with numeric certificate and threshold
- [x] Symbolic derivation + feature extraction (9 real coefficients of L_A)
- [x] Augmentation via unitary conjugation, batched
- [x] Assembler with correct split (augmentation post-split, no leakage)
- [x] Standardization (fit on train, applied to all splits)
- [ ] **"Hard" pool (dedicated boundary/test set)**: not yet integrated.
      Requires defining `generator_hard_kwargs` in `DatasetConfig` (tighter
      thresholds to push matrices close to the boundary between classes).
      The pipeline is already set up to accommodate it (see `standardize`,
      which iterates over all keys of `splits`).
- [ ] PyTorch classifier (next work block)

## Quickstart

```python
from dataset_config import DatasetConfig
from dataset_builder import build_final_dataset, sanity_check, save_dataset

config = DatasetConfig()
dataset = build_final_dataset(config)
sanity_check(dataset)
save_dataset(dataset, "kippenhahn_dataset_v1")
```

See also `notebooks/dataset_smoke_test.ipynb` for a quick end-to-end test
on a reduced config.