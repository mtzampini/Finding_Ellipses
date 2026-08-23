import numpy as np
from generator import gen_case1, gen_case2, gen_case3, gen_case4
from feature_extraction import feature_extractor
from dataset_config import DatasetConfig
from sklearn.model_selection import train_test_split

def random_orthonormal_basis_batch(N, n=3):
    A = np.random.randn(N, n, n) + 1j * np.random.randn(N, n, n)
    Q, R = np.linalg.qr(A)  
    d = np.diagonal(R, axis1=1, axis2=2) 
    d = d / np.abs(d)
    return Q * d[:, None, :]  

def augmentation(A_batch, N):
    U = random_orthonormal_basis_batch(N)
    A_aug = U @ A_batch @ U.conj().transpose(0, 2, 1)
    return A_aug

def generate_case_pool(case_id, n, generator_fn, generator_kwargs):
    results = []

    for _ in range(n):
        
        kwargs = dict(generator_kwargs)

        if case_id == 2:
            kwargs['inside'] = np.random.rand() < 0.5

        result = generator_fn(**kwargs)

        if result['case'] != case_id:
            raise ValueError(f'returned {result['case']}, expected {case_id}')
        
        results.append(result)

    A_batch = np.stack([r['matrix'] for r in results], axis=0)

    metadata = [
        {
            "case": r["case"],
            "eigenvalues": r["eigenvalues"],
            "certificate": r["certificate"],
            "certificate_threshold": r["certificate_threshold"],
            "certificate_passed": r["certificate_passed"],
            "is_augmented": False,      
            "source_index": i,          
        }
        for i, r in enumerate(results)
    ]
    return A_batch, metadata

def augment_pool(A_batch, metadata, k, include_original):
    n = A_batch.shape[0]
    pieces_A = []
    pieces_meta = []
    if include_original:
        pieces_A.append(A_batch)
        pieces_meta.extend(
            {**metadata[i], "is_augmented": False, "source_index": i}
            for i in range(n)
        )
    if k > 0:
        A_repeated = np.repeat(A_batch, repeats=k, axis=0)
        U = random_orthonormal_basis_batch(A_repeated.shape[0])
        A_augmented = U @ A_repeated @ U.conj().transpose(0, 2, 1)

        pieces_A.append(A_augmented)
        pieces_meta.extend(
            {**metadata[i], "is_augmented": True, "source_index": i}
            for i in range(n)
            for _ in range(k)
        )
    
    A_out = np.concatenate(pieces_A, axis=0)
    metadata_out = pieces_meta
    return A_out, metadata_out

def build_base_pool(config: DatasetConfig):
    np.random.seed(config.seed)
    generator_map = {
        1: gen_case1,
        2: gen_case2,
        3: gen_case3,
        4: gen_case4,
    }
    n_per_case = config.n_per_case()

    pieces_A = []
    pieces_labels = []
    pieces_meta = []

    for case_id in (1, 2, 3, 4):
        n = n_per_case[case_id]
        generator_fn = generator_map[case_id]
        kwargs = config.generator_kwargs[case_id]

        A_base, meta_base = generate_case_pool(case_id, n, generator_fn, kwargs)
        labels_base = np.full(A_base.shape[0], case_id, dtype=int)

        pieces_A.append(A_base)
        pieces_labels.append(labels_base)
        pieces_meta.extend(meta_base)

    A_all = np.concatenate(pieces_A, axis=0)
    labels_all = np.concatenate(pieces_labels, axis=0)
    metadata_all = pieces_meta

    return A_all, labels_all, metadata_all

def split_base_pool(A_all, labels_all, metadata_all, config: DatasetConfig):

    indices = np.arange(A_all.shape[0])

    idx_train, idx_val = train_test_split(
        indices, train_size=config.train_fraction,
        stratify=labels_all,
        random_state=config.seed,
        shuffle=True,
    )

    def _subset(idx):
        return {
            "A": A_all[idx],
            "y": labels_all[idx],
            "metadata": [metadata_all[i] for i in idx],
        }

    return {
        "train": _subset(idx_train),
        "val": _subset(idx_val),
    }

def augment_train_split(splits, config: DatasetConfig):

    A_train, y_train, meta_train = splits["train"]["A"], splits["train"]["y"], splits["train"]["metadata"]

    pieces_A, pieces_labels, pieces_meta = [], [], []

    for case_id in (1, 2, 3, 4):
        mask = (y_train == case_id)
        A_case = A_train[mask]
        meta_case = [m for m, keep in zip(meta_train, mask) if keep]
        k = config.n_augment_per_case[case_id]

        A_aug, meta_aug = augment_pool(A_case, meta_case, k, include_original=True)
        labels_aug = np.full(A_aug.shape[0], case_id, dtype=int)

        pieces_A.append(A_aug)
        pieces_labels.append(labels_aug)
        pieces_meta.extend(meta_aug)

    splits["train"]["A"] = np.concatenate(pieces_A, axis=0)
    splits["train"]["y"] = np.concatenate(pieces_labels, axis=0)
    splits["train"]["metadata"] = pieces_meta

    return splits

def extract_features_for_splits(splits):
    for split_name in splits:
        A = splits[split_name].pop("A")
        splits[split_name]["X"] = feature_extractor(A)

    return splits

def standardize(splits):
    X_train = splits["train"]["X"]
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.maximum(std, 1e-12)

    for split_name in splits:
        splits[split_name]["X"] = (splits[split_name]["X"] - mean) / std

    return {"mean": mean, "std": std}

def build_final_dataset(config):
    A_all, labels_all, metadata_all = build_base_pool(config)
    splits = split_base_pool(A_all, labels_all, metadata_all, config)
    splits = augment_train_split(splits, config)
    splits = extract_features_for_splits(splits)

    scaler_params = standardize(splits)

    dataset = {
        'train': splits['train'],
        'val': splits['val'],
        "scaler": scaler_params,
        "config": config,        
    }

    return dataset

def sanity_check(dataset):
    for split_name in ("train", "val"):
        X, y, meta = dataset[split_name]["X"], dataset[split_name]["y"], dataset[split_name]["metadata"]

        assert X.shape[0] == y.shape[0] == len(meta), \
            f"{split_name}: disaligned lenghts X={X.shape[0]}, y={y.shape[0]}, meta={len(meta)}"
        assert not np.isnan(X).any(), f"{split_name}: NaN found"
        assert not np.isinf(X).any(), f"{split_name}: inf found"
        assert X.shape[1] == 9, f"{split_name}: expected 9 features, found {X.shape[1]}"

import pickle

def save_dataset(dataset, path_prefix):
    np.savez(
        f"{path_prefix}_arrays.npz",
        X_train=dataset["train"]["X"],
        y_train=dataset["train"]["y"],
        X_val=dataset["val"]["X"],
        y_val=dataset["val"]["y"],
    )

    with open(f"{path_prefix}_meta.pkl", "wb") as f:
        pickle.dump({
            "metadata_train": dataset["train"]["metadata"],
            "metadata_val": dataset["val"]["metadata"],
            "scaler": dataset["scaler"],
            "config": dataset["config"],
        }, f)

    print(f"Salvato: {path_prefix}_arrays.npz, {path_prefix}_meta.pkl")

