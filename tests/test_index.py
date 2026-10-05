import importlib

import numpy as np

index_mod = importlib.import_module("04_vector_index")


def build_clustered(n_per=50, dim=16, n_clusters=6, seed=0):
    """生成若干高斯簇，模拟真实向量的"块状"分布。"""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 8
    X = np.concatenate([c + rng.standard_normal((n_per, dim)) for c in centers])
    return X.astype(np.float32)


def test_brute_force_is_ground_truth():
    X = build_clustered()
    brute = index_mod.BruteForceIndex(X)
    top = brute.search(X[0], 5)
    scores = X @ X[0]
    expected = set(np.argsort(-scores)[:5].tolist())
    assert set(top.tolist()) == expected


def test_ivf_exact_when_full_probe():
    X = build_clustered(seed=1)
    ivf = index_mod.IVFIndex(X, n_clusters=10, seed=1)
    brute = index_mod.BruteForceIndex(X)
    assert np.array_equal(ivf.search(X[7], 5, nprobe=10), brute.search(X[7], 5))


def test_ivf_high_recall_with_small_probe():
    X = build_clustered(seed=2)
    ivf = index_mod.IVFIndex(X, n_clusters=10, seed=2)
    brute = index_mod.BruteForceIndex(X)
    rng = np.random.default_rng(3)
    recalls = []
    for qi in rng.choice(len(X), 10, replace=False):
        q = X[qi]
        hit = len(set(ivf.search(q, 10, nprobe=3)) & set(brute.search(q, 10))) / 10
        recalls.append(hit)
    assert np.mean(recalls) >= 0.7


def test_kmeans_deterministic():
    X = build_clustered(seed=4)
    c1, a1 = index_mod.kmeans(X, 6, seed=4)
    c2, a2 = index_mod.kmeans(X, 6, seed=4)
    assert np.array_equal(c1, c2) and np.array_equal(a1, a2)
