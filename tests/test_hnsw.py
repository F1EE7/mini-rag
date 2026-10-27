import importlib

import numpy as np

hnsw_mod = importlib.import_module("08_hnsw")


def build_clustered(n_per=100, dim=16, n_clusters=8, seed=0):
    """高斯簇 + 归一化，贴近真实 embedding 的分布。"""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 4
    X = np.concatenate([c + rng.standard_normal((n_per, dim)) for c in centers])
    X = X.astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_hnsw_exact_with_large_ef():
    """ef 足够大时，束搜索退化为足够宽的遍历，应与暴力结果一致。"""
    X = build_clustered()
    index = hnsw_mod.HNSWIndex(M=8, ef_construction=128, seed=0).build(X)
    brute = np.argsort(-(X @ X[0]))[:10]
    got = index.search(X[0], 10, ef=256)
    assert len(got) == 10
    assert set(got) == set(brute.tolist())


def test_hnsw_recall_high_and_ef_helps():
    """簇状数据上：ef=64 应接近暴力；ef 增大召回应不降。"""
    X = build_clustered(seed=1)
    index = hnsw_mod.HNSWIndex(M=8, ef_construction=64, seed=1).build(X)
    rng = np.random.default_rng(2)
    queries = X[rng.choice(len(X), 20, replace=False)]
    brute = [np.argsort(-(X @ q))[:10] for q in queries]

    def recall(ef):
        top = [index.search(q, 10, ef) for q in queries]
        return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top, brute)])

    r16, r64 = recall(16), recall(64)
    assert r64 >= 0.85
    assert r64 >= r16


def test_hnsw_deterministic():
    X = build_clustered(seed=3)
    q = X[5]
    a = hnsw_mod.HNSWIndex(M=8, seed=3).build(X).search(q, 10)
    b = hnsw_mod.HNSWIndex(M=8, seed=3).build(X).search(q, 10)
    assert a == b


def test_hnsw_links_bounded_by_m():
    X = build_clustered(n_per=40, seed=4)
    index = hnsw_mod.HNSWIndex(M=6, ef_construction=32, seed=4).build(X)
    for node_layers in index.links:
        for neighbors in node_layers:
            assert len(neighbors) <= 6


def test_hnsw_k_larger_than_results():
    """k 大于返回候选数时不崩溃，返回全部候选。"""
    X = build_clustered(n_per=10, seed=5)
    index = hnsw_mod.HNSWIndex(seed=5).build(X)
    got = index.search(X[0], k=10, ef=8)
    assert 0 < len(got) <= 10
