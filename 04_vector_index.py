"""04 · 向量索引：暴力检索是基准，IVF 是第一级加速

向量多了以后（百万级），和每一条都算相似度（暴力检索）就太慢了。
近似最近邻（ANN）索引的思路：放弃一点点精度，换来几十上百倍的速度。

IVF（倒排文件索引）是最容易理解的 ANN：

  1. 用 k-means 把全库向量聚成约 √N 个簇；
  2. 查询时只扫描离查询向量最近的 nprobe 个簇，而不是全库。

nprobe 是精度/速度的旋钮：nprobe=1 最快但可能漏掉真近邻；
nprobe=簇数 时退化为暴力检索（精度 100%）。

运行演示：python 04_vector_index.py（打印速度与召回率对比表）
"""

import time

import numpy as np


def kmeans(X: np.ndarray, k: int, iters: int = 5, seed: int = 0):
    """最朴素的 k-means：随机初始化 → 按距离分配 → 更新中心，重复 iters 轮。

    距离用欧氏距离，但用恒等式 ||x-c||² = ||x||² - 2x·c + ||c||² 展开——
    只靠矩阵乘法就能算出全部 N×k 距离，不用真的做三维广播（那会爆内存）。
    对归一化向量，最近欧氏距离等价于最大内积（||x||² 是每行常数）；
    对未归一化的向量，两者并不等价，按内积分配就不是标准 k-means 了。
    空簇保留旧中心，不更新。
    """
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    x_sq = (X ** 2).sum(axis=1)[:, None]
    assign = None
    for _ in range(iters):
        d2 = x_sq - 2 * (X @ centroids.T) + (centroids ** 2).sum(axis=1)[None, :]
        assign = np.argmin(d2, axis=1)
        for j in range(k):
            members = X[assign == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    d2 = x_sq - 2 * (X @ centroids.T) + (centroids ** 2).sum(axis=1)[None, :]
    assign = np.argmin(d2, axis=1)
    return centroids, assign


class BruteForceIndex:
    """暴力检索：全库内积，精度 100%，复杂度 O(N·d)。"""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, query: np.ndarray, k: int = 10) -> np.ndarray:
        scores = self.vectors @ query
        return np.argsort(-scores)[:k]


class IVFIndex:
    """倒排文件索引：k-means 分桶，查询只扫最近的 nprobe 个桶。"""

    def __init__(self, vectors: np.ndarray, n_clusters: int | None = None, seed: int = 0):
        n = len(vectors)
        k = n_clusters or max(1, int(np.sqrt(n)))
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.centroids, assign = kmeans(self.vectors, k, seed=seed)
        self.buckets = [np.where(assign == j)[0] for j in range(k)]

    def search(self, query: np.ndarray, k: int = 10, nprobe: int = 1) -> np.ndarray:
        probe = np.argsort(-(self.centroids @ query))[:nprobe]
        candidates = np.concatenate([self.buckets[j] for j in probe])
        scores = self.vectors[candidates] @ query
        top = np.argsort(-scores)[:k]
        return candidates[top]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, D, N_CLUSTERS = 50000, 64, 20
    # 真实 embedding 通常成簇（按主题聚集）。均匀随机数据没有簇结构，
    # 是 IVF 的最坏情况——那正是"为什么需要 ANN"要选对数据的原因。
    centers = rng.standard_normal((N_CLUSTERS, D)) * 4
    X = np.concatenate(
        [c + rng.standard_normal((N // N_CLUSTERS, D)) for c in centers]
    ).astype(np.float32)
    # 真实系统的 embedding 一般先归一化（配余弦相似度检索）。归一化后，
    # 最近欧氏距离与最大内积的排序一致，k-means 聚类和检索用的度量就对齐了
    # ——这正是 kmeans 注释里说的那个等价条件。
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    queries = X[rng.choice(N, size=20, replace=False)]  # 查询与库同分布（真实场景如此）

    brute = BruteForceIndex(X)
    ivf = IVFIndex(X)

    t0 = time.perf_counter()
    true_top = [brute.search(q, 10) for q in queries]
    t_brute = (time.perf_counter() - t0) * 1000

    print(f"库大小 N={N}，维度 D={D}，查询 20 条（向量已归一化）")
    print(f"暴力检索总耗时：{t_brute:.1f} ms")
    print(f"{'nprobe':>6} {'耗时(ms)':>10} {'recall@10':>10}")
    for nprobe in (1, 5, 10, 50):
        t0 = time.perf_counter()
        top = [ivf.search(q, 10, nprobe) for q in queries]
        dt = (time.perf_counter() - t0) * 1000
        recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top, true_top)])
        print(f"{nprobe:>6} {dt:>10.1f} {recall:>10.2f}")
