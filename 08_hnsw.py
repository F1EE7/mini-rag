"""08 · HNSW：跳表思想的分层图近似最近邻

IVF 只有一个旋钮（nprobe）；HNSW（分层可导航小世界图）把"跳表"搬进
向量检索：建图时每个点以递减概率被提升到更高层——顶层稀疏、只剩"长边"，
底层稠密、覆盖全部点。查询从顶层入口出发：

  1. 上层用贪心搜索（每步跳向更近的邻居）快速锁定区域；
  2. 每下降一层，用 ef 大小的束搜索（保留 ef 个候选同时展开）精化；
  3. 第 0 层跑完，候选队列里就是最终近邻。

两个旋钮：
  M          每个点每层最多连几条边（图密度：影响召回、内存、建图时间）
  ef_search  查询候选队列长度（精度旋钮：越大越准越慢）

贪心 vs 束搜索的分工值得记住：贪心快，但会困在"局部极小"——附近的点
都比当前位置远，其实更近的点在别的簇里；束搜索用 ef 个候选并行推进，
用一定冗余换全局性。上层要快所以贪心，底层要准所以束搜索。

本实现带论文的启发式选边（Algorithm 4）。为什么必须用它？我们实测过朴素
"取最近 M 个"：簇状数据上后插入的点全是同簇近邻，跨簇长边不断被挤出
邻居表，图被剪成互不连通的孤岛——8 簇数据上 recall@10 只有 0.45。
启发式的核心思想：候选 e 只在"e 到查询点比到任何已选邻居更近"时才收
（没人"覆盖"这个方向才收），簇间因此保住少量长边，图保持可导航。
"""

import heapq
import importlib

import numpy as np


class HNSWIndex:
    """从零实现的 HNSW。距离用欧氏距离平方（单调，排序与欧氏等价）。"""

    def __init__(self, M: int = 8, ef_construction: int = 64, seed: int = 0):
        self.M = M
        self.efc = ef_construction
        self.p = 1.0 / M
        self.seed = seed

    def build(self, vectors: np.ndarray) -> "HNSWIndex":
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.rng = np.random.default_rng(self.seed)
        self.links: list[list[list[int]]] = []  # links[i][层号] = 邻居 id 列表
        self.entry: int | None = None
        self.max_level = -1
        for i in range(len(self.vectors)):
            self._insert(i)
        return self

    def _dist(self, idx: int, q: np.ndarray) -> float:
        d = self.vectors[idx] - q
        return float(d @ d)

    def _dists_to(self, ids, q: np.ndarray) -> np.ndarray:
        """批量算一组点到 q 的距离平方。按跳批量向量化是纯 Python 实现
        HNSW 的关键工程手段——逐点调 numpy 的固定开销会吃掉全部收益。"""
        diff = self.vectors[ids] - q
        return (diff * diff).sum(axis=1)

    def _level(self) -> int:
        """与跳表一致：以 p=1/M 的概率逐层上升，层数期望 O(log N)。"""
        layer = 0
        while self.rng.random() < self.p and layer < 16:
            layer += 1
        return layer

    def _select_neighbors(self, q: np.ndarray, cands, M: int):
        """启发式选边（论文 Algorithm 4 简化版）。

        候选按距离升序遍历；只有当 e 到 q 比到任何已选邻居都近时才收——
        说明已选邻居没有"覆盖"e 这个方向。没有这条规则，簇状数据上
        跨簇长边会被同簇点不断挤出，图退化成互不连通的孤岛（实测
        recall@10 从 0.45 修到 0.9+ 的关键）。
        """
        selected: list[tuple[float, int]] = []
        for d, e in cands:
            if len(selected) >= M:
                break
            ev = self.vectors[e]
            covered = any(
                float((ev - self.vectors[s]) @ (ev - self.vectors[s])) < d
                for _, s in selected
            )
            if not covered:
                selected.append((d, e))
        if len(selected) < M:  # keepPrunedConnections：不满 M 个就用最近的补齐
            chosen = {e for _, e in selected}
            for d, e in cands:
                if len(selected) >= M:
                    break
                if e not in chosen:
                    selected.append((d, e))
                    chosen.add(e)
        return selected

    def _insert(self, i: int) -> None:
        level = self._level()
        self.links.append([[] for _ in range(level + 1)])
        if self.entry is None:
            self.entry, self.max_level = i, level
            return

        q = self.vectors[i]
        ep, ep_dist = self.entry, self._dist(self.entry, q)
        # 高于新点层级的各层：只贪心下降（不建边）
        for layer in range(self.max_level, level, -1):
            ep, ep_dist = self._greedy(q, ep, ep_dist, layer)
        # 从 min(level, max_level) 层到 0 层：束搜索 + 建边
        eps = [(ep_dist, ep)]
        for layer in range(min(level, self.max_level), -1, -1):
            cands = self._search_layer(q, eps, self.efc, layer)
            self.links[i][layer] = [e for _, e in self._select_neighbors(q, cands, self.M)]
            for _, j in cands[: self.M]:
                self.links[j][layer].append(i)
                if len(self.links[j][layer]) > self.M:
                    # 溢出裁剪也用启发式重选，而不是"最近 M 个"——后者会把
                    # 出向跨簇边定向剪掉，图变成有向孤岛（实测从入口只能
                    # 到达 174/800 个点，查询永远困在错误的簇里）。
                    jq = self.vectors[j]
                    nbrs = self.links[j][layer]
                    cand2 = sorted(
                        (float((self.vectors[t] - jq) @ (self.vectors[t] - jq)), t)
                        for t in nbrs
                    )
                    self.links[j][layer] = [
                        e for _, e in self._select_neighbors(jq, cand2, self.M)
                    ]
            eps = cands  # 本层结果作为下一层的入口集
        if level > self.max_level:
            self.entry, self.max_level = i, level

    def _greedy(self, q: np.ndarray, ep: int, ep_dist: float, layer: int):
        """贪心：反复跳向更近的邻居，直到没有更近的为止。"""
        while True:
            neighbors = self.links[ep][layer]
            if not neighbors:
                return ep, ep_dist
            dists = self._dists_to(neighbors, q)
            best = int(np.argmin(dists))
            if dists[best] < ep_dist:
                ep, ep_dist = neighbors[best], float(dists[best])
            else:
                return ep, ep_dist

    def _search_layer(self, q, eps, ef, layer):
        """ef-束搜索。eps 是 [(距离, id)] 入口集，返回按距离升序的候选。"""
        visited = {e for _, e in eps}
        cand = list(eps)  # 小顶堆：待扩展候选
        heapq.heapify(cand)
        result = [(-d, e) for d, e in eps]  # 堆顶 = 当前最差候选
        heapq.heapify(result)
        while cand:
            d, c = heapq.heappop(cand)
            if len(result) >= ef and d > -result[0][0]:
                break  # 最近的待扩展点都比已有最差候选远：不可能再有改进
            fresh = [nb for nb in self.links[c][layer] if nb not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            dists = self._dists_to(fresh, q)
            for dnb, nb in zip(dists, fresh):
                dnb = float(dnb)
                if len(result) < ef or dnb < -result[0][0]:
                    heapq.heappush(cand, (dnb, nb))
                    heapq.heappush(result, (-dnb, nb))
                    if len(result) > ef:
                        heapq.heappop(result)
        return sorted((-nd, e) for nd, e in result)

    def search(self, query, k: int = 10, ef: int | None = None) -> list[int]:
        q = np.asarray(query, dtype=np.float32)
        ef = max(ef or self.efc, k)
        ep, ep_dist = self.entry, self._dist(self.entry, q)
        for layer in range(self.max_level, 0, -1):
            ep, ep_dist = self._greedy(q, ep, ep_dist, layer)
        res = self._search_layer(q, [(ep_dist, ep)], ef, 0)
        return [e for _, e in res[:k]]


if __name__ == "__main__":
    import time

    _ivf_mod = importlib.import_module("04_vector_index")

    rng = np.random.default_rng(0)
    N, D, N_CLUSTERS = 10000, 32, 20
    centers = rng.standard_normal((N_CLUSTERS, D)) * 4
    X = np.concatenate(
        [c + rng.standard_normal((N // N_CLUSTERS, D)) for c in centers]
    ).astype(np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # 归一化：贴近真实 embedding
    queries = X[rng.choice(N, size=20, replace=False)]

    brute = _ivf_mod.BruteForceIndex(X)
    ivf = _ivf_mod.IVFIndex(X, n_clusters=50)
    t0 = time.perf_counter()
    hnsw = HNSWIndex(M=8, ef_construction=64).build(X)
    t_build = time.perf_counter() - t0
    true_top = [brute.search(q, 10) for q in queries]

    t0 = time.perf_counter()
    _ = [brute.search(q, 10) for q in queries]
    t_brute = (time.perf_counter() - t0) * 1000

    print(f"库大小 N={N}，维度 D={D}，查询 20 条（HNSW 建图 {t_build:.1f}s）")
    print(f"暴力检索总耗时：{t_brute:.1f} ms")
    print(f"{'方法':<10} {'参数':<12} {'耗时(ms)':>9} {'recall@10':>10}")
    for nprobe in (1, 5, 10):
        t0 = time.perf_counter()
        top = [ivf.search(q, 10, nprobe) for q in queries]
        dt = (time.perf_counter() - t0) * 1000
        rec = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top, true_top)])
        print(f"{'IVF':<10} {'nprobe=' + str(nprobe):<12} {dt:>9.1f} {rec:>10.2f}")
    for ef in (8, 32, 64):
        t0 = time.perf_counter()
        top = [hnsw.search(q, 10, ef) for q in queries]
        dt = (time.perf_counter() - t0) * 1000
        rec = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top, true_top)])
        print(f"{'HNSW':<10} {'ef=' + str(ef):<12} {dt:>9.1f} {rec:>10.2f}")
