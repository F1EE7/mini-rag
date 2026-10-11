"""05 · 混合检索：BM25 + 稠密向量，RRF 融合

TF-IDF 有个升级版叫 BM25：加入词频饱和（一个词出现 10 次和 100 次差别不大）
和文档长度惩罚。实践中 BM25 几乎总是优于 TF-IDF，是稀疏检索的事实标准。

但真正强的系统是"稀疏 + 稠密"两路并用：

- 稀疏（BM25）擅长：精确的词、编号、专有名词——"牛顿-莱布尼茨公式"必须字面命中；
- 稠密（向量）擅长：换了说法的语义匹配——"怎么求原函数" 对应 "不定积分"。

融合两路排序最简单有效的方法是 RRF（Reciprocal Rank Fusion）：
每个文档得分 = Σ 1/(k + rank)。只看名次不看原始分数，天然免去两路分数
量纲不统一的问题。

诚实提醒：本模块的稠密路是 PPMI+SVD 词向量的**教学演示**。它的 doc_vector
只是词向量平均，小语料（几百个块以内）上噪声可能压过信号——本文件的演示
里它就把数学文档排进了"球类运动"的结果。真实系统的稠密路应换成预训练
多语种 embedding（见 app.py 的 sentence-transformers 可选增强）。

小技巧：本仓库文件名以数字开头，不能用 `import 03_embeddings` 语句，
所以统一用 importlib 按名字加载。

运行演示：python 05_hybrid_search.py
"""

import importlib
import math
from collections import Counter, defaultdict

import numpy as np

_sparse = importlib.import_module("02_sparse_search")
bigram_tokenize = _sparse.bigram_tokenize


class BM25Index:
    """BM25 检索。k1 控制词频饱和速度，b 控制文档长度惩罚强度。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[str]) -> "BM25Index":
        self.toklists = [bigram_tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.toklists) / max(self.N, 1)
        self.tfs = [Counter(t) for t in self.toklists]
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q_terms = set(bigram_tokenize(query))
        results = []
        for i in range(self.N):
            dl, score = len(self.toklists[i]), 0.0
            for t in q_terms:
                tf = self.tfs[i].get(t, 0)
                if tf == 0 or t not in self.idf:
                    continue
                score += self.idf[t] * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            if score > 0:
                results.append((i, score))
        results.sort(key=lambda x: -x[1])
        return results[:k]


def rrf_fuse(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """RRF 融合多路排序，返回按融合分从高到低的文档 id。"""
    scores: defaultdict = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])


class HybridSearch:
    """稀疏（BM25）+ 稠密（PPMI 词向量）两路检索，RRF 融合。"""

    def __init__(self, dense_top: int = 20, sparse_top: int = 20, rrf_k: int = 60):
        self.dense_top, self.sparse_top, self.rrf_k = dense_top, sparse_top, rrf_k

    def fit(self, docs: list[str]) -> "HybridSearch":
        self.docs = docs
        self.sparse = BM25Index().fit(docs)
        embeddings = importlib.import_module("03_embeddings")
        tokenized = [bigram_tokenize(d) for d in docs]  # bigram 当"词"用
        self.embedder = embeddings.PPMIEmbedder(dim=16).fit(tokenized)
        self.doc_vecs = np.stack([self.embedder.doc_vector(t) for t in tokenized])
        return self

    def search(self, query: str, k: int = 5) -> list[int]:
        sparse_rank = [i for i, _ in self.sparse.search(query, self.sparse_top)]
        qv = self.embedder.doc_vector(bigram_tokenize(query))
        dense_rank = list(np.argsort(-(self.doc_vecs @ qv))[: self.dense_top])
        return rrf_fuse([sparse_rank, dense_rank], self.rrf_k)[:k]


if __name__ == "__main__":
    docs = [
        "泰勒公式用多项式逼近复杂函数，是近似计算的重要工具。",
        "不定积分是求导的逆运算，求原函数的过程就叫积分。",
        "牛顿-莱布尼茨公式把定积分与原函数联系起来。",
        "拉格朗日中值定理连接了切线斜率与割线斜率。",
        "足球比赛每队十一人，在球场上进行对抗。",
        "篮球运动需要手感和团队配合。",
    ]
    system = HybridSearch().fit(docs)
    for q in ["牛顿-莱布尼茨公式", "如何求原函数", "球类运动"]:
        sparse_rank = [i for i, _ in system.sparse.search(q, 3)]
        qv = system.embedder.doc_vector(bigram_tokenize(q))
        dense_rank = list(np.argsort(-(system.doc_vecs @ qv))[:3])
        fused = system.search(q, 3)
        fmt = lambda rank: " ".join(f"{i}:{docs[i][:8]}…" for i in rank)
        print(f"Q: {q}\n  BM25 ：{fmt(sparse_rank)}\n  稠密 ：{fmt(dense_rank)}\n  融合 ：{fmt(fused)}")
