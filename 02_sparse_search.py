"""02 · 稀疏检索：TF-IDF + 余弦相似度

有了块，下一步是"给一个问题，找出最相关的块"。最经典的起点是 TF-IDF：

- TF（词频）：一个词在某块里出现越多，越能代表这块；
- IDF（逆文档频率）：一个词在越多块里出现，越没有区分度（比如"的""是"）。

两者相乘，得到"对这块有代表性、又足够独特"的权重。检索时把问题和每个块
都表示成权重向量，夹角越小（余弦相似度越大）越相关。

中文没有空格，本模块用"字符二元组（bigram）"切词：把"机器学习"切成
"机器 / 器学 / 学习"。它不需要任何分词器依赖，是零成本的中文化方案。
注意 bigram 只在**连续中文串内部**生成，不跨标点、不跨中英文边界，
否则会产生"好。"、"ow" 这类噪声 token 撑大词表、稀释统计。

运行演示：python 02_sparse_search.py
"""

import re
from collections import Counter

import numpy as np


_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-z0-9]+")


def bigram_tokenize(text: str) -> list[str]:
    """中文按相邻两字切 bigram（仅限连续中文串内部）；英文与数字按整词保留。

    关键：英文整词要在**未去空白**的原文上提取——先把空白删掉再提取，
    会让 "hello world" 粘成 "helloworld" 只得到一个伪词，而不是
    "hello" 和 "world" 两个词。
    """
    low = text.lower()
    tokens: list[str] = []
    for run in _CJK_RUN.findall(low):
        if len(run) == 1:
            tokens.append(run)  # 单字成不了 bigram，保留单字本身
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    tokens.extend(_ASCII_WORD.findall(low))
    return tokens


class TfidfIndex:
    """TF-IDF 检索索引。fit 后用 search(query, k) 返回 [(块号, 相似度)]。"""

    def __init__(self) -> None:
        self.matrix = None  # 形状 (n_docs, n_terms)，每行已归一化

    def fit(self, docs: list[str]) -> "TfidfIndex":
        token_lists = [bigram_tokenize(d) for d in docs]
        vocab = sorted({t for toks in token_lists for t in toks})
        self.vocab = {t: i for i, t in enumerate(vocab)}

        tf = np.zeros((len(docs), len(vocab)))
        for i, toks in enumerate(token_lists):
            for term, n in Counter(toks).items():
                tf[i, self.vocab[term]] = n / len(toks)

        df = (tf > 0).sum(axis=0)
        self.idf = np.log(len(docs) / (df + 1e-9)) + 1.0  # +1 平滑，避免除零

        self.matrix = self._normalize(tf * self.idf)
        return self

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return m / norm

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = np.zeros(len(self.vocab))
        for term, n in Counter(bigram_tokenize(query)).items():
            if term in self.vocab:
                q[self.vocab[term]] = n
        q = self._normalize((q * self.idf).reshape(1, -1))[0]  # 查询侧同样 IDF 加权
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


if __name__ == "__main__":
    docs = [
        "泰勒公式用多项式逼近复杂函数，在一点附近展开。",
        "定积分计算曲边梯形的面积，牛顿-莱布尼茨公式联系积分与导数。",
        "足球比赛每队十一人，目标是把球射入对方球门。",
    ]
    index = TfidfIndex().fit(docs)
    for q in ["泰勒公式有什么用", "足球每队多少人", "定积分和导数的关系"]:
        i, s = index.search(q, k=1)[0]
        print(f"Q: {q}\n  -> [{i}] {docs[i][:24]}...  相似度={s:.3f}")
