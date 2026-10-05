"""03 · 稠密向量：从共现矩阵到词向量（PPMI + SVD）

TF-IDF 的盲点：它认为"复习"和"温习"毫无关系，因为两者字面不相交。
稠密向量（embedding）的思路相反：一个词的含义，由它周围的词决定。

三步从零得到词向量（全部只用 numpy）：

  1. 统计共现：窗口内一起出现的词对计数，得到共现矩阵 C；
  2. PPMI：把计数换成"正逐点互信息"，只保留真正超出随机共现的关联；
  3. SVD：对 PPMI 矩阵做奇异值分解，取前 dim 维——每个词变成稠密短向量。

这样"复习"和"温习"因为常出现在相似上下文里，向量会很接近——
这就是语义检索能比关键词检索聪明的根本原因。

运行演示：python 03_embeddings.py
"""

import numpy as np


def cooccurrence_matrix(docs: list[list[str]], window: int = 4) -> tuple[np.ndarray, dict]:
    """统计窗口共现矩阵。docs 是分好词的文档（每个元素是词列表）。"""
    vocab = sorted({w for toks in docs for w in toks})
    word2id = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)))
    for toks in docs:
        ids = [word2id[w] for w in toks]
        for pos, wi in enumerate(ids):
            for offset in range(1, window + 1):
                for j in (pos - offset, pos + offset):
                    if 0 <= j < len(ids):
                        C[wi, ids[j]] += 1.0
    C = C + C.T  # 共现是对称关系
    return C, word2id


def ppmi(C: np.ndarray) -> np.ndarray:
    """正逐点互信息。PMI = log( P(w,c) / (P(w)·P(c)) )，负值截为 0。"""
    total = C.sum()
    p_word = C.sum(axis=1, keepdims=True) / total
    p_context = C.sum(axis=0, keepdims=True) / total
    p_joint = C / total
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.log(p_joint / (p_word * p_context + 1e-12))
    m[~np.isfinite(m)] = 0.0
    return np.maximum(m, 0.0)


def svd_embed(P: np.ndarray, dim: int = 32) -> np.ndarray:
    """SVD 降维：U·sqrt(S) 取前 dim 维，行归一化后即为词向量。"""
    dim = min(dim, P.shape[0])
    U, S, _ = np.linalg.svd(P)
    E = U[:, :dim] * np.sqrt(S[:dim])
    norm = np.linalg.norm(E, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return E / norm


class PPMIEmbedder:
    """fit(分词文档) 后：transform(word) 取词向量，doc_vector 取块向量。"""

    def __init__(self, dim: int = 32, window: int = 4):
        self.dim = dim
        self.window = window

    def fit(self, docs: list[list[str]]) -> "PPMIEmbedder":
        C, self.word2id = cooccurrence_matrix(docs, self.window)
        self.id2word = list(self.word2id)  # word2id 按 sorted 插入，顺序即行号
        self.vectors = svd_embed(ppmi(C), self.dim)
        return self

    def transform(self, word: str) -> np.ndarray:
        i = self.word2id.get(word)
        return self.vectors[i] if i is not None else np.zeros(self.vectors.shape[1])

    def doc_vector(self, tokens: list[str]) -> np.ndarray:
        """块/句向量 = 词向量的平均（词袋假设，教程够用）。"""
        vecs = [self.transform(w) for w in tokens if w in self.word2id]
        if not vecs:
            return np.zeros(self.vectors.shape[1])
        v = np.mean(vecs, axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v

    def nearest(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        v = self.transform(word)
        sims = self.vectors @ v
        out = []
        for i in np.argsort(-sims):
            w = self.id2word[int(i)]
            if w != word:
                out.append((w, float(sims[i])))
            if len(out) == k:
                break
        return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


if __name__ == "__main__":
    study = [
        ["我", "复习", "数学", "知识"],
        ["他", "温习", "数学", "知识"],
        ["我", "复习", "笔记", "准备", "考试"],
        ["他", "温习", "笔记", "准备", "考试"],
        ["复习", "和", "温习", "帮助", "记忆"],
        ["老师", "要求", "复习", "重点"],
        ["老师", "要求", "温习", "重点"],
        ["温习", "旧", "知识", "非常", "重要"],
        ["她", "复习", "课文", "然后", "做", "练习"],
    ]
    sport = [
        ["我们", "踢", "足球", "在", "操场"],
        ["足球", "是", "流行", "的", "运动"],
        ["他", "喜欢", "篮球", "和", "足球"],
        ["运动", "以后", "我", "非常", "开心"],
        ["操场", "上", "有", "人", "在", "踢", "足球"],
        ["篮球", "比赛", "非常", "精彩"],
        ["周末", "我们", "打", "篮球"],
    ]
    emb = PPMIEmbedder().fit((study + sport) * 6)
    fmt = lambda pairs: ", ".join(f"{w}({s:.3f})" for w, s in pairs)
    print("与「复习」最相近的词（同义词温习排第一；其余是上下文共现词）：")
    print("  ", fmt(emb.nearest("复习", 5)))
    print("与「足球」最相近的词：", fmt(emb.nearest("足球", 4)))
    print(f"sim(复习, 温习) = {cosine(emb.transform('复习'), emb.transform('温习')):.3f}")
    print(f"sim(复习, 足球) = {cosine(emb.transform('复习'), emb.transform('足球')):.3f}")
