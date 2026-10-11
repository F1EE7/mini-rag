"""07 · 效果评测：不测量的优化都是玄学

改了 chunk 大小、换了检索器，到底有没有变好？需要一套小评测：

1. 评测集：若干（问题 → 相关块）对。本模块用"每段讲一件事"的语料，
   问题指向特定段落，相关块 = 与该段落范围有重叠的所有块；
2. 指标：
   - recall@k：前 k 个结果命中了多少比例的相关块；
   - MRR：第一个相关结果排名的倒数（排第 1 得 1 分，第 2 得 0.5…）；
3. 消融实验：固定其他条件，一次只变一个参数（chunk 大小 / 检索器）。

运行演示：python 07_evaluation.py（打印对比表；装了 matplotlib 会顺手出图）
"""

import importlib
import random

_chunking = importlib.import_module("01_chunking")
_sparse = importlib.import_module("02_sparse_search")
_hybrid = importlib.import_module("05_hybrid_search")

# 语料：每段只讲一个主题，便于构造"问题 → 相关段落"的评测对。
CORPUS_PARAS = [
    "极限描述的是这样的过程：当自变量 x 无限接近某个点时，函数值无限接近的那个确定的数，就称为函数在该点的极限。它是整个微积分大厦的地基。",
    "极限具有唯一性，即极限若存在则只能有一个。保号性则说：若某点处极限为正，则该点附近函数值也为正，这一性质常用于不等式证明。",
    "若函数在某点没有极限，或极限值不等于函数值，该点就是间断点。间断点分为第一类（可去间断点与跳跃间断点）和第二类（无穷间断点与振荡间断点）。",
    "导数是函数值增量与自变量增量之比的极限，几何意义是切线的斜率。可导必连续，但连续不一定可导，例如绝对值函数在原点连续却不可导。",
    "求导法则包括四则运算求导与复合函数求导。链式法则处理复合函数：外层函数的导数乘以内层函数的导数，像剥洋葱一样由外向内逐层求导。",
    "拉格朗日中值定理说：闭区间上连续、开区间内可导的函数，区间内必存在一点，其切线斜率等于两端点连线的斜率，故也称微分中值定理。",
    "不定积分是求导的逆运算：若 F 的导数是 f，则 F 是 f 的一个原函数，f 的全部原函数加上常数 C 就是 f 的不定积分。",
    "牛顿-莱布尼茨公式（微积分基本定理）指出：定积分的值等于原函数在上下限处取值之差。它把求面积、路程这类累积问题转化为求原函数。",
    "泰勒公式的核心思想是用多项式逼近复杂函数：在一点附近，用函数值与各阶导数构造多项式，阶数越高逼近越好，是近似计算的基础。",
]

FULL_TEXT = "\n\n".join(CORPUS_PARAS)

# 评测集：(问题, 相关段落编号)
QA_SET = [
    ("极限的定义是什么？", 0),
    ("极限的保号性说的是什么？", 1),
    ("间断点分为哪几类？", 2),
    ("导数的几何意义是什么？", 3),
    ("链式法则用来处理什么问题？", 4),
    ("拉格朗日中值定理的内容是什么？", 5),
    ("不定积分和导数是什么关系？", 6),
    ("牛顿-莱布尼茨公式有什么意义？", 7),
    ("泰勒公式的核心思想是什么？", 8),
]


def recall_at_k(ranked_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mrr(ranked_ids: list[int], relevant: set[int]) -> float:
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _para_spans() -> list[tuple[int, int]]:
    """每个段落在全文中的字符区间。"""
    spans, pos = [], 0
    for p in CORPUS_PARAS:
        start = FULL_TEXT.index(p, pos)
        spans.append((start, start + len(p)))
        pos = start + len(p)
    return spans


def _build_chunks(size: int):
    """全文定长切块，返回 (块文本列表, 每个段落对应的相关块集合)。"""
    chunks = _chunking.fixed_chunks(FULL_TEXT, size=size, overlap=0)
    chunk_spans = [(i * size, i * size + len(c)) for i, c in enumerate(chunks)]
    relevant = [
        {i for i, (cs, ce) in enumerate(chunk_spans) if cs < pe and ce > ps}
        for ps, pe in _para_spans()
    ]
    return chunks, relevant


def run_experiment(size: int, retriever: str, top_k: int = 3) -> dict:
    """在给定 chunk 大小与检索器下评测，返回平均 recall@k 与 MRR。

    retriever 额外支持 "random"：随机基线。没有基线的绝对数字是没有
    区分度的——语料切块数少到接近 top_k 时，随机排序也能拿高分。
    """
    chunks, relevant = _build_chunks(size)
    texts = [c for c in chunks]
    if retriever == "tfidf":
        index = _sparse.TfidfIndex().fit(texts)

        def search(q):
            return [i for i, _ in index.search(q, top_k)]
    elif retriever == "bm25":
        index = _hybrid.BM25Index().fit(texts)

        def search(q):
            return [i for i, _ in index.search(q, top_k)]
    elif retriever == "random":
        rng = random.Random(0)  # 固定种子，结果可复现

        def search(q):
            return rng.sample(range(len(texts)), min(top_k, len(texts)))
    else:
        raise ValueError(f"未知检索器：{retriever}")

    if len(texts) <= top_k:
        print(f"[警告] size={size} 时语料只切出 {len(texts)} 块（<= top_k={top_k}），"
              f"任何排序都必然命中，该行数字无区分度——请缩小 top_k 或加大语料。")

    recalls, rr = [], []
    for question, para_id in QA_SET:
        ranked = search(question)
        rel = relevant[para_id]
        recalls.append(recall_at_k(ranked, rel, top_k))
        rr.append(mrr(ranked, rel))
    return {"size": size, "retriever": retriever, "recall": sum(recalls) / len(recalls), "mrr": sum(rr) / len(rr)}


def run_ablation(sizes=(40, 80, 160, 320), retrievers=("tfidf", "bm25", "random"), top_k: int = 3) -> list[dict]:
    """消融实验：变 chunk 大小 × 检索器（含随机基线），输出对比表并尝试出图。"""
    rows = [run_experiment(s, r, top_k) for s in sizes for r in retrievers]

    print(f"\n| chunk 大小 | 检索器 | recall@{top_k} | MRR |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['size']} | {r['retriever']} | {r['recall']:.3f} | {r['mrr']:.3f} |")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pathlib

        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / len(retrievers)
        xs = range(len(sizes))
        for off, r in enumerate(retrievers):
            ys = [row["recall"] for row in rows if row["retriever"] == r]
            ax.bar([x + off * width for x in xs], ys, width, label=r)
        ax.set_xticks([x + width * (len(retrievers) - 1) / 2 for x in xs])
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("chunk 大小（字符）")
        ax.set_ylabel(f"recall@{top_k}")
        ax.set_title("chunk 大小 × 检索器 消融实验")
        ax.legend()
        fig.tight_layout()
        out = pathlib.Path(__file__).parent / "assets" / "eval.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=120)
        print(f"\n图表已保存：{out}")
    except ImportError:
        print("\n（未安装 matplotlib，跳过图表生成）")
    return rows


if __name__ == "__main__":
    run_ablation()
