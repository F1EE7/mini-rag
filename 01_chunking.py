"""01 · 文本切块（Chunking）

RAG 的第一步：把长文档切成小块（chunk）。切分策略直接决定检索质量的上限：

- 块太大：一段里混了多个主题，向量被"稀释"，检索不精准，prompt 也浪费 token；
- 块太小：上下文被切断，召回的片段单独读不懂；
- 边界问题：固定长度切分会把一句话从中间斩断——重叠（overlap）是常用缓解手段。

本模块实现三种基础策略（纯标准库）：

  fixed_chunks()      固定长度 + 可选重叠
  paragraph_chunks()  按空行分段，过短段落自动并入上一块
  chunk_text()        默认推荐策略：段落优先，超长段落再固定切分

运行演示：python 01_chunking.py
"""

import re


def fixed_chunks(text: str, size: int = 200, overlap: int = 0) -> list[str]:
    """按固定字符数切块。overlap > 0 时相邻块共享 overlap 个字符。"""
    if size <= 0:
        raise ValueError("size 必须为正数")
    if not 0 <= overlap < size:
        raise ValueError("overlap 必须满足 0 <= overlap < size")
    text = text.strip()
    if not text:
        return []
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def paragraph_chunks(text: str, min_size: int = 50) -> list[str]:
    """按空行分段；过短的段落并入上一块，避免产生碎片块。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged: list[str] = []
    for p in paras:
        if merged and len(merged[-1]) + len(p) + 1 <= min_size:
            merged[-1] += "\n" + p
        else:
            merged.append(p)
    return merged


def chunk_text(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """默认策略：先按段落聚合；超过 size 的段落再固定切分（带 overlap）。"""
    chunks: list[str] = []
    for para in paragraph_chunks(text, min_size=size // 3):
        if len(para) <= size:
            chunks.append(para)
        else:
            chunks.extend(fixed_chunks(para, size=size, overlap=overlap))
    return chunks


if __name__ == "__main__":
    demo = (
        "极限是微积分的基石。当自变量无限接近某个点时，函数值无限接近的数就是极限。\n\n"
        "导数刻画瞬时变化率。几何上它是切线的斜率，物理上它可以是瞬时速度。\n\n"
        "定积分刻画累积量。牛顿-莱布尼茨公式把定积分与原函数联系起来，使计算大为简化。"
    )
    for name, chunks in [
        ("fixed_chunks(size=60)", fixed_chunks(demo, 60)),
        ("paragraph_chunks()", paragraph_chunks(demo)),
        ("chunk_text(size=120, overlap=20)", chunk_text(demo, 120, 20)),
    ]:
        print(f"\n== {name}：共 {len(chunks)} 块 ==")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] {c[:28]}...（{len(c)} 字）")
