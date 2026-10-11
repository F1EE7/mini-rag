"""01 · 文本切块（Chunking）

RAG 的第一步：把长文档切成小块（chunk）。切分策略直接决定检索质量的上限：

- 块太大：一段里混了多个主题，向量被"稀释"，检索不精准，prompt 也浪费 token；
- 块太小：上下文被切断，召回的片段单独读不懂；
- 边界问题：固定长度切分会把一句话从中间斩断——重叠（overlap）是常用缓解手段。

本模块实现三种基础策略（纯标准库）：

  fixed_chunks()      固定长度 + 可选重叠
  paragraph_chunks()  按空行分段，过短段落并入相邻块（标题并入其后正文）
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


_HEADING = re.compile(r"#{1,6}\s")


def paragraph_chunks(text: str, min_size: int = 50) -> list[str]:
    """按空行分段；过短的段落并入相邻块，避免产生碎片块。

    两条规则：
    1. 标题段落（以 # 开头）并入**紧随其后的正文**——标题标记的是它下面
       那一节，检索时应与该节内容一起被召回，而不是粘到上一节末尾；
    2. 其余过短段落并入相邻块：待并入的段落太短，或上一块仍然太短，都要
       合并。只要求"合并后仍小于阈值"是不够的——正文段落通常早已超过阈值，
       跟在后面的短标题会漏成孤立块，短块后面的正文同样会被拆散。
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    combined: list[str] = []
    i = 0
    while i < len(paras):
        if _HEADING.match(paras[i]) and i + 1 < len(paras):
            combined.append(paras[i] + "\n" + paras[i + 1])
            i += 2
        else:
            combined.append(paras[i])
            i += 1
    merged: list[str] = []
    for p in combined:
        if merged and (len(p) < min_size or len(merged[-1]) < min_size):
            merged[-1] += "\n" + p
        else:
            merged.append(p)
    return merged


def chunk_text(text: str, size: int = 300, overlap: int = 50, min_para: int = 50) -> list[str]:
    """默认策略：先按段落聚合；超过 size 的段落再固定切分（带 overlap）。

    min_para 是段落合并阈值：短于它的段落视为碎片，并入相邻块。
    调大它块会更整但更粗（可能把正常段落也并掉、打散段落边界）；
    调小则保留更细的段落粒度。
    """
    chunks: list[str] = []
    for para in paragraph_chunks(text, min_size=min_para):
        if len(para) <= size:
            chunks.append(para)
        else:
            chunks.extend(fixed_chunks(para, size=size, overlap=overlap))
    return chunks


if __name__ == "__main__":
    demo = (
        "# 微积分 · 核心概念\n\n"
        "极限是微积分的基石。当自变量无限接近某个点时，函数值无限接近的那个确定的数就是极限，它是整个微积分大厦的地基。\n\n"
        "导数刻画瞬时变化率。几何上它是切线的斜率，物理上它可以是瞬时速度，链式法则是求导法则中最常用的一个。\n\n"
        "以上就是本章的三个核心概念。\n\n"
        "## 定积分\n\n"
        "定积分刻画累积量。牛顿-莱布尼茨公式把定积分与原函数联系起来，使面积、路程这类累积问题的计算大为简化。"
    )
    for name, chunks in [
        ("fixed_chunks(size=60)", fixed_chunks(demo, 60)),
        ("paragraph_chunks()", paragraph_chunks(demo)),
        ("chunk_text(size=120, overlap=20)", chunk_text(demo, 120, 20)),
    ]:
        print(f"\n== {name}：共 {len(chunks)} 块 ==")
        for i, c in enumerate(chunks, 1):
            preview = c.replace("\n", "⏎")[:30]
            print(f"  [{i}] {preview}…（{len(c)} 字）")
