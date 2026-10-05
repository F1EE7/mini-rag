# mini-rag：从零实现一个迷你 RAG

> 不用 LangChain，不用 LlamaIndex——只用 Python 标准库和 numpy，
> 亲手实现检索增强生成（RAG）的每一层。

<!-- TODO(W8): 在这里放演示 GIF：python app.py ask "泰勒公式什么时候不能用？" 的录屏，带 [1][2] 引用输出 -->

## 这是什么

7 个逐步递进的 Python 文件，每个文件独立可运行、配一篇中文讲解。
按顺序读完，你将亲手实现：

```
文本切块 → TF-IDF 检索 → 词向量 → 向量索引
        → 混合检索 → 带引用的问答 → 效果评测
```

最终得到一个能对自己本地文档提问、答案可溯源的命令行工具。

## 为什么从零写

- **框架会魔术**：`VectorStore.from_documents()` 一行跑通，但面试官问"为什么 chunk 不能太大"就露馅。这里每一层都是你写的。
- **只用 numpy**：全仓库核心代码零第三方依赖（LLM API 是可选增强），任何机器 clone 下来就能跑。
- **有实验数据**：每个模块附带真实消融实验（chunk 大小、bigram vs 单字、暴力 vs 索引），不只讲 what，还讲 why。

## 课程目录

| # | 文件 | 内容 | 你会学到 |
|---|------|------|----------|
| 1 | `01_chunking.py` | 文本切块 | chunk 大小/overlap 的取舍 |
| 2 | `02_sparse_search.py` | TF-IDF + 余弦相似度 | 倒排索引、中文 bigram 分词 |
| 3 | `03_embeddings.py` | 共现矩阵 + PPMI + SVD | 词向量从哪来、dense 检索的本质 |
| 4 | `04_vector_index.py` | 暴力检索 → IVF 索引 | 为什么需要 ANN、速度/精度权衡 |
| 5 | `05_hybrid_search.py` | BM25 + RRF 混合 | 什么时候向量检索反而输给关键词 |
| 6 | `06_answer.py` | 带引用的生成 | prompt 组装、防引用幻觉 |
| 7 | `07_evaluation.py` | recall@k / MRR | 怎么科学地证明改进有效 |

<!-- TODO: 每完成一个模块，把该行状态改为 ✅ 并附上讲解链接 -->

## 快速开始

```bash
git clone https://github.com/<你的用户名>/mini-rag.git
cd mini-rag

# 逐模块学习
python 01_chunking.py

# 或者直接用：对你自己的文档提问（离线模式不需要任何 API）
python app.py index ./你的资料目录
python app.py ask "你的问题" --offline
```

## 实现进度（Roadmap）

- [ ] 01_chunking
- [ ] 02_sparse_search
- [ ] 03_embeddings
- [ ] 04_vector_index
- [ ] 05_hybrid_search
- [ ] 06_answer
- [ ] 07_evaluation
- [ ] app.py 完整闭环

## FAQ

**Q：需要 OpenAI API 吗？**
不需要。默认离线可跑；`06_answer.py` 接入 LLM 后才需要 API（OpenAI 兼容接口，可配 base_url）。

**Q：为什么不直接用 LangChain？**
见「为什么从零写」。这不是反框架，而是先把原理焊死再上车。

## License

MIT
