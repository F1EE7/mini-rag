"""mini-rag 命令行入口

  python app.py index ./资料目录          # 建索引（结果存 ./.minirag/）
  python app.py ask "你的问题" --offline  # 离线问答（不需要任何 API）

支持的文件：.txt / .md；.pdf 需要可选依赖 pypdf（pip install pypdf）。
检索：TF-IDF + BM25 两路 RRF 融合；如果安装了 sentence-transformers，
会自动加入语义向量这一路（可选增强，不装不影响使用）。

工程要点：索引在 `index` 阶段 fit 一次并持久化（pickle），`ask` 只打分——
不要每次提问都全量重建索引、重载 embedding 模型。
"""

import argparse
import importlib
import json
import pathlib
import pickle
import sys

import numpy as np

chunking = importlib.import_module("01_chunking")
sparse = importlib.import_module("02_sparse_search")
hybrid = importlib.import_module("05_hybrid_search")
answering = importlib.import_module("06_answer")

INDEX_DIR = pathlib.Path(".minirag")
INDEX_FILE = INDEX_DIR / "index.json"   # 人类可读的块文本（调试用）
PKL_FILE = INDEX_DIR / "index.pkl"      # 已 fit 的检索器（查询用）
MIN_CHUNK_CHARS = 50                    # 丢弃孤立标题等碎片块（这类块检索时白占坑）

_ST_MODEL = None  # sentence-transformers 模型：进程内只加载一次


def _get_st_model():
    """惰性加载 sentence-transformers 模型（装了才调用）。"""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _ST_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _ST_MODEL


def read_document(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        # utf-8-sig：顺手吃掉 UTF-8 BOM，避免 BOM 混进第一个 token
        return path.read_text(encoding="utf-8-sig", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"[跳过] {path.name}: 读取 PDF 需要 pypdf（pip install pypdf）")
            return ""
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return ""


class Retriever:
    """一次 fit、多次 search 的多路检索器（TF-IDF + BM25 [+ 语义向量]）。

    语义向量一路是可选增强：fit 时若装了 sentence-transformers 就把全库
    chunk 编码存下来（模型本身不落盘，查询时惰性加载一次）。
    """

    def __init__(self, dense: bool = True):
        self.dense = dense
        self.dense_vecs: np.ndarray | None = None

    def fit(self, chunks: list[dict]) -> "Retriever":
        self.chunks = chunks
        docs = [c["text"] for c in chunks]
        self.tfidf = sparse.TfidfIndex().fit(docs)
        self.bm25 = hybrid.BM25Index().fit(docs)
        if self.dense:
            try:
                model = _get_st_model()
                self.dense_vecs = model.encode(docs, normalize_embeddings=True)
            except ImportError:
                self.dense_vecs = None
        return self

    def search(self, query: str, k: int = 5) -> list[dict]:
        ranks = [
            [i for i, _ in self.tfidf.search(query, 20)],
            [i for i, _ in self.bm25.search(query, 20)],
        ]
        if self.dense_vecs is not None:
            try:
                model = _get_st_model()
                qv = model.encode([query], normalize_embeddings=True)[0]
                ranks.append([int(i) for i in np.argsort(-(self.dense_vecs @ qv))[:20]])
            except ImportError:
                pass  # 建索引时装了、查询时没装：降级为两路，不崩
        ranked = hybrid.rrf_fuse(ranks)[:k] if any(ranks) else []
        if not ranked:
            ranked = self._substring_fallback(query, k)
        return [self.chunks[i] for i in ranked]

    def _substring_fallback(self, query: str, k: int) -> list[int]:
        """兜底：bigram 分词对单字查询（如"住"）必然 0 命中，退化为子串匹配。"""
        return [i for i, c in enumerate(self.chunks) if query in c["text"]][:k]

    def save(self, path: pathlib.Path) -> None:
        pickle.dump(
            {
                "chunks": self.chunks,
                "tfidf": self.tfidf,
                "bm25": self.bm25,
                "dense_vecs": self.dense_vecs,
            },
            path.open("wb"),
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "Retriever":
        data = pickle.load(path.open("rb"))
        r = cls(dense=False)
        r.chunks = data["chunks"]
        r.tfidf = data["tfidf"]
        r.bm25 = data["bm25"]
        r.dense_vecs = data.get("dense_vecs")
        return r


def build_index(src_dir: str) -> list[dict]:
    """读取目录下所有支持的文章件，切块并持久化（文本 + 已 fit 的检索器）。"""
    src = pathlib.Path(src_dir)
    if not src.is_dir():
        sys.exit(f"[错误] 目录不存在：{src_dir}")
    files = sorted(p for p in src.rglob("*") if p.is_file())
    chunks: list[dict] = []
    indexed_files = 0
    dropped = 0
    for f in files:
        text = read_document(f)
        pieces = chunking.chunk_text(text, size=300, overlap=50) if text.strip() else []
        if pieces:
            indexed_files += 1
        for j, piece in enumerate(pieces):
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append({"text": piece, "source": f"{f.name}#{j}"})
            else:
                dropped += 1
    if not chunks:
        # 空索引绝不落盘——路径打错一个字母也不该静默销毁已有索引
        sys.exit(
            "[错误] 没有切出任何块：目录里没有可读的 .txt/.md（.pdf 需要 pypdf），"
            "或内容都低于最小块长。已保留原索引。"
        )
    INDEX_DIR.mkdir(exist_ok=True)
    retriever = Retriever().fit(chunks)
    retriever.save(PKL_FILE)
    INDEX_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    note = f"，过滤碎片块 {dropped} 个" if dropped else ""
    print(f"已索引 {indexed_files} 个文件，切分出 {len(chunks)} 个块{note} -> {INDEX_DIR}/")
    return chunks


def load_index() -> tuple[list[dict], Retriever]:
    if PKL_FILE.exists():
        retriever = Retriever.load(PKL_FILE)
        return retriever.chunks, retriever
    if INDEX_FILE.exists():  # 兼容只有 json 的旧索引：现场重建
        chunks = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        return chunks, Retriever().fit(chunks)
    sys.exit("找不到索引，请先运行: python app.py index <资料目录>")


def retrieve(query: str, chunks: list[dict], k: int = 5) -> list[dict]:
    """无持久化索引时的便捷路径：现场 fit（慢，仅供编程调用/测试）。"""
    return Retriever().fit(chunks).search(query, k)


def ask(query: str, chunks: list[dict], k: int = 5, offline: bool = True) -> str:
    hits = retrieve(query, chunks, k)
    return answering.answer(query, hits, offline=offline)


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-rag：对自己的文档提问")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="对目录建索引")
    p_index.add_argument("directory")

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question")
    p_ask.add_argument("--k", type=int, default=5)
    p_ask.add_argument("--offline", action="store_true", help="不调用 LLM，直接返回相关原文")

    args = parser.parse_args()
    if args.command == "index":
        build_index(args.directory)
    else:
        if args.k < 1:
            parser.error("--k 必须是正整数")
        chunks, retriever = load_index()
        hits = retriever.search(args.question, args.k)
        print(answering.answer(args.question, hits, offline=args.offline))


if __name__ == "__main__":
    main()
