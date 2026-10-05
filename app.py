"""mini-rag 命令行入口

  python app.py index ./资料目录          # 建索引（结果存 ./.minirag/）
  python app.py ask "你的问题" --offline  # 离线问答（不需要任何 API）

支持的文件：.txt / .md；.pdf 需要可选依赖 pypdf（pip install pypdf）。
检索：BM25 + TF-IDF 两路 RRF 融合；如果安装了 sentence-transformers，
会自动加入语义向量这一路（可选增强，不装不影响使用）。
"""

import argparse
import importlib
import json
import pathlib
import sys

import numpy as np

chunking = importlib.import_module("01_chunking")
sparse = importlib.import_module("02_sparse_search")
hybrid = importlib.import_module("05_hybrid_search")
answering = importlib.import_module("06_answer")

INDEX_DIR = pathlib.Path(".minirag")
INDEX_FILE = INDEX_DIR / "index.json"


def read_document(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"[跳过] {path.name}: 读取 PDF 需要 pypdf（pip install pypdf）")
            return ""
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return ""


def build_index(src_dir: str) -> list[dict]:
    """读取目录下所有支持的文章件，切块并落盘。"""
    src = pathlib.Path(src_dir)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    chunks: list[dict] = []
    for f in files:
        text = read_document(f)
        for j, piece in enumerate(chunking.chunk_text(text, size=300, overlap=50)):
            chunks.append({"text": piece, "source": f"{f.name}#{j}"})
    INDEX_DIR.mkdir(exist_ok=True)
    INDEX_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已索引 {len(files)} 个文件，切分出 {len(chunks)} 个块 -> {INDEX_FILE}")
    return chunks


def load_index() -> list[dict]:
    if not INDEX_FILE.exists():
        sys.exit("找不到索引，请先运行: python app.py index <资料目录>")
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def _dense_rank(query: str, docs: list[str], top: int = 20) -> list[int]:
    """语义向量这一路是可选增强：装了 sentence-transformers 才启用。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return []
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    vectors = model.encode(docs, normalize_embeddings=True)
    qv = model.encode([query], normalize_embeddings=True)[0]
    return [int(i) for i in np.argsort(-(vectors @ qv))[:top]]


def retrieve(query: str, chunks: list[dict], k: int = 5) -> list[dict]:
    """BM25 + TF-IDF（+ 可选语义向量）多路召回，RRF 融合取前 k。"""
    docs = [c["text"] for c in chunks]
    sparse_rank = [i for i, _ in sparse.TfidfIndex().fit(docs).search(query, 20)]
    bm25_rank = [i for i, _ in hybrid.BM25Index().fit(docs).search(query, 20)]
    ranked = hybrid.rrf_fuse([sparse_rank, bm25_rank, _dense_rank(query, docs)])[:k]
    return [chunks[i] for i in ranked]


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
        print(ask(args.question, load_index(), args.k, offline=args.offline))


if __name__ == "__main__":
    main()
