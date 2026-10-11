import importlib
import sys

import pytest

app = importlib.import_module("app")


def _make_docs(tmp_path):
    docs = tmp_path / "资料"
    docs.mkdir()
    (docs / "math.md").write_text(
        "泰勒公式用多项式逼近复杂函数，是近似计算的重要工具。" * 3, encoding="utf-8"
    )
    (docs / "ball.md").write_text(
        "足球比赛每队十一人，在操场上进行对抗。住宿安排另行通知。" * 3, encoding="utf-8"
    )
    return docs


def test_end_to_end_offline(tmp_path, monkeypatch):
    docs = _make_docs(tmp_path)
    monkeypatch.chdir(tmp_path)

    chunks = app.build_index(str(docs))
    assert len(chunks) == 2
    assert {c["source"].split("#")[0] for c in chunks} == {"math.md", "ball.md"}

    hits = app.retrieve("足球 一共多少人", chunks, k=1)
    assert hits[0]["source"].startswith("ball.md")

    out = app.ask("泰勒公式有什么用？", chunks, k=2, offline=True)
    assert "离线" in out and "泰勒" in out

    # 0 命中：诚实拒答
    assert "资料不足" in app.ask("怎么给猫剪指甲", chunks, k=2, offline=True)


def test_index_persists_fitted_retriever(tmp_path, monkeypatch):
    """索引阶段 fit 一次落盘；ask 阶段加载 pickle，不重建。"""
    docs = _make_docs(tmp_path)
    monkeypatch.chdir(tmp_path)
    app.build_index(str(docs))
    assert app.PKL_FILE.exists() and app.INDEX_FILE.exists()

    chunks, retriever = app.load_index()
    assert len(chunks) == 2
    assert retriever.tfidf is not None and retriever.bm25 is not None
    hits = retriever.search("泰勒公式", k=1)
    assert hits[0]["source"].startswith("math.md")


def test_index_rejects_missing_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        app.build_index(str(tmp_path / "goodd"))  # 目录名打错一个字母
    assert "不存在" in str(exc.value.code)


def test_empty_index_never_clobbers_existing(tmp_path, monkeypatch):
    """空目录/无可读文件时绝不覆盖已有索引（防路径手误静默销毁）。"""
    docs = _make_docs(tmp_path)
    monkeypatch.chdir(tmp_path)
    app.build_index(str(docs))
    before = app.PKL_FILE.read_bytes()

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        app.build_index(str(empty))
    assert app.PKL_FILE.read_bytes() == before


def test_single_char_query_fallback(tmp_path, monkeypatch):
    """单字查询 bigram 必然 0 命中，应有子串兜底。"""
    docs = _make_docs(tmp_path)
    monkeypatch.chdir(tmp_path)
    chunks = app.build_index(str(docs))
    hits = app.retrieve("住", chunks, k=1)
    assert hits and "住宿" in hits[0]["text"]


def test_index_filters_fragment_chunks(tmp_path, monkeypatch):
    docs = tmp_path / "资料2"
    docs.mkdir()
    (docs / "a.md").write_text(
        "足够长的正文段落，用来承载主要内容。" * 5
        + "\n\n## 孤立短标题\n\n"
        + "另一段足够长的正文内容，同样超过阈值。" * 5,
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    chunks = app.build_index(str(docs))
    assert all(len(c["text"]) >= app.MIN_CHUNK_CHARS for c in chunks)


def test_read_document_strips_bom(tmp_path):
    f = tmp_path / "bom.md"
    f.write_bytes("\ufeff泰勒公式".encode("utf-8"))
    assert app.read_document(f).startswith("泰勒公式")


def test_cli_rejects_nonpositive_k(tmp_path, monkeypatch, capsys):
    docs = _make_docs(tmp_path)
    monkeypatch.chdir(tmp_path)
    app.build_index(str(docs))
    monkeypatch.setattr(sys, "argv", ["app.py", "ask", "泰勒公式", "--k", "0"])
    with pytest.raises(SystemExit):
        app.main()
    assert "--k" in capsys.readouterr().err
