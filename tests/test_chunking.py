import importlib

chunking = importlib.import_module("01_chunking")


def test_fixed_chunks_exact_offsets():
    text = "".join(str(i % 10) for i in range(100))
    chunks = chunking.fixed_chunks(text, size=30, overlap=10)
    assert len(chunks) == 5  # 步长 20，起点 0/20/40/60/80
    for i, c in enumerate(chunks):
        assert c == text[i * 20:i * 20 + 30]


def test_fixed_chunks_overlap_shares_content():
    text = "0123456789" * 4
    chunks = chunking.fixed_chunks(text, size=10, overlap=4)
    assert chunks[1][:4] == chunks[0][-4:] == text[6:10]


def test_fixed_chunks_rejects_bad_params():
    import pytest

    with pytest.raises(ValueError):
        chunking.fixed_chunks("abc", size=5, overlap=5)
    with pytest.raises(ValueError):
        chunking.fixed_chunks("abc", size=0)


def test_paragraph_merging_short_paras():
    """任一侧过短就合并：短段落之间合并，也并入后面的长段落。"""
    text = "短\n\n更短\n\n这是一个足够长的段落，超过最小长度限制所以独立成块。"
    chunks = chunking.paragraph_chunks(text, min_size=10)
    assert len(chunks) == 1  # "短\n更短"（4 字 < 10）也并入后面的长段落
    assert "短\n更短\n这是一个足够长的段落" in chunks[0]


def test_paragraph_no_merge_when_both_long():
    text = "这是一个足够长的段落，超过最小长度限制。\n\n这是另一个足够长的段落，也超过限制。"
    chunks = chunking.paragraph_chunks(text, min_size=10)
    assert len(chunks) == 2


def test_paragraph_heading_attaches_to_following_body():
    """回归：标题不再成为碎片块，且应并入它下面的正文（而不是上一节）。"""
    body = "正文内容" * 40  # 160 字
    text = body + "\n\n## 监测设备与数据记录\n\n" + body
    chunks = chunking.paragraph_chunks(text, min_size=100)
    assert len(chunks) == 2
    assert chunks[0] == body  # 上一节不受影响
    assert chunks[1].startswith("## 监测设备与数据记录")  # 标题领着它自己的正文


def test_chunk_text_handles_empty_and_long():
    assert chunking.chunk_text("") == []
    assert chunking.chunk_text("   ") == []
    pieces = chunking.chunk_text("长" * 500, size=200, overlap=50)
    assert all(len(p) <= 200 for p in pieces)
    assert sum(len(p) for p in pieces) >= 500  # overlap 会重复计入


def test_chunk_text_min_para_controls_merging():
    """min_para 是段落合并阈值：调大更粗，调小保留段落粒度。"""
    text = "第一段大约二十个字的内容。\n\n第二段也是差不多的长度。"
    assert len(chunking.chunk_text(text, size=300, min_para=5)) == 2
    assert len(chunking.chunk_text(text, size=300, min_para=50)) == 1
