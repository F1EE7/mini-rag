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
    text = "短\n\n更短\n\n这是一个足够长的段落，超过最小长度限制所以独立成块。"
    chunks = chunking.paragraph_chunks(text, min_size=10)
    assert chunks[0] == "短\n更短"
    assert len(chunks) == 2


def test_chunk_text_handles_empty_and_long():
    assert chunking.chunk_text("") == []
    assert chunking.chunk_text("   ") == []
    pieces = chunking.chunk_text("长" * 500, size=200, overlap=50)
    assert all(len(p) <= 200 for p in pieces)
    assert sum(len(p) for p in pieces) >= 500  # overlap 会重复计入
