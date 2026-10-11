import importlib

sparse = importlib.import_module("02_sparse_search")


def test_bigram_tokenize_chinese():
    assert sparse.bigram_tokenize("机器学习") == ["机器", "器学", "学习"]


def test_bigram_tokenize_keeps_ascii_words():
    assert "rag" in sparse.bigram_tokenize("用 RAG 检索")


def test_bigram_tokenize_no_cross_boundary_noise():
    """bigram 不跨标点、不跨中英文/单词边界；英文整词在原文上提取。"""
    tokens = sparse.bigram_tokenize("hello world")
    assert "hello" in tokens and "world" in tokens
    assert "ow" not in tokens  # 跨单词边界的伪 bigram
    # 回归用例：旧实现先删空白再取整词，"hello world foo" 只会得到粘团伪词
    assert {"hello", "world", "foo"} <= set(sparse.bigram_tokenize("hello world foo"))
    assert "signature" in sparse.bigram_tokenize("A signature stays valid for 300 seconds.")


def test_bigram_tokenize_no_bigram_across_punctuation():
    tokens = sparse.bigram_tokenize("你好。世界")
    assert tokens == ["你好", "世界"]  # 不产生 "好。" / "。世"


def test_bigram_tokenize_single_char_and_empty():
    assert sparse.bigram_tokenize("住") == ["住"]
    assert sparse.bigram_tokenize("") == []
    assert sparse.bigram_tokenize("  ") == []


def test_tfidf_ranks_relevant_first():
    docs = [
        "泰勒公式用多项式逼近复杂函数，是近似计算的重要工具。",
        "定积分与牛顿-莱布尼茨公式联系积分与导数。",
        "足球比赛每队十一人，在球场上对抗。",
    ]
    index = sparse.TfidfIndex().fit(docs)
    hits = index.search("泰勒公式 逼近", k=3)
    assert hits[0][0] == 0
    assert hits[0][1] > 0


def test_tfidf_finds_english_word():
    """回归：英文整词分词修复后，纯英文查询能命中。"""
    docs = ["The signature validity period is 300 seconds.", "签名与有效期无关的中文段落。"]
    index = sparse.TfidfIndex().fit(docs)
    hits = index.search("signature validity", k=2)
    assert hits and hits[0][0] == 0


def test_oov_query_returns_empty():
    index = sparse.TfidfIndex().fit(["数学分析是数学的核心课程"])
    assert index.search("足球") == []
