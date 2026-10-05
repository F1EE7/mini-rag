import importlib

sparse = importlib.import_module("02_sparse_search")


def test_bigram_tokenize_chinese():
    assert sparse.bigram_tokenize("机器学习") == ["机器", "器学", "学习"]


def test_bigram_tokenize_keeps_ascii_words():
    assert "rag" in sparse.bigram_tokenize("用 RAG 检索")


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


def test_oov_query_returns_empty():
    index = sparse.TfidfIndex().fit(["数学分析是数学的核心课程"])
    assert index.search("足球") == []
