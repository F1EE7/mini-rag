import importlib

hybrid = importlib.import_module("05_hybrid_search")


def test_rrf_known_values():
    # list1: 0→1/61, 1→1/62, 2→1/63；list2: 1→1/61, 3→1/62
    # 融合分：1 > 0 > 3 > 2
    assert hybrid.rrf_fuse([[0, 1, 2], [1, 3]]) == [1, 0, 3, 2]


def test_rrf_tie_keeps_insertion_order():
    assert hybrid.rrf_fuse([[0], [1]]) == [0, 1]


def test_bm25_prefers_exact_term():
    docs = ["泰勒公式用多项式逼近函数。", "足球比赛每队十一人。"]
    bm25 = hybrid.BM25Index().fit(docs)
    assert bm25.search("泰勒公式", 1)[0][0] == 0
    assert bm25.search("足球", 1)[0][0] == 1


def test_hybrid_search_returns_k_results():
    docs = ["泰勒公式逼近函数。", "不定积分是求导的逆运算。", "足球比赛十一个人。"]
    system = hybrid.HybridSearch().fit(docs)
    assert len(system.search("求原函数", 2)) == 2
