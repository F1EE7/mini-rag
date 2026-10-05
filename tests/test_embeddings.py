import importlib

import numpy as np

embeddings = importlib.import_module("03_embeddings")


def build_corpus():
    study = [
        ["我", "复习", "数学", "知识"],
        ["他", "温习", "数学", "知识"],
        ["我", "复习", "笔记", "准备", "考试"],
        ["他", "温习", "笔记", "准备", "考试"],
        ["复习", "和", "温习", "帮助", "记忆"],
        ["老师", "要求", "复习", "重点"],
        ["老师", "要求", "温习", "重点"],
        ["温习", "旧", "知识", "非常", "重要"],
        ["她", "复习", "课文", "然后", "做", "练习"],
    ]
    sport = [
        ["我们", "踢", "足球", "在", "操场"],
        ["足球", "是", "流行", "的", "运动"],
        ["他", "喜欢", "篮球", "和", "足球"],
        ["运动", "以后", "我", "非常", "开心"],
        ["操场", "上", "有", "人", "在", "踢", "足球"],
        ["篮球", "比赛", "非常", "精彩"],
        ["周末", "我们", "打", "篮球"],
    ]
    return (study + sport) * 6


def test_vector_shapes_and_normalization():
    emb = embeddings.PPMIEmbedder(dim=8).fit(build_corpus())
    assert emb.vectors.shape == (len(emb.word2id), 8)
    assert np.allclose(np.linalg.norm(emb.vectors, axis=1), 1.0)


def test_semantic_similarity():
    emb = embeddings.PPMIEmbedder(dim=16).fit(build_corpus())
    close = embeddings.cosine(emb.transform("复习"), emb.transform("温习"))
    far = embeddings.cosine(emb.transform("复习"), emb.transform("足球"))
    assert close > far
    neighbors = [w for w, _ in emb.nearest("复习", 5)]
    # 近邻应全部来自"学习"主题：SVD 空间里同主题词聚在一起
    # （近邻可能包含同义词"温习"，也可能包含直接共现的词——一阶 vs 二阶相似）
    assert not {"足球", "篮球", "操场", "运动"} & set(neighbors)
    assert "温习" in neighbors


def test_oov_word_is_zero_vector():
    emb = embeddings.PPMIEmbedder(dim=8).fit(build_corpus())
    assert np.all(emb.transform("查无此词") == 0)
    assert np.all(emb.doc_vector(["查无此词"]) == 0)
