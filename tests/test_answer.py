import importlib

import pytest

answering = importlib.import_module("06_answer")

HITS = [
    {"text": "导数的几何意义是切线的斜率。", "source": "微积分.md#3"},
    {"text": "可导必连续，连续不一定可导。", "source": "微积分.md#4"},
]


def test_build_context_numbers_sources():
    ctx = answering.build_context(HITS)
    assert ctx.startswith("[1]")
    assert "导数的几何意义" in ctx and "微积分.md#3" in ctx
    assert "<doc" in ctx and "</doc>" in ctx  # 分隔符把结构信息与语料正文隔开
    assert "[2]" in ctx and "微积分.md#4" in ctx


def test_prompt_contains_question_and_honesty_rule():
    prompt = answering.build_prompt("导数的几何意义？", HITS)
    assert "导数的几何意义？" in prompt
    assert "资料不足" in prompt  # 诚实条款必须写进 prompt
    assert "[1]" in prompt


def test_messages_split_system_and_user():
    """角色隔离：指令走 system，语料与问题走 user。"""
    messages = answering.build_messages("导数的几何意义？", HITS)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "资料不足" in messages[0]["content"]
    assert "导数的几何意义" in messages[1]["content"]


def test_offline_answer_lists_hits():
    out = answering.answer("任意问题", HITS, offline=True)
    assert "离线" in out
    assert "[1]" in out and "[2]" in out


def test_empty_hits_short_circuits():
    """0 命中必须拒答：离线不装样子，在线也不该发空资料请求。"""
    out = answering.answer("怎么给猫剪指甲", [], offline=True)
    assert "资料不足" in out
    assert "最相关的原文片段" not in out


def test_out_of_range_citations_detected():
    assert answering.out_of_range_citations("深冷区 [2]，消毒 [7]，总述 [3][9]", 1) == [2, 3, 7, 9]
    assert answering.out_of_range_citations("依据 [1] 与 [2]", 2) == []
    assert answering.out_of_range_citations("无引用", 2) == []


def test_answer_appends_warning_for_bad_citations(monkeypatch):
    """代码级护栏：越界引用穿透不了 answer()。"""
    fake = "答案 [1] 与 [7]。"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(answering, "call_llm", lambda *a, **kw: fake)
    out = answering.answer("问题", HITS, offline=False)
    assert "[系统提示]" in out and "[7]" in out


def test_call_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MINIRAG_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        answering.call_llm("hi")


def test_call_llm_messages_and_temperature(monkeypatch):
    """请求体：system/user 分离，temperature=0。"""
    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices": [{"message": {"content": "ok"}}]}'

    def fake_urlopen(req, timeout=None):
        import json

        captured["body"] = json.loads(req.data.decode())
        return FakeResp()

    import urllib.request

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = answering.call_llm("问题", system="系统指令")
    assert out == "ok"
    body = captured["body"]
    assert body["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
