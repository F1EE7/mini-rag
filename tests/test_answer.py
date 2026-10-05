import importlib

import pytest

answering = importlib.import_module("06_answer")

HITS = [
    {"text": "导数的几何意义是切线的斜率。", "source": "微积分.md#3"},
    {"text": "可导必连续，连续不一定可导。", "source": "微积分.md#4"},
]


def test_build_context_numbers_sources():
    ctx = answering.build_context(HITS)
    assert ctx.startswith("[1] 导数的几何意义")
    assert "[2]" in ctx and "微积分.md#4" in ctx


def test_prompt_contains_question_and_honesty_rule():
    prompt = answering.build_prompt("导数的几何意义？", HITS)
    assert "导数的几何意义？" in prompt
    assert "资料不足" in prompt  # 诚实条款必须写进 prompt
    assert "[1]" in prompt


def test_offline_answer_lists_hits():
    out = answering.answer("任意问题", HITS, offline=True)
    assert "离线" in out
    assert "[1]" in out and "[2]" in out


def test_call_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        answering.call_llm("hi")
