"""06 · 带引用的回答：把检索结果交给 LLM

检索到位还不够，最后一步是把相关块组织进 prompt，让 LLM 生成答案。
两个工程要点：

1. 引用编号：给每个块编号 [1][2]…，要求模型回答时标注引用——
   用户能溯源，幻觉也更容易被发现；
2. 诚实降级：没配 API Key 时进入"离线模式"，直接返回检索到的原文，
   而不是假装回答。工具应该诚实，而不是编造。

调用 LLM 用标准库 urllib 直连 OpenAI 兼容接口（不引入 openai SDK）。
接入时配置环境变量：

  OPENAI_API_KEY    必填
  MINIRAG_BASE_URL  选填，默认 https://api.openai.com/v1（兼容中转/本地服务）
  MINIRAG_MODEL     选填，默认 gpt-4o-mini

运行演示：python 06_answer.py（离线模式，无需 API）
"""

import json
import os
import urllib.request

PROMPT_TEMPLATE = """你是一个严谨的资料问答助手。请仅根据下面的资料回答问题，\
引用资料时在句末标注编号（如 [1]）。如果资料不足以回答，请直接回答"资料不足"。

资料：
{context}

问题：{question}

回答："""


def build_context(hits: list[dict]) -> str:
    """把检索结果（dict：text/source）编号成 prompt 里的资料区。"""
    return "\n".join(
        f"[{i}] {h['text']}（来源：{h['source']}）" for i, h in enumerate(hits, 1)
    )


def build_prompt(question: str, hits: list[dict]) -> str:
    return PROMPT_TEMPLATE.format(context=build_context(hits), question=question)


def answer(question: str, hits: list[dict], offline: bool = True) -> str:
    if offline:
        lines = [
            f"[{i}] {h['text']}（来源：{h['source']}）" for i, h in enumerate(hits, 1)
        ]
        return "【离线模式】未接入 LLM，以下是与问题最相关的原文片段：\n" + "\n".join(lines)
    return call_llm(build_prompt(question, hits))


def call_llm(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY；或改用 answer(..., offline=True)")
    base = os.environ.get("MINIRAG_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("MINIRAG_MODEL", "gpt-4o-mini")
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


if __name__ == "__main__":
    hits = [
        {"text": "导数的几何意义是切线的斜率。", "source": "微积分.md#3"},
        {"text": "可导必连续，连续不一定可导。", "source": "微积分.md#4"},
    ]
    print(answer("导数的几何意义是什么？", hits, offline=True))
