"""06 · 带引用的回答：把检索结果交给 LLM

检索到位还不够，最后一步是把相关块组织进 prompt，让 LLM 生成答案。
工程要点（按防线从便宜到贵排序）：

1. 空结果短路：检索为空就直接回答"资料不足"，**不调用 API**——
   既省钱，也不把拒答押在模型自觉上；
2. 角色隔离与分隔符：系统指令放 system 消息，语料每块用 <doc> 标签
   包起来——语料里混进的"[99] 来源：xxx"式伪造标记不再与真实格式同形
   （注意：这只是提高注入门槛，不能根除提示注入）；
3. 引用校验：模型输出后检查 [n] 是否越界（超出提供的资料条数），
   越界即追加警告——代码级护栏，不依赖模型配合；
4. 诚实降级：没配 API Key 时进入"离线模式"，直接返回检索到的原文。

注意：代码能拦的是**编号越界**；"编号合法但段落不支撑该句"的无依据引用
无法在本地零成本检出，评测时需要逐引用审计（见 07 的思路）。

调用 LLM 用标准库 urllib 直连 OpenAI 兼容接口（不引入 openai SDK）。
接入时配置环境变量：

  OPENAI_API_KEY / MINIRAG_API_KEY  必填（后者优先级更高）
  MINIRAG_BASE_URL                  选填，默认 https://api.openai.com/v1
  MINIRAG_MODEL                     选填，默认 gpt-4o-mini

运行演示：python 06_answer.py（离线模式，无需 API）
"""

import json
import os
import re
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "你是一个严谨的资料问答助手。请仅根据用户消息中 <doc> 标签提供的资料回答问题，"
    '引用资料时在句末标注资料编号（如 [1]）。如果资料不足以回答，'
    '请直接回答"资料不足"，不要用你自己的知识补答。'
)

USER_TEMPLATE = """资料：
{context}

问题：{question}

回答："""

_CITATION = re.compile(r"\[(\d+)\]")


def build_context(hits: list[dict]) -> str:
    """把检索结果编号成 prompt 里的资料区，每块用 <doc> 标签包裹。

    标签把"资料正文"与"结构信息"隔开：语料内部再出现"[99]（来源：…）"
    这样伪造的来源标记，也只是普通文本，不再与真实结构同形。
    """
    return "\n".join(
        f'[{i}] <doc source="{h["source"]}">{h["text"]}</doc>'
        for i, h in enumerate(hits, 1)
    )


def build_prompt(question: str, hits: list[dict]) -> str:
    """拼接完整 prompt 文本（演示/测试用；实际请求走 build_messages）。"""
    return SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
        context=build_context(hits), question=question
    )


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    """系统指令与语料分属 system / user 两条消息（角色隔离）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=build_context(hits), question=question),
        },
    ]


def out_of_range_citations(text: str, n_docs: int) -> list[int]:
    """找出回答中超出资料条数的引用编号（代码级反幻觉护栏）。"""
    return sorted(
        {int(n) for n in _CITATION.findall(text) if not 1 <= int(n) <= n_docs}
    )


def answer(question: str, hits: list[dict], offline: bool = True) -> str:
    if not hits:
        # 0 命中必须短路：离线时不假装"以下是相关片段"，在线时不发空资料请求
        return "资料不足：没有检索到与问题相关的内容。"
    if offline:
        lines = [
            f"[{i}] {h['text']}（来源：{h['source']}）" for i, h in enumerate(hits, 1)
        ]
        return "【离线模式】未接入 LLM，以下是与问题最相关的原文片段：\n" + "\n".join(lines)
    text = call_llm(
        USER_TEMPLATE.format(context=build_context(hits), question=question),
        system=SYSTEM_PROMPT,
    )
    bad = out_of_range_citations(text, len(hits))
    if bad:
        refs = ", ".join(f"[{b}]" for b in bad)
        text += f"\n\n[系统提示] 检测到越界引用 {refs}（资料只有 {len(hits)} 条），请谨慎采信。"
    return text


def call_llm(prompt: str, system: str | None = None) -> str:
    api_key = os.environ.get("MINIRAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY（或 MINIRAG_API_KEY）；或改用 answer(..., offline=True)")
    base = os.environ.get("MINIRAG_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("MINIRAG_MODEL", "gpt-4o-mini")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": 0}  # 问答要确定性
    ).encode()
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode("utf-8", "ignore")
        raise RuntimeError(f"LLM 接口返回 HTTP {e.code}：{detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接 LLM 接口 {base}：{e.reason}") from e


if __name__ == "__main__":
    hits = [
        {"text": "导数的几何意义是切线的斜率。", "source": "微积分.md#3"},
        {"text": "可导必连续，连续不一定可导。", "source": "微积分.md#4"},
    ]
    print(answer("导数的几何意义是什么？", hits, offline=True))
    print()
    print(answer("月球背面的温度是多少？", [], offline=True))  # 0 命中：诚实拒答
