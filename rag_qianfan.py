import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from rag_retrieval import search

load_dotenv()

API_KEY = os.getenv("QIANFAN_API_KEY")
if not API_KEY:
    raise RuntimeError("QIANFAN_API_KEY 未设置，请在 .env 中配置。")

# 千帆 OpenAI 协议 v2 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url="https://qianfan.baidubce.com/v2",
)

# 按你账号可用模型替换
DEFAULT_MODEL = "ernie-speed-8k"


def build_context(retrieved: List[Dict], max_chars_each: int = 600) -> str:
    parts = []
    for i, r in enumerate(retrieved):
        md = r.get("metadata", {}) or {}
        title = md.get("title") or md.get("file") or md.get("source") or f"片段{i+1}"
        url = md.get("url", "")
        chunk = r.get("chunk", "")
        snippet = chunk[:max_chars_each]
        parts.append(
            f"[{i+1}] 标题：{title}\n链接：{url}\n内容片段：{snippet}"
        )
    return "\n\n".join(parts)


def generate_answer(
    user_question: str,
    days: int,
    top_k: int = 5,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    city: str | None = None,
):
    """根据用户问题 + 天数 + 检索结果，生成结构化行程"""
    days = max(1, min(days, 7))  # 安全限制：1~7 天

    retrieved = search(user_question, top_k=top_k)
    # 根据城市名过滤（文件路径中包含城市名，比如 paris, budapest）
    if city:
        city_l = city.strip().lower()
        filtered = []
        for r in retrieved:
            md = r.get("metadata", {}) or {}
            md_city = str(md.get("city", "")).lower()
            if md_city and md_city == city_l:
                filtered.append(r)

        if filtered:
            retrieved = filtered
    context = build_context(retrieved)

    prompt = f"""
你是一名中文旅行规划顾问。现在你要根据用户的需求和真实游记片段，设计一份结构清晰、易执行的旅行计划。

【用户需求】
{user_question}

【检索到的游记片段】
{context}

【输出格式要求 —— 请严格遵守】

1. 开场总结（1–2 句）
   - 概括这趟旅行的整体风格、适合人群和大致节奏。

2. 行程安排（共 {days} 天）：
   - 每一天必须用英文 Day 开头，并严格按照下面的模板输出：

   Day 1 ｜ 一句话概括当天主题
     - 上午：...
     - 下午：...
     - 晚上：...

   Day 2 ｜ 一句话概括当天主题
     - 上午：...
     - 下午：...
     - 晚上：...

   ...
   Day {days} ｜ 一句话概括当天主题
     - 上午：...
     - 下午：...
     - 晚上：...

   要求：
   - 只生成 Day 1 ~ Day {days}，不要多也不要少；
   - 尽量优先使用上面游记片段中提到的地点、体验和路线；
   - 如果材料不足，可以补充一些常见景点和活动，但要合理、不要胡编乱造；
   - 景点名称尽量使用英文名称（例如：Louvre Museum, Eiffel Tower, Montmartre）。

3. 注意事项：
   - 至少 3 条，和天气、节奏、预算、安全、交通等相关；
   - 可以结合游记片段中的提醒。

4. 参考来源：
   - 用列表形式列出参考到的游记片段，格式如下：
     [1] 标题：xxx（如果有） 链接：xxx（如果有）
     [2] 标题：...

只按照上述结构输出，不要加入多余的解释性文字。
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一名严谨、专业的中文旅行规划顾问，擅长根据真实游记和用户偏好制定行程。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=1200,
    )

    # 兼容不同字段形式
    content = ""
    try:
        content = resp.choices[0].message["content"]
    except Exception:
        try:
            content = resp.choices[0].message.content
        except Exception:
            try:
                content = resp.choices[0].text
            except Exception:
                content = str(resp)

    return content, retrieved