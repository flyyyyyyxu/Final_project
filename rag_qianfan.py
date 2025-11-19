# rag_qianfan.py
import os
from openai import OpenAI
from typing import List
from rag_retrieval import search
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 从 .env 读取 QIANFAN_API_KEY（或设置环境变量）
API_KEY = os.getenv("QIANFAN_API_KEY")
if not API_KEY:
    raise EnvironmentError("Please set QIANFAN_API_KEY in your environment or .env file")

# OpenAI-compatible client pointing to 千帆 V2
client = OpenAI(api_key=API_KEY, base_url="https://qianfan.baidubce.com/v2")

# 可调整：千帆模型名（你在控制台可用的模型）
DEFAULT_MODEL = "ernie-speed-8k"   # 或 "ernie-speed-128k", "ernie-4.0-8k" 等

def build_context(retrieved: List[dict], max_chars_each=800):
    parts = []
    for i, r in enumerate(retrieved):
        meta = r.get("metadata", {})
        title = meta.get("title") or meta.get("file") or meta.get("source", "unknown")
        url = meta.get("url", "")
        chunk_text = r.get("chunk", "")
        # 限制长度，避免 prompt 过长
        snippet = chunk_text[:max_chars_each]
        parts.append(f"[{i+1}] 来源: {title}\n链接: {url}\n片段: {snippet}")
    return "\n\n".join(parts)

def generate_answer(user_question: str, top_k: int = 5, model: str = DEFAULT_MODEL, temperature: float = 0.2):
    # 1) Retrieve
    retrieved = search(user_question, top_k=top_k)

    # 2) Build context string
    context = build_context(retrieved)

    # 3) Prompt template：重点改这里
    prompt = f"""
你是一个专业旅行规划顾问，请严格按照以下格式为用户生成旅行方案。

【用户问题】
{user_question}

【检索到的真实游记片段】
{context}

---

⚠️【回答格式要求——必须严格遵守】⚠️

1. 开场：用 1～2 句话概述这趟旅行的风格与适合人群。
   示例： “这是一趟适合第一次来巴黎、喜欢城市漫步与轻松节奏的三日行程。”

2. 分日行程（必须为 Day 1 / Day 2 / Day 3）
   Day 1:
      - 上午：xxx
      - 下午：xxx
      - 晚上：xxx

   Day 2:
      - 上午：xxx
      - 下午：xxx
      - 晚上：xxx

   Day 3:
      - 上午：xxx
      - 下午：xxx
      - 晚上：xxx

   要求：
   - 每个景点尽量基于检索片段
   - 若检索片段不足，可补充通用建议（需说明“补充”）

3. 行程亮点总结（可选但推荐）
   格式示例：
   - 美食亮点：
   - 拍照亮点：
   - 小众体验亮点：

4. 注意事项（至少 3 条）

5. 参考来源（编号 + 标题 + 链接）
   示例：
   [1] 标题：https://xxx
   [2] 标题：https://xxx

请严格按以上格式输出，不要输出其他说明。
"""

    # 4) Call Qianfan (OpenAI-compatible)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是资深旅行规划顾问，回答要务实、礼貌、结构清晰，并且严格遵守用户提供的格式要求。"},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=800
    )

    # 5) Extract text
    try:
        content = resp.choices[0].message["content"]
    except Exception:
        try:
            content = resp.choices[0].text
        except Exception:
            content = str(resp)

    return content, retrieved